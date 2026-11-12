# Shared settings for build.sh and the experiments (sourced, not executed): environment, versions, paths,
# profiler configuration. The experiment parameters (sizes, profilers, ...) are in experiments/*.sbatch.

# ---------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------

# The machine: musica (x86_64, H100, Slurm, modules), leonardo (x86_64, A100, Slurm, modules) or spark (3 DGX Sparks
# dgx1-3, aarch64, GB10, no Slurm, spack).
# Its settings are in machines/<machine>.sh (sourced at the end of this file), which defines:
#   CUDA_ARCH        GPU architecture to compile for, e.g. 90
#   NCCL_TAG         NCCL checkout: the profiler headers, and NCCL itself where build.sh builds it
#   BUILD_COMMA      true: build.sh fetches and builds CoMMA
#   SBATCH_OPTIONS   only with Slurm: associative array <sbatch option> -> value for submit.sh (partition, ...)
#   load_modules     loads the environment; sets NCCL_HOME, CUDA_HOME, MPI_HOME (exported) and LAUNCHER, the
#                    launcher of the benchmarks (one process per rank). build.sh builds NCCL if NCCL_HOME is
#                    build/nccl/build
#   list_modules     prints the loaded environment, for job-info.txt
#   RESULTS_DIR      optional, default results/ in the clone: where the experiments write (e.g. outside a small
#                    home directory); on a filesystem the compute nodes can write to. fetch.sh copies from there
# and anything else of the machine (e.g. NCCL network variables). A new machine: a new file, and its detection in
# detect_machine below.
# Set MACHINE to override the detection: by Slurm's cluster name (in jobs SLURM_CLUSTER_NAME, on the login nodes only
# in the Slurm config), without Slurm by the CPU architecture
detect_machine() {
    local cluster
    cluster="${SLURM_CLUSTER_NAME:-$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" {print $3}')}"
    case $cluster in
        vie-musica) echo musica ;;
        leonardo) echo leonardo ;;
        "")
            if [ "$(uname -m)" = aarch64 ]; then
                echo spark
            else
                echo "no Slurm cluster and not aarch64: unknown machine, set MACHINE (config.sh)" >&2; return 1
            fi
            ;;
        *) echo "unknown Slurm cluster $cluster: add it to detect_machine (config.sh) or set MACHINE" >&2; return 1 ;;
    esac
}
MACHINE="${MACHINE:-$(detect_machine)}" || return 1

# Library directories our binaries need at runtime. The modules on MUSICA come from EESSI (/cvmfs/software.eessi.io),
# which does not set LD_LIBRARY_PATH: EESSI's own software finds its libraries via RUNPATH, ours does not. That
# includes libstdc++ of the module's GCC (otherwise EESSI's older system libstdc++ is found: "CXXABI_1.3.15 not
# found"). Only set for the benchmark processes (profiled.sh) and the check in build.sh, not for the whole job:
# EESSI's libraries need EESSI's glibc and might break system programs.
runtime_library_path() {
    echo "$NCCL_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib${EBROOTGCCCORE:+:$EBROOTGCCCORE/lib64}"
}

# the environment, sorted, for the results (job-info.txt, env-rank<N>.txt): without exported shell functions and
# without variables whose name looks like a credential (TOKEN, SECRET, PASSWORD, KEY)
print_environment() {
    env -0 | grep -zv '^BASH_FUNC_' | grep -zviE '^[^=]*(TOKEN|SECRET|PASSW|CREDENTIAL|KEY)[^=]*=' \
        | sort -z | tr '\0' '\n'
}

# version of the NCCL in NCCL_HOME, e.g. 2.27.7
nccl_version() {
    grep -E '^#define NCCL_(MAJOR|MINOR|PATCH) ' "$NCCL_HOME/include/nccl.h" | awk '{print $3}' | paste -sd.
}

# warns if the loaded NCCL is not the version of NCCL_TAG (whose headers the profilers are built against)
check_nccl_version() {
    local loaded tag
    loaded=$(nccl_version)
    tag=${NCCL_TAG#v}
    tag=${tag%-*}
    if [ "$loaded" != "$tag" ]; then
        echo "WARNING: loaded NCCL is ${loaded:-unknown} ($NCCL_HOME), but NCCL_TAG is $NCCL_TAG" >&2
    fi
}

# ---------------------------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------------------------

# NCCL_TAG: per machine (machines/*.sh)
NCCL_TESTS_TAG="v2.19.6"
NCCLBENCH_COMMIT="3b9d2c4ffb73a486fcae4eb3f881ad6efe93f3fc"
COMMA_COMMIT="1d5b9676f66c1ab560bccd69f13745c11f0437a5"
NCCLSEE_COMMIT="0e8433a340e68e71cf195ca040b75db0cb48d004"
RUST_TOOLCHAIN="1.99.0"

# ---------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sources and builds; has to be on a filesystem the compute nodes can read
BUILD_DIR="$REPO_DIR/build"
RESULTS_DIR="$REPO_DIR/results"

# rust is installed into the build directory instead of ~/.cargo
export RUSTUP_HOME="$BUILD_DIR/rustup"
export CARGO_HOME="$BUILD_DIR/cargo"

NCCL_TESTS_BIN="$BUILD_DIR/nccl-tests/build"
NCCLBENCH_BIN="$BUILD_DIR/lib-ncclbench/build/example/ncclbench"
COMMA_PLUGIN="$BUILD_DIR/CoMMA/target/release/libnccl_profiler.so"
INSPECTOR_PLUGIN="$BUILD_DIR/nccl/plugins/profiler/inspector/libnccl-profiler-inspector.so"
NCCLSEE_PLUGIN="$BUILD_DIR/ncclsee-cpp/build/libncclsee.so"

# ---------------------------------------------------------------------------------------------
# profilers
# ---------------------------------------------------------------------------------------------

# environment of every run. Workaround for an NCCL bug (since v2.29.2) that hides plain broadcasts from
# profiler plugins (see notes/nccl-broadcast-profiler-events.md in the Chameleon project); NCCL 2.27 does not
# have this variable and ignores it
export NCCL_ALLGATHERV_ENABLE=0

# plugin of a profiler (empty if unknown)
profiler_plugin() {
    case $1 in
        comma) echo "$COMMA_PLUGIN" ;;
        inspector) echo "$INSPECTOR_PLUGIN" ;;
        ncclsee) echo "$NCCLSEE_PLUGIN" ;;
    esac
}

# environment per profiler: profiler_env <profiler> <output directory of this process (per node)>
profiler_env() {
    local profiler=$1 out=$2
    case $profiler in
        comma)
            export NCCL_PROFILER_PLUGIN="$COMMA_PLUGIN"
            export NCCL_PROFILER_USE_GPUVIZ=false                # GCP-only telemetry backend
            export NCCL_PROFILER_LATENCY_FILE="$out/latency-%p.txt"
            export NCCL_PROFILER_SKIP_SMALL_COLLECTIVE=false     # default skips non-allreduce collectives <= 64 KiB
            ;;
        inspector)
            export NCCL_PROFILER_PLUGIN="$INSPECTOR_PLUGIN"
            export NCCL_INSPECTOR_ENABLE=1
            export NCCL_INSPECTOR_DUMP_DIR="$out"
            # default (-1) only dumps at comm teardown, so older entries get overwritten in the per-comm ring buffer
            export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
            export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=0          # default only records ops >= 8 KiB
            ;;
        ncclsee)
            export NCCL_PROFILER_PLUGIN="$NCCLSEE_PLUGIN"
            export NCCLSEE_MODE=tracer                           # json trace of every event (needed for per-op times)
            export NCCLSEE_OUTPUT_DIR="$out"
            ;;
        *)
            echo "unknown profiler $profiler" >&2; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------------------------
# NCCL logging (experiments with LOG_NCCL=true, e.g. experiments/logging.sbatch)
# ---------------------------------------------------------------------------------------------

# NCCL's debug log and its topology/graph dumps: nccl_logging_env <output directory of this process> <rank>
# Subsystems (NCCL 2.27, src/debug.cc; setting NCCL_DEBUG_SUBSYS replaces the default INIT,BOOTSTRAP,ENV):
#   INIT    version, network, plugin loading (tuner, profiler), rings/trees, connection setup
#   ENV     every NCCL_* variable NCCL reads that is set (also from /etc/nccl.conf, ~/.nccl.conf)
#   GRAPH   topology search, channel layout
#   TUNING  rank 0: latency/bandwidth table of the cost model at init, and one line per call with the chosen
#           algorithm, protocol and channels (src/enqueue.cc). The per-call line is written while planning the
#           launch, so it adds to the host time of every call (small sizes: a few us); it goes to a file (buffered,
#           unlike stderr) to keep that low, but the times of such a run are not comparable to runs without it
#   NET     network devices, GDR
# Every rank writes its own log; the dumps are written by rank 0 only (NCCL_*_DUMP_FILE_RANK, default 0).
nccl_logging_env() {
    local out=$1 rank=$2
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING,NET
    export NCCL_DEBUG_FILE="$out/nccl-rank$rank.log"
    export NCCL_TOPO_DUMP_FILE="$out/topo-rank$rank.xml"
    export NCCL_GRAPH_DUMP_FILE="$out/graph-rank$rank.xml"
}

# ---------------------------------------------------------------------------------------------
# machine
# ---------------------------------------------------------------------------------------------

# last, so that a machine file can use and override everything above
MACHINE_FILE="$REPO_DIR/machines/$MACHINE.sh"
[ -f "$MACHINE_FILE" ] || { echo "no machine file $MACHINE_FILE (MACHINE=$MACHINE)" >&2; return 1; }
source "$MACHINE_FILE" || return 1
for var in CUDA_ARCH NCCL_TAG BUILD_COMMA; do
    [ -n "${!var}" ] || { echo "$MACHINE_FILE does not set $var" >&2; return 1; }
done
for fn in load_modules list_modules; do
    declare -F "$fn" >/dev/null || { echo "$MACHINE_FILE does not define $fn" >&2; return 1; }
done
