# Shared settings for build.sh and the experiments (sourced, not executed): environment, versions, paths,
# profiler configuration. The experiment parameters (sizes, profilers, ...) are in experiments/*.sbatch.

# ---------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------

load_modules() {
    module purge
    module load NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1
    module load gompi/2025b
    module load protobuf/31.1-GCCcore-14.3.0
    module load LLVM/20.1.8-GCCcore-14.3.0
    module load CUDA/12.9.1

    # taken from the modules if they set them, otherwise derived from nvcc/mpicc; set them here if that fails.
    # Exported, since profiled.sh needs them on every rank (see runtime_library_path)
    export NCCL_HOME="${NCCL_HOME:-${NCCL_ROOT:-${EBROOTNCCL:-}}}"  # EBROOTNCCL: set by EasyBuild modules
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
}

# Library directories our binaries need at runtime. The modules on MUSICA come from EESSI (/cvmfs/software.eessi.io),
# which does not set LD_LIBRARY_PATH: EESSI's own software finds its libraries via RUNPATH, ours does not. That
# includes libstdc++ of the module's GCC (otherwise EESSI's older system libstdc++ is found: "CXXABI_1.3.15 not
# found"). Only set for the benchmark processes (profiled.sh) and the check in build.sh, not for the whole job:
# EESSI's libraries need EESSI's glibc and might break system programs.
runtime_library_path() {
    echo "$NCCL_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib${EBROOTGCCCORE:+:$EBROOTGCCCORE/lib64}"
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

# GPU architecture to compile for, e.g. 90 for H100
CUDA_ARCH="90"

# launcher for the benchmarks inside the job, one process per task
# --gpu-bind=none: every rank sees all GPUs of its node; both benchmarks pick their GPU by the local rank among all
# visible GPUs, so with one GPU per rank (e.g. MUSICA's comfort mode) ranks 1-3 fail ("invalid device ordinal")
LAUNCHER="srun --cpu-bind=verbose --gpu-bind=none"
# alternative if srun has no matching PMI support for the MPI module:
# LAUNCHER="mpirun"

# ---------------------------------------------------------------------------------------------
# versions (same as the Chameleon experiment)
# ---------------------------------------------------------------------------------------------

# NCCL itself comes from the module; this checkout is only used for the profiler headers CoMMA builds
# against (and the sources of the Inspector, which only exists since v2.29.7) -> has to match the module's version.
# NCCL 2.27 loads profiler plugins up to API v4: fine for CoMMA and ncclsee (v4), but the Inspector (v5 only)
# cannot be used with it, so build.sh skips it.
NCCL_TAG="v2.27.7-1"
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
