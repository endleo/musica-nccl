# Shared settings for build.sh and run.sbatch (sourced, not executed).

# ---------------------------------------------------------------------------------------------
# environment (TODO: fill in)
# ---------------------------------------------------------------------------------------------

load_modules() {
    module purge
    # TODO: CUDA, NCCL, MPI, cmake; for CoMMA also clang/libclang (bindgen) and protobuf (protoc)
    # module load ...

    # taken from the modules if they set them, otherwise derived from nvcc/mpicc; set them here if that fails
    NCCL_HOME="${NCCL_HOME:-${NCCL_ROOT:-}}"
    CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
    export LD_LIBRARY_PATH="$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

# version of the NCCL in NCCL_HOME, e.g. 2.32.3
nccl_version() {
    grep -E '^#define NCCL_(MAJOR|MINOR|PATCH) ' "$NCCL_HOME/include/nccl.h" | awk '{print $3}' | paste -sd.
}

# GPU architecture to compile for, e.g. 90 for H100 (TODO: fill in)
CUDA_ARCH=""

# launcher for the benchmarks inside the job, one process per task
LAUNCHER="srun --cpu-bind=verbose"
# alternative if srun has no matching PMI support for the MPI module:
# LAUNCHER="mpirun"

# ---------------------------------------------------------------------------------------------
# versions (same as the Chameleon experiment)
# ---------------------------------------------------------------------------------------------

# NCCL itself comes from the module; this checkout is only used for the sources of the Inspector
# plugin and the profiler headers CoMMA builds against -> should match the module's version
NCCL_TAG="v2.32.3-1"
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
# experiment
# ---------------------------------------------------------------------------------------------

PROFILERS=(comma inspector ncclsee)
NCCL_TESTS=(all_reduce broadcast)                                             # nccl-tests binaries <test>_perf
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)

NCCL_TESTS_ARGS="-b 8 -e 1G -f 2 -g 1"
NCCLBENCH_ARGS="--sizes $(for i in $(seq 3 30); do printf '%s ' $((2**i)); done)--data-type float --blocking --csv --warmups 5 --iterations 50"

# environment of every run. Workaround for an NCCL bug (since v2.29.2) that hides plain broadcasts from
# profiler plugins (see notes/nccl-broadcast-profiler-events.md in the Chameleon project)
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
