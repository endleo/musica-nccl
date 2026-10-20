#!/bin/bash -l
# Fetches and builds nccl-tests, lib-ncclbench and the profiler plugins into build/ (versions in config.sh),
# against the NCCL, CUDA and MPI of the modules. Run from the repository root:
#   ./build.sh                  fetch + compile (on a login node)
#   ./build.sh fetch            only download sources, rust and crates (needs internet)
#   sbatch build.sh compile     compile on a compute node (after fetch, e.g. if the login node has no nvcc)
#SBATCH --job-name=musica-nccl-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/build-%j.out
# TODO: fill in
#SBATCH --partition=
#SBATCH --account=

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./config.sh

clone() {  # clone <url> <directory> <commit or tag>
    [ -d "$2" ] || git clone "$1" "$2"
    git -C "$2" checkout --quiet "$3"
}

fetch() {
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    clone https://github.com/NVIDIA/nccl.git nccl "$NCCL_TAG"
    clone https://github.com/NVIDIA/nccl-tests.git nccl-tests "$NCCL_TESTS_TAG"
    clone https://github.com/parlab-tuwien/lib-ncclbench lib-ncclbench "$NCCLBENCH_COMMIT"

    # ncclsee-cpp bundles its own profiler headers; submodules: spdlog, nlohmann/json, range-v3
    clone https://github.com/ruben-laso/ncclsee-cpp.git ncclsee-cpp "$NCCLSEE_COMMIT"
    git -C ncclsee-cpp submodule update --init --recursive

    # CoMMA reads the NCCL profiler headers from its nccl submodule; point it at our NCCL checkout instead.
    # Its repo has no Cargo.lock; ours is from a working build of COMMA_COMMIT (same as on Chameleon)
    clone https://github.com/google/CoMMA.git CoMMA "$COMMA_COMMIT"
    [ -L CoMMA/third_party/nccl ] || { rmdir CoMMA/third_party/nccl && ln -s "$BUILD_DIR/nccl" CoMMA/third_party/nccl; }
    cp "$REPO_DIR/files/comma/Cargo.lock" CoMMA/Cargo.lock
    # CoMMA needs rust >= 1.80
    [ -x "$CARGO_HOME/bin/cargo" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
    (cd CoMMA && "$CARGO_HOME/bin/cargo" fetch --locked)
}

compile() {
    load_modules
    [ -n "$CUDA_ARCH" ] || { echo "CUDA_ARCH is not set (config.sh)" >&2; exit 1; }
    [ -f "$NCCL_HOME/include/nccl.h" ] || { echo "no nccl.h in NCCL_HOME=$NCCL_HOME" >&2; exit 1; }
    echo "NCCL $(nccl_version) from $NCCL_HOME, sources for the plugins: $NCCL_TAG"
    check_nccl_version
    echo "CUDA from $CUDA_HOME, MPI from $MPI_HOME"
    local jobs=${SLURM_CPUS_PER_TASK:-8}
    cd "$BUILD_DIR"

    make -C nccl-tests -j "$jobs" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME="$NCCL_HOME" CUDA_HOME="$CUDA_HOME" \
        NVCC_GENCODE="-gencode=arch=compute_$CUDA_ARCH,code=sm_$CUDA_ARCH"

    cmake -S lib-ncclbench -B lib-ncclbench/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON \
        -DNCCL_LIBRARIES="$NCCL_HOME/lib/libnccl.so" -DNCCL_INCLUDE_DIRS="$NCCL_HOME/include"
    cmake --build lib-ncclbench/build --parallel "$jobs"

    # the Inspector is part of the NCCL repo (since v2.29.7) and has a standalone makefile
    if [ -d nccl/plugins/profiler/inspector ]; then
        make -C nccl/plugins/profiler/inspector -j "$jobs" CUDA_HOME="$CUDA_HOME"
    else
        echo "Skipping the Inspector: not in NCCL $NCCL_TAG"
    fi

    cmake -S ncclsee-cpp -B ncclsee-cpp/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    cmake --build ncclsee-cpp/build --parallel "$jobs"

    # --locked: fail instead of resolving other crate versions than in Cargo.lock
    (cd CoMMA && "$CARGO_HOME/bin/cargo" build --release --locked --jobs "$jobs")

    ls -l "$NCCL_TESTS_BIN"/{all_reduce,broadcast}_perf "$NCCLBENCH_BIN" "$COMMA_PLUGIN" "$NCCLSEE_PLUGIN"
}

case "${1:-all}" in
    fetch) fetch ;;
    compile) compile ;;
    all) fetch; compile ;;
    *) echo "usage: $0 [fetch|compile|all]" >&2; exit 1 ;;
esac
