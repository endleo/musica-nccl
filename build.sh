#!/bin/bash -l
#ASC --vanilla
# Fetches and builds nccl-tests, lib-ncclbench and the profiler plugins into build/ (versions in config.sh),
# against the NCCL, CUDA and MPI of the modules. Run from the repository root:
#   ./build.sh                  fetch + compile (on a login node)
#   ./build.sh fetch            only download sources, rust and crates (needs internet)
#   sbatch build.sh compile     compile on a compute node (after fetch, e.g. if the login node has no nvcc)
#   ./build.sh clean            remove the build outputs (keeps sources, rust and crates), e.g. after changing modules:
#                               make and cmake do not notice that, and cmake keeps the CUDA it found in its cache
# `#ASC --vanilla`: plain Slurm on MUSICA, i.e. the options below apply as written (see README)
#SBATCH --job-name=musica-nccl-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/build-%j.out
#SBATCH --partition=zen4_0768
#SBATCH --account=p201270
#SBATCH --qos=zen4_0768

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
    # its example (the ncclbench binary) would download yaml-cpp while configuring; cloned here instead, so that
    # compile needs no internet (same tag as in lib-ncclbench/example/CMakeLists.txt)
    clone https://github.com/jbeder/yaml-cpp.git yaml-cpp 0.8.0

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

    # CMAKE_POLICY_VERSION_MINIMUM: yaml-cpp 0.8.0 requires CMake < 3.5, which CMake 4 no longer accepts
    cmake -S lib-ncclbench -B lib-ncclbench/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON \
        -DNCCL_LIBRARIES="$NCCL_HOME/lib/libnccl.so" -DNCCL_INCLUDE_DIRS="$NCCL_HOME/include" \
        -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="$BUILD_DIR/yaml-cpp" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
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

    # bindgen (clang-sys) would take the libclang with the highest version anywhere on the system (on MUSICA an
    # incomplete /usr/lib64/llvm21, which lacks libLLVM); LIBCLANG_PATH restricts it to the loaded LLVM module.
    # --locked: fail instead of resolving other crate versions than in Cargo.lock
    (cd CoMMA && LIBCLANG_PATH="$(llvm-config --libdir)" "$CARGO_HOME/bin/cargo" build --release --locked --jobs "$jobs")

    local outputs=("$NCCL_TESTS_BIN"/{all_reduce,broadcast}_perf "$NCCLBENCH_BIN" "$COMMA_PLUGIN" "$NCCLSEE_PLUGIN")
    ls -l "${outputs[@]}"
    # libraries that cannot be found at runtime (with the library path of profiled.sh), e.g. outputs of an earlier
    # build against another CUDA
    if LD_LIBRARY_PATH="$(runtime_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "${outputs[@]}" | awk '/^\// {file = $0} /not found/ {print file, $0; missing = 1} END {exit !missing}'; then
        echo "WARNING: missing libraries (see above); after changing modules: ./build.sh clean && ./build.sh compile" >&2
    fi
}

clean() {
    cd "$BUILD_DIR"
    rm -rf nccl-tests/build lib-ncclbench/build ncclsee-cpp/build CoMMA/target
    [ ! -d nccl/plugins/profiler/inspector ] || make -C nccl/plugins/profiler/inspector clean
}

case "${1:-all}" in
    fetch) fetch ;;
    compile) compile ;;
    all) fetch; compile ;;
    clean) clean ;;
    *) echo "usage: $0 [fetch|compile|all|clean]" >&2; exit 1 ;;
esac
