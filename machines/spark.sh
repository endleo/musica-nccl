# DGX Sparks: 3 nodes dgx1-3, aarch64, one GB10 each, no Slurm (experiments are started with run-local.sh on dgx1),
# spack. Sourced by config.sh (MACHINE=spark), see there for what a machine file has to define.

# GPU architecture to compile for (GB10)
CUDA_ARCH="121"

# no NCCL module: build.sh builds this checkout into NCCL_HOME (as on Chameleon)
NCCL_TAG="v2.32.3-1"

# not for now: its build needs rust, libclang and protobuf
BUILD_COMMA=false

# no SBATCH_OPTIONS: no Slurm

# OpenMPI from the shared spack (/opt/dgx-shared), the preinstalled CUDA, and NCCL built by build.sh from the
# NCCL_TAG checkout (spack also has an NCCL 2.31.2, not used)
SPACK_ROOT="${SPACK_ROOT:-/opt/dgx-shared/spack/spack_git}"
load_modules() {
    source "$SPACK_ROOT/share/spack/setup-env.sh"
    spack load openmpi@5.0.10
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$CUDA_HOME/bin:$PATH"  # not in the PATH there; cmake finds CUDA by nvcc
    export NCCL_HOME="${NCCL_HOME:-$BUILD_DIR/nccl/build}"
    export MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
    # spack's OpenMPI uses an external PRRTE (~internal-prrte): its daemon prted is in PRRTE's prefix, not in MPI_HOME
    PRRTE_HOME="$(dirname "$(dirname "$(command -v prted)")")"
    # one rank per node (one GPU each). --prefix: PRRTE's daemons (prted) on the other nodes are started from there,
    # the non-interactive ssh shells there do not have spack's packages in their PATH.
    # MPI (startup, barriers of the benchmarks) over the university network (enP7s7): the direct links between the
    # nodes are point-to-point subnets (/etc/hosts), so not every node reaches every address on them.
    # mpirun does not pass the environment on: -x for the paths profiled.sh needs (runtime_library_path)
    # CPUs: mpirun's default binding put each rank on CPU 0 alone, where the benchmark's main thread (busy-waiting
    # on the stream) starved NCCL's proxy thread (ncclbench at ~45 % of nccl-tests' bandwidth, ms per small call).
    # Each rank is pinned to the 10 performance cores (Cortex-X925, 3.9 GHz; 0-4 and 10-14 are the Cortex-A725
    # efficiency cores, lscpu -e) with taskset rather than mpirun's --map-by: lscpu reports the core ids of both
    # clusters twice, so mpirun's per-core mapping may not pick these. The affinity is recorded in job-info.txt.
    SPARK_CPUS="${SPARK_CPUS:-5-9,15-19}"
    LAUNCHER="$MPI_HOME/bin/mpirun --prefix $PRRTE_HOME -np 3 --host dgx1:1,dgx2:1,dgx3:1 --bind-to none
        --mca btl self,tcp --mca btl_tcp_if_include enP7s7 --mca oob_tcp_if_include enP7s7
        -x NCCL_HOME -x CUDA_HOME -x MPI_HOME taskset -c $SPARK_CPUS"
}

list_modules() {
    spack find --loaded 2>&1
}

# NCCL's bootstrap over the university network, which connects all nodes; the data goes over the RoCE HCAs of the
# direct links (4 per node). The links form a ring (each pair of nodes has its own two /30 subnets, no switch), so
# each HCA only reaches one neighbor. By default NCCL picks the HCA by the GPU's locality, not by the peer; with
# subnet-aware routing (NCCL >= 2.32, src/transport/net_ib/connect.cc, default off) the connecting side takes a
# local HCA on the same subnet as the listener's ports ("NET/IB: Subnet-aware routing: overriding dev" in the NET log)
export NCCL_SOCKET_IFNAME=enP7s7
export NCCL_IB_SUBNET_AWARE_ROUTING=1
