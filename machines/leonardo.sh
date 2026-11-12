# Leonardo (CINECA), Booster partition: x86_64, 4 A100 (64 GB, sm_80) per node, Slurm, modules (spack-built,
# environment-modules). Sourced by config.sh (MACHINE=leonardo), see there for what a machine file has to define.
# UNTESTED: neither built nor run yet.

# GPU architecture to compile for (A100)
CUDA_ARCH="80"

# the NCCL module (2.22.3) is too old: build.sh builds this checkout into NCCL_HOME, same version as on the Sparks
NCCL_TAG="v2.32.3-1"

# not for now: there is no protobuf module (CoMMA's build needs rust, libclang and protobuf)
BUILD_COMMA=false

# Slurm options of the machine for submit.sh: Booster nodes, all 4 GPUs. Plain Slurm, so tasks and CPUs are explicit
# (unlike MUSICA's comfort mode): one rank per GPU, the node's 32 cores split among them (more than one core per
# rank matters: bound to a single core, CoMMA's polling thread competes with NCCL's proxy thread)
# QoS: normal by default; boost_qos_dbg has a higher priority but at most 8 nodes, useful for smaller experiments:
#   ./submit.sh --qos=boost_qos_dbg experiments/<experiment>.sbatch
# TODO: --account=<project>, if needed
declare -gA SBATCH_OPTIONS=([partition]=boost_usr_prod [qos]=normal [gres]=gpu:4 [ntasks-per-node]=4
    [cpus-per-task]=8)

# the home directory has a 50 GB quota, too small for the profiler outputs of large runs: results in $WORK (the
# project's area, writable from the compute nodes). fetch.sh asks the cluster for this path
[ -n "$WORK" ] || { echo "WORK is not set (machines/leonardo.sh: RESULTS_DIR)" >&2; return 1; }
RESULTS_DIR="$WORK/musica-nccl/results"

load_modules() {
    module purge
    module load profile/base     # probably not necessary (not affected by purge), but just to be safe
    module load gcc/12.2.0       # the system GCC (RHEL 8) is 8.5, too old for ncclsee-cpp
    module load cmake/4.1.2
    module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
    module load cuda/12.6        # after openmpi, which was built against cuda 12.2 (and might load it)

    # derived from nvcc/mpicc; set them here if that fails. Exported, since profiled.sh needs them on every rank
    # (see runtime_library_path)
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
    export NCCL_HOME="${NCCL_HOME:-$BUILD_DIR/nccl/build}"

    # launcher for the benchmarks inside the job, one process per task (as on MUSICA). Slurm's default MPI plugin is
    # pmix_v3. --gpu-bind=none: both benchmarks pick their GPU by the local rank among all visible GPUs, so every
    # rank has to see all GPUs of its node
    LAUNCHER="srun --cpu-bind=verbose --gpu-bind=none"
}

list_modules() {
    module list 2>&1
}
