# MUSICA (ASC Vienna): x86_64, 4 H100 per node, Slurm, modules (EESSI). Sourced by config.sh (MACHINE=musica),
# see there for what a machine file has to define.

# GPU architecture to compile for (H100)
CUDA_ARCH="90"

# NCCL itself comes from the module; this checkout is only used for the profiler headers CoMMA builds against (and
# the sources of the Inspector, which only exists since v2.29.7) -> has to match the module's version.
# NCCL 2.27 loads profiler plugins up to API v4: fine for CoMMA and ncclsee (v4), but the Inspector (v5 only) cannot
# be used with it, so build.sh skips it.
NCCL_TAG="v2.27.7-1"

BUILD_COMMA=true

# Slurm options of the machine for submit.sh (each can be replaced there, e.g. --qos=dev_zen4_0768_h100x4 for short
# test runs). GPU nodes, see https://docs.asc.ac.at/musica/musica_queues.html.
# MUSICA's comfort mode (the default without the ASC vanilla directive in the batch file, see README): only nodes and
# GPUs are requested, MUSICA sets the rest. For full GPU nodes (-N X --gres=gpu:4) it chooses (scontrol show job in
# job-info.txt, e.g. quick runs 1863239 and 1938071):
#   - 4 tasks per node, 22 CPUs per task (more than one core per rank matters: bound to a single core, CoMMA's
#     polling thread competes with NCCL's proxy thread)
#   - one GPU per task (TresPerTask=gres/gpu=1, TresBind=gres/gpu:single:1); overridden by --gpu-bind=none in
#     LAUNCHER, so that every rank sees all GPUs of its node
#   - exclusive nodes, all memory
# MUSICA finds the vanilla directive anywhere in a batch file, even in a comment, so the batch files must not mention
# it literally while in comfort mode
declare -gA SBATCH_OPTIONS=([partition]=zen4_0768_h100x4 [qos]=zen4_0768_h100x4 [gres]=gpu:4)

NCCL_MODULE=NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1

load_modules() {
    module purge
    module load "$NCCL_MODULE"
    module load gompi/2025b
    module load protobuf/31.1-GCCcore-14.3.0
    module load LLVM/20.1.8-GCCcore-14.3.0
    module load CUDA/12.9.1

    # taken from the modules if they set them, otherwise derived from nvcc/mpicc; set them here if that fails.
    # Exported, since profiled.sh needs them on every rank (see runtime_library_path)
    export NCCL_HOME="${NCCL_HOME:-${NCCL_ROOT:-${EBROOTNCCL:-}}}"  # EBROOTNCCL: set by EasyBuild modules
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"

    # launcher for the benchmarks inside the job, one process per task.
    # --gpu-bind=none: every rank sees all GPUs of its node; both benchmarks pick their GPU by the local rank among
    # all visible GPUs, so with one GPU per rank (e.g. MUSICA's comfort mode) ranks 1-3 fail ("invalid device ordinal")
    LAUNCHER="srun --cpu-bind=verbose --gpu-bind=none"
    # alternative if srun has no matching PMI support for the MPI module:
    # LAUNCHER="mpirun"
}

list_modules() {
    module list 2>&1
    # what the NCCL module sets besides the paths (e.g. NCCL_* variables)
    echo; echo "module show $NCCL_MODULE:"
    module show "$NCCL_MODULE" 2>&1
}
