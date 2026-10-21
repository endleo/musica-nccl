#!/bin/bash -l
#ASC --vanilla
# The full experiment (as on Chameleon): every benchmark once per profiler, 8 B - 1 GiB.
# Submit from the repository root: sbatch experiments/full.sbatch
# Each array task is one iteration of the whole experiment (on whatever nodes the task gets);
# --array=1-3%1: 3 iterations, at most 1 running at a time. Other count: sbatch --array=1-5%1 experiments/full.sbatch
# the ASC directive above: vanilla mode, i.e. plain Slurm on MUSICA, the options below apply as written (see README)
#SBATCH --job-name=full
#SBATCH --array=1-3%1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# More than one core per rank: bound to a single core, CoMMA's polling thread competes with NCCL's proxy thread
# (suspected cause of the slow CoMMA runs on Chameleon)
#SBATCH --cpus-per-task=8
# all 4 GPUs per node, visible to every rank; do not bind one GPU
# per task (--gpus-per-task), both benchmarks pick their GPU by the local rank among all GPUs of the node
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:4

cd "$SLURM_SUBMIT_DIR"
source ./experiment.sh

EXPERIMENT=full
# no Inspector with NCCL 2.27 (see NCCL_TAG in config.sh)
PROFILERS=(comma ncclsee)
NCCL_TESTS=(all_reduce broadcast)
NCCL_TESTS_ARGS="-b 8 -e 1G -f 2 -g 1"
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**30))) --data-type float --blocking --csv --warmups 5 --iterations 50"

run_experiment
