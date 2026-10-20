#!/bin/bash -l
# Quick check of the setup (< 10 min): ncclsee only, 8 B - 1 MiB, one iteration.
# Submit from the repository root: sbatch experiments/quick.sbatch
#SBATCH --job-name=quick
#SBATCH --array=1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --cpus-per-task=8
# TODO: fill in (as in full.sbatch)
#SBATCH --partition=
#SBATCH --account=
#SBATCH --gres=

cd "$SLURM_SUBMIT_DIR"
source ./experiment.sh

EXPERIMENT=quick
PROFILERS=(ncclsee)
NCCL_TESTS=(all_reduce broadcast)
NCCL_TESTS_ARGS="-b 8 -e 1M -f 2 -g 1"
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**20))) --data-type float --blocking --csv --warmups 5 --iterations 50"

run_experiment
