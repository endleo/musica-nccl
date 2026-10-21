#!/bin/bash -l
# Quick check of the setup (< 10 min): ncclsee only, 8 B - 1 MiB, one iteration.
# Submit from the repository root: sbatch experiments/quick.sbatch
#
# Uses MUSICA's comfort mode (the default without the ASC vanilla directive, see README): only nodes and GPUs are
# requested, MUSICA sets the rest. For full GPU nodes (-N X --gres=gpu:4) that is 4 tasks per node, 22 CPUs per
# task, exclusive nodes, all memory. The applied settings are in the job output and in job-info.txt.
# TODO (TODO.md): switch to the explicit options and vanilla mode, as in full.sbatch
# Do not write the vanilla directive in comments here: MUSICA finds it anywhere in the file and enables vanilla mode
#SBATCH --job-name=quick
#SBATCH --array=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4

cd "$SLURM_SUBMIT_DIR"
source ./experiment.sh

EXPERIMENT=quick
PROFILERS=(ncclsee)
NCCL_TESTS=(all_reduce broadcast)
NCCL_TESTS_ARGS="-b 8 -e 1M -f 2 -g 1"
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**20))) --data-type float --blocking --csv --warmups 5 --iterations 50"

run_experiment
