#!/bin/bash -l
# The full experiment (as on Chameleon): every benchmark once per profiler, 8 B - 1 GiB.
# Submit from the repository root: sbatch experiments/full.sbatch
# Each array task is one iteration of the whole experiment (on whatever nodes the task gets);
# --array=1-3%1: 3 iterations, at most 1 running at a time. Other count: sbatch --array=1-5%1 experiments/full.sbatch
#
# Same Slurm options as the working quick run (job 1863239): MUSICA's comfort mode (the default without the ASC
# vanilla directive, see README), only nodes and GPUs are requested, MUSICA sets the rest. For full GPU nodes
# (-N X --gres=gpu:4) that is 4 tasks per node, 22 CPUs per task (more than one core per rank matters: bound to a
# single core, CoMMA's polling thread competes with NCCL's proxy thread), exclusive nodes, all memory. Every rank
# sees all GPUs of its node via --gpu-bind=none in LAUNCHER (config.sh).
# TODO (TODO.md): switch to the explicit options and vanilla mode
# Do not write the vanilla directive in comments here: MUSICA finds it anywhere in the file and enables vanilla mode
#SBATCH --job-name=full
#SBATCH --array=1-3%1
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4

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
