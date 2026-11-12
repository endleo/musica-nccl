#!/bin/bash -l
# The full experiment (as on Chameleon): every benchmark once per profiler, 8 B - 1 GiB.
# Submit: ./submit.sh experiments/full.sbatch
# (DGX Sparks, no Slurm: ./run-local.sh experiments/full.sbatch)
# Each array task is one iteration of the whole experiment (on whatever nodes the task gets);
# --array=1-3%1: 3 iterations, at most 1 running at a time.
# Other count: ./submit.sh --array=1-5%1 experiments/full.sbatch
# Slurm options of the machine (partition, QoS, GPUs per node; MUSICA's comfort mode): submit.sh
#SBATCH --job-name=full
#SBATCH --array=1-3%1
#SBATCH --nodes=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # without Slurm: run-local.sh
source ./experiment.sh

EXPERIMENT=full
# no Inspector with NCCL 2.27 (see NCCL_TAG in config.sh)
PROFILERS=(comma ncclsee)
NCCL_TESTS=(all_reduce broadcast)
NCCL_TESTS_ARGS="-b 8 -e 1G -f 2 -g 1"
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**30))) --data-type float --blocking --csv --warmups 5 --iterations 50"

run_experiment
