#!/bin/bash -l
# Quick check of the setup (< 10 min): ncclsee only, 8 B - 1 MiB, one iteration.
# Submit: ./submit.sh experiments/quick.sbatch
# (DGX Sparks, no Slurm: ./run-local.sh experiments/quick.sbatch)
# Slurm options of the machine (partition, QoS, GPUs per node; MUSICA's comfort mode): submit.sh
#SBATCH --job-name=quick
#SBATCH --array=1
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # without Slurm: run-local.sh
source ./experiment.sh

EXPERIMENT=quick
PROFILERS=(ncclsee)
NCCL_TESTS=(all_reduce broadcast)
NCCL_TESTS_ARGS="-b 8 -e 1M -f 2 -g 1"
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**20))) --data-type float --blocking --csv --warmups 5 --iterations 50"

run_experiment
