#!/bin/bash -l
# Logging experiment: what NCCL decides and why, from its own logs. ncclbench with ncclsee (to compare NCCL's
# per-call log with what the profiler records), 8 B - 1 GiB, one iteration, with NCCL's debug log for INIT, ENV,
# GRAPH, TUNING and NET and the topology/graph dumps (nccl_logging_env in config.sh).
# Submit from the repository root: sbatch experiments/logging.sbatch
#
# The TUNING log writes a line per call on rank 0, which adds host time to every call: the times of this run are
# not comparable to the other experiments.
# Slurm options as in full.sbatch (MUSICA's comfort mode, see there)
#SBATCH --job-name=logging
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4

cd "$SLURM_SUBMIT_DIR"
source ./experiment.sh

EXPERIMENT=logging
PROFILERS=(ncclsee)
NCCL_TESTS=()
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**30))) --data-type float --blocking --csv --warmups 5 --iterations 50"
LOG_NCCL=true

run_experiment
