#!/bin/bash -l
# Tuning experiment: NCCL's own choice ("default", no NCCL_ALGO/NCCL_PROTO) against pinned algorithms/protocols.
# ncclbench with ncclsee (to check per call which algorithm/protocol was actually used), 8 B - 1 GiB. All variants
# run in one job on the same nodes, in 3 rounds, each round runs every variant once (order rotated per round, see
# experiment.sh); results in results/tuning/<job>/<round>/<variant>/.
# Submit from the repository root: sbatch experiments/tuning.sbatch
#
# Slurm options as in full.sbatch (MUSICA's comfort mode, see there)
#SBATCH --job-name=tuning
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4

cd "$SLURM_SUBMIT_DIR"
source ./experiment.sh

EXPERIMENT=tuning
PROFILERS=(ncclsee)
NCCL_TESTS=()
declare -A NCCLBENCH_TESTS=([all_reduce_bench]=ncclAllReduce [broadcast_bench]=ncclBroadcast)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**30))) --data-type float --blocking --csv --warmups 5 --iterations 50"
ROUNDS=3

# Placeholders until the logging experiment is evaluated (TODO.md). Syntax (NCCL 2.27, src/graph/tuning.cc):
# "<list>" for all operations, "<op>:<list>" for one operation, entries separated by ";", "^" excludes, e.g.
# NCCL_PROTO="^LL128;allreduce:LL128". Pin per operation: a plain NCCL_ALGO=Tree fails for Broadcast (Ring only).
# A pinned algorithm leaves the protocol and the channels to NCCL, and vice versa.
# Algorithms: Tree Ring CollNetDirect CollNetChain NVLS NVLSTree PAT; protocols: LL LL128 Simple
VARIANTS=(default allreduce_ring allreduce_tree)
declare -A VARIANT_ENV=(
    [allreduce_ring]="NCCL_ALGO=allreduce:ring"
    [allreduce_tree]="NCCL_ALGO=allreduce:tree"
)

run_experiment
