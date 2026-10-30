#!/bin/bash -l
# Tuning experiment: NCCL's own choice ("default", no NCCL_ALGO/NCCL_PROTO) against every available
# algorithm/protocol combination, pinned one at a time (measured version of NCCL's cost table).
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
# powers of two, plus 1.5 and 3 MiB: the model predicts Ring instead of Tree for AllReduce from about 1.1 MB to 4 MiB
# (logging run 1889653, report/logging/summary.typ)
NCCLBENCH_ARGS="--sizes $(powers_of_two 8 $((2**20))) 1572864 $((2**21)) 3145728 $(powers_of_two $((2**22)) $((2**30)))"
NCCLBENCH_ARGS+=" --data-type float --blocking --csv --warmups 5 --iterations 50"
ROUNDS=3

# Syntax (NCCL 2.27, src/graph/tuning.cc): "<list>" for all operations, "<op>:<list>" for one operation (the others
# keep their default), entries separated by ";", "^" excludes, e.g. NCCL_PROTO="^LL128;allreduce:LL128". Pin per
# operation: a plain NCCL_ALGO=Tree fails for Broadcast (Ring only). A combination is only allowed if both
# NCCL_ALGO and NCCL_PROTO allow it, so setting both leaves exactly one; the channels stay NCCL's choice.
# Algorithms: Tree Ring CollNetDirect CollNetChain NVLS NVLSTree PAT; protocols: LL LL128 Simple
# Available on MUSICA (TUNING table of run 1889653, everything else 0): AllReduce Tree and Ring, Broadcast Ring, each
# with LL, LL128 and Simple. LL128 is enabled by default there, so pinning it does not override NCCL's hardware check.
# Every benchmark is its own process, so the Ring variants also pin Broadcast's protocol; in the Tree variants
# Broadcast runs with NCCL's choice (extra default samples).
VARIANTS=(default tree_ll tree_ll128 tree_simple ring_ll ring_ll128 ring_simple)
declare -A VARIANT_ENV=(
    [tree_ll]="NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL"
    [tree_ll128]="NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL128"
    [tree_simple]="NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:Simple"
    [ring_ll]="NCCL_ALGO=allreduce:ring NCCL_PROTO=allreduce:LL;broadcast:LL"
    [ring_ll128]="NCCL_ALGO=allreduce:ring NCCL_PROTO=allreduce:LL128;broadcast:LL128"
    [ring_simple]="NCCL_ALGO=allreduce:ring NCCL_PROTO=allreduce:Simple;broadcast:Simple"
)

run_experiment
