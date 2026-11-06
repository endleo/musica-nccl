#!/bin/bash -l
# LL thread experiment: is Broadcast Ring/LL slower for 32 KiB than for 64 KiB (8 nodes; 16 nodes: 32 and 64 KiB
# slower than 128 KiB) because of the thread count? (tuning runs 1911532, 1912206, 1914035)
#
# Theory (NCCL 2.27.7): Ring/LL starts at 512 threads and halves them while nBytes < nChannels x nThreads x
# threshold, with threshold = 8 B x nRanks (src/enqueue.cc topoGetAlgoInfo, src/graph/tuning.cc), minimum 3 warps.
# One LL chunk (= one step) holds 32 KiB of payload in 4096 lines of 16 B; within a chunk, each thread polls and
# forwards every nThreads-th line, one after the other (src/device/prims_ll.h), and each network hop waits for the
# whole chunk (src/transport/net.cc). So time ~ network hops x lines per thread per chunk: 32 KiB with 128 threads
# (8 nodes) = 32 lines per thread, 64 KiB with 256 threads = 2 chunks of 16.
#
# Variants (all Ring/LL on one channel; the threshold also sets the channel count, so the channels are pinned):
#   default       NCCL's choice (Ring/LL, 1 channel up to 128 KiB on 8 / 256 KiB on 16 nodes), the reference
#   ll_1ch        pinned to 1 channel, NCCL's thread count: should match default where default has 1 channel
#                 (checks that the channel pin changes nothing else)
#   ll_1ch_512t   always 512 threads (threshold 0): the dip should disappear, 32 KiB = 8 lines per thread
#   ll_1ch_128t   always 128 threads (NCCL_NTHREADS=128, threshold 0): 32 KiB as in default (8 nodes), but 64 KiB
#                 slower than 32 KiB (2 chunks of 32 lines per thread instead of 2 of 16)
# ncclsee records the warps and channels per call, to check that the pinning worked.
# Only Broadcast, 1 KiB - 1 MiB (powers of two and 1.5x in between).
# Results in results/ll_threads/<job>/<round>/<variant>/ (as experiments/tuning.sbatch; analysis/analyze_tuning.py
# expects the tuning variants and does not read this experiment).
# Submit from the repository root: sbatch experiments/ll_threads.sbatch (the effect grows with the number of nodes;
# 8 nodes for the numbers above: sbatch --nodes=8 ...)
#
# Slurm options as in full.sbatch (MUSICA's comfort mode, see there)
#SBATCH --job-name=ll_threads
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # without Slurm: run-local.sh
source ./experiment.sh

EXPERIMENT=ll_threads
PROFILERS=(ncclsee)
NCCL_TESTS=()
declare -A NCCLBENCH_TESTS=([broadcast_bench]=ncclBroadcast)
sizes=""
for ((size = 1024; size <= 2**20; size *= 2)); do
    sizes+="$size "
    if ((size < 2**20)); then sizes+="$((size * 3 / 2)) "; fi
done
NCCLBENCH_ARGS="--sizes $sizes--data-type float --blocking --csv --warmups 5 --iterations 50"
ROUNDS=3

# pinning syntax: see experiments/tuning.sbatch. NCCL_THREAD_THRESHOLDS has spaces in its value, so it is set in an
# NCCL config file (NCCL_CONF_FILE; NCCL then does not read ~/.nccl.conf, but still /etc/nccl.conf). NCCL_NTHREADS
# caps the LL (and Simple) threads, must be a multiple of 32 in [64, 512]
pin="NCCL_ALGO=broadcast:ring NCCL_PROTO=broadcast:LL NCCL_MAX_NCHANNELS=1"
max_threads="NCCL_CONF_FILE=$REPO_DIR/experiments/nccl-conf/ll_threads_max.conf"
VARIANTS=(default ll_1ch ll_1ch_512t ll_1ch_128t)
declare -A VARIANT_ENV=(
    [ll_1ch]="$pin"
    [ll_1ch_512t]="$pin $max_threads"
    [ll_1ch_128t]="$pin $max_threads NCCL_NTHREADS=128"
)

run_experiment
