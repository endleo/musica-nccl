#!/bin/bash -l
# LL128 chunk experiment: is Broadcast Ring/LL128 slow up to ~4.4 MiB because each channel only has one chunk?
# (tuning runs 1911532, 1912206, 1914035: LL128 slower than LL up to ~8 MiB, flat from 4 to 8 MiB on 16 nodes)
#
# Theory (NCCL 2.27.7): one LL128 step holds 576,000 B of payload per channel (DEFAULT_LL128_BUFFSIZE = 4,915,200 B,
# / NCCL_STEPS, x 15/16), the message is split evenly over the 8 channels, and the net proxy only sends a step once
# the GPU has posted all of it (src/transport/net.cc). Up to 8 x 576,000 B = 4,608,000 B (~4.4 MiB) every channel has
# a single chunk, which crosses the network hops one after the other (no pipelining): time ~ hops x size. Above
# that, the extra data goes into further chunks that pipeline, and the slope drops by about the number of hops.
#
# Test: shrink the LL128 buffer (NCCL_LL128_BUFFSIZE, read by kernel and proxy), so the kink should move with the
# chunk size: half -> 288,000 B per chunk, kink at 8 x 288,000 B (~2.2 MiB); quarter -> 144,000 B, ~1.1 MiB. Ring/LL
# for comparison: what NCCL picks below the LL128 switch, and what LL128 has to beat. Smaller chunks should make
# LL128 faster than LL from smaller sizes on, but may cost bandwidth at large sizes (more steps, more proxy work).
# Variants:
#   default                NCCL's choice: Ring/LL up to 1 MiB (8 nodes) / 2 MiB (16 nodes), then Ring/LL128 with
#                          the default buffer, i.e. the unchanged kink at ~4.4 MiB
#   ring_ll128_halfbuf     Ring/LL128, half the LL128 buffer
#   ring_ll128_quarterbuf  Ring/LL128, a quarter of the LL128 buffer
#   ring_ll                Ring/LL (matches default where NCCL picks Ring/LL itself, the noise floor's reference)
# Only Broadcast, sizes as in experiments/tuning.sbatch (powers of two), from 256 KiB to 512 MiB, plus 1.5x steps
# from 1.5 MiB to 24 MiB around the kinks.
# Results in results/ll128_chunks/<job>/<round>/<variant>/ (as experiments/tuning.sbatch; analysis/analyze_tuning.py
# expects the tuning variants and does not read this experiment, see analysis/analyze_anomaly_checks.py).
# Submit from the repository root: sbatch experiments/ll128_chunks.sbatch (the effect grows with the number of
# nodes, so run with >= 8, e.g. sbatch --nodes=8 ...)
#
# Slurm options as in full.sbatch (MUSICA's comfort mode, see there)
#SBATCH --job-name=ll128_chunks
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # without Slurm: run-local.sh
source ./experiment.sh

EXPERIMENT=ll128_chunks
PROFILERS=(ncclsee)
NCCL_TESTS=()
declare -A NCCLBENCH_TESTS=([broadcast_bench]=ncclBroadcast)
sizes=""
for ((size = 2**18; size <= 2**29; size *= 2)); do
    sizes+="$size "
    if ((size >= 2**20 && size < 2**25)); then sizes+="$((size * 3 / 2)) "; fi  # 1.5 MiB - 24 MiB
done
NCCLBENCH_ARGS="--sizes $sizes--data-type float --blocking --csv --warmups 5 --iterations 50"
ROUNDS=3

# pinning syntax: see experiments/tuning.sbatch. Half and a quarter of DEFAULT_LL128_BUFFSIZE (4,915,200 B), so the
# chunk stays a multiple of the LL128 grain (1920 B): 288,000 B = 150 grains, 144,000 B = 75 grains
ll128="NCCL_ALGO=broadcast:ring NCCL_PROTO=broadcast:LL128"
VARIANTS=(default ring_ll128_halfbuf ring_ll128_quarterbuf ring_ll)
declare -A VARIANT_ENV=(
    [ring_ll128_halfbuf]="$ll128 NCCL_LL128_BUFFSIZE=2457600"
    [ring_ll128_quarterbuf]="$ll128 NCCL_LL128_BUFFSIZE=1228800"
    [ring_ll]="NCCL_ALGO=broadcast:ring NCCL_PROTO=broadcast:LL"
)

run_experiment
