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
# Test: halve the LL128 buffer (NCCL_LL128_BUFFSIZE, read by kernel and proxy) -> 288,000 B per chunk, so the kink
# should move from ~4.4 MiB to 8 x 288,000 B = 2,304,000 B (~2.2 MiB). The default variant is NCCL's choice: LL
# up to 1 MiB (8 nodes) / 2 MiB (16 nodes), then LL128 with the default buffer, i.e. the unchanged kink at ~4.4 MiB.
# Only Broadcast. Sizes in 256 KiB steps from 512 KiB to 8 MiB around both kinks, plus a few large sizes to see
# whether the smaller chunks cost bandwidth there.
# Results in results/ll128_chunks/<job>/<round>/<variant>/ (as experiments/tuning.sbatch; analysis/analyze_tuning.py
# expects the tuning variants and does not read this experiment).
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
step=$((256 * 1024))
NCCLBENCH_ARGS="--sizes $(seq -s ' ' $((2 * step)) $step $((32 * step))) $((12 * 2**20)) $((16 * 2**20))"
NCCLBENCH_ARGS+=" $((32 * 2**20)) $((64 * 2**20)) $((256 * 2**20)) $((2**30))"
NCCLBENCH_ARGS+=" --data-type float --blocking --csv --warmups 5 --iterations 50"
ROUNDS=3

# pinning syntax: see experiments/tuning.sbatch. Half of DEFAULT_LL128_BUFFSIZE (4,915,200 B), so the chunk stays a
# multiple of the LL128 grain (1920 B): 288,000 B = 150 grains
VARIANTS=(default ring_ll128_halfbuf)
declare -A VARIANT_ENV=(
    [ring_ll128_halfbuf]="NCCL_ALGO=broadcast:ring NCCL_PROTO=broadcast:LL128 NCCL_LL128_BUFFSIZE=2457600"
)

run_experiment
