#!/bin/bash
# Copies results and job logs from the clone on the cluster into this clone (run locally):
#   ./fetch.sh [<ssh host>:<path of the clone on the cluster>]
# The default assumes an ssh config entry "musica" and the clone in the home directory there.
set -eo pipefail
REMOTE="${1:-musica:musica-nccl}"
cd "$(dirname "$0")"

# profiler-out-extracted: only exists if the analysis was run on the cluster
rsync -av --exclude profiler-out-extracted "$REMOTE/results/" results/
rsync -av "$REMOTE/logs/" logs/
