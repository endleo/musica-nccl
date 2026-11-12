#!/bin/bash
# Copies results and job logs from the clone on the cluster into this clone (run locally):
#   ./fetch.sh [<ssh host>:<path of the clone on the cluster>]
# The default assumes an ssh config entry "musica" and the clone in the home directory there.
# The results come from the cluster's RESULTS_DIR (config.sh, may be set per machine, e.g. outside the home
# directory), asked for over ssh; the logs from the clone. Locally, both go into results/ and logs/.
set -eo pipefail
REMOTE="${1:-musica:musica-nccl}"
host=${REMOTE%%:*}
clone=${REMOTE#*:}
cd "$(dirname "$0")"

# shellcheck disable=SC2016  # expanded on the cluster
results=$(ssh "$host" "bash -c $(printf %q 'cd "$1" && source ./config.sh >/dev/null && echo "$RESULTS_DIR"') _ $(printf %q "$clone")") \
    || { echo "could not get RESULTS_DIR from $REMOTE (config.sh)" >&2; exit 1; }
[ -n "$results" ] || { echo "RESULTS_DIR is empty on $REMOTE" >&2; exit 1; }
echo "results on $host: $results"

# profiler-out-extracted: only exists if the analysis was run on the cluster
rsync -av --exclude profiler-out-extracted "$host:$results/" results/
rsync -av "$REMOTE/logs/" logs/
