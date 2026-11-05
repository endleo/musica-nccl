#!/bin/bash
# Runs an experiment without Slurm, in place of sbatch (DGX Sparks: on dgx1; the nodes are in LAUNCHER, config.sh):
#   ./run-local.sh experiments/<experiment>.sbatch [<iterations>]
# The iterations (default: as the array of the batch file, e.g. 3 for --array=1-3%1, otherwise 1) run one after the
# other, into results/<experiment>/<run id>/<NN>/. The run id is <machine>-<date>, e.g. spark-20261104-153000: unique
# without a job id and distinct from MUSICA's job ids in a results/ fetched from both. The output goes to
# logs/<job name>_<run id>.out and to the terminal; the full experiment takes about an hour, so start it in tmux.
set -eo pipefail
cd "$(dirname "$0")"
batch=${1:?usage: $0 experiments/<experiment>.sbatch [<iterations>]}
[ -f "$batch" ] || { echo "no batch file $batch" >&2; exit 1; }
source ./config.sh

# --array=1-3%1 -> 3, --array=1 -> 1
iterations=${2:-$(sed -nE 's/^#SBATCH --array=([0-9]+-)?([0-9]+).*/\2/p' "$batch")}
iterations=${iterations:-1}
name=$(sed -nE 's/^#SBATCH --job-name=(.*)/\1/p' "$batch")
export RUN_ID="$MACHINE-$(date +%Y%m%d-%H%M%S)"
log="logs/${name:-$(basename "$batch" .sbatch)}_$RUN_ID.out"

{
    echo "run $RUN_ID: $batch, $iterations iteration(s), on $(hostname -s)"
    status=0
    for ((i = 1; i <= iterations; i++)); do
        echo "[$(date +%T)] iteration $i"
        # like an array task: a failed iteration does not stop the next one
        ITERATION=$i bash "$batch" || { echo "iteration $i FAILED"; status=1; }
    done
    exit $status
} 2>&1 | tee -a "$log"
