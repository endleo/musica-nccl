#!/bin/bash
# Submits an experiment with the Slurm options of this machine (MACHINE, config.sh):
#   ./submit.sh [--<sbatch option>=<value> ...] experiments/<experiment>.sbatch [<arguments of the batch file>]
# The batch files only set the options of the experiment (job name, array, nodes, time, output); the options of the
# machine (partition, QoS, GPUs per node, ...) are in machines/<machine>.sh (SBATCH_OPTIONS). An option given here
# replaces the machine's, e.g. --qos=dev_zen4_0768_h100x4 for a short test run on MUSICA; any other option goes to
# sbatch as it is and overrides the batch file, e.g. --nodes=8 or --array=1-5%1. Options only as --<name>=<value>.
# Submits from the repository root, which the batch files expect (SLURM_SUBMIT_DIR, logs/).
# Machines without Slurm (DGX Sparks): ./run-local.sh experiments/<experiment>.sbatch instead.
set -eo pipefail
source "$(dirname "$0")/config.sh"

usage() {
    echo "usage: $0 [--<sbatch option>=<value> ...] experiments/<experiment>.sbatch [<arguments>]" >&2
    exit 1
}

# the machine's options (SBATCH_OPTIONS in machines/<machine>.sh), copied: the overrides below change them
[ "${#SBATCH_OPTIONS[@]}" -gt 0 ] \
    || { echo "no Slurm options for $MACHINE (machines/$MACHINE.sh); without Slurm: ./run-local.sh" >&2; exit 1; }
declare -A machine_options
for name in "${!SBATCH_OPTIONS[@]}"; do
    machine_options[$name]=${SBATCH_OPTIONS[$name]}
done

other_options=()
while [[ $1 == -* ]]; do
    [[ $1 == --?*=* ]] || { echo "options only as --<name>=<value>: $1" >&2; usage; }
    name=${1%%=*}
    name=${name#--}
    if [[ -v machine_options[$name] ]]; then
        machine_options[$name]=${1#*=}
    else
        other_options+=("$1")
    fi
    shift
done
[ -f "${1:-}" ] || usage
batch=$(realpath "$1")
shift

options=()
for name in $(printf '%s\n' "${!machine_options[@]}" | sort); do
    options+=("--$name=${machine_options[$name]}")
done

cd "$REPO_DIR"
echo "sbatch ${options[*]} ${other_options[*]} $batch $*"
sbatch "${options[@]}" "${other_options[@]}" "$batch" "$@"
