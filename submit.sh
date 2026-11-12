#!/bin/bash
# Submits an experiment with the Slurm options of this machine (MACHINE, config.sh):
#   ./submit.sh [--<sbatch option>=<value> ...] experiments/<experiment>.sbatch [<arguments of the batch file>]
# The batch files only set the options of the experiment (job name, array, nodes, time, output); the options of the
# machine (partition, QoS, GPUs per node, ...) are here. An option given here replaces the machine's, e.g.
# --qos=dev_zen4_0768_h100x4 for a short test run on MUSICA; any other option goes to sbatch as it is and overrides
# the batch file, e.g. --nodes=8 or --array=1-5%1. Options only as --<name>=<value>.
# Submits from the repository root, which the batch files expect (SLURM_SUBMIT_DIR, logs/).
# Machines without Slurm (DGX Sparks): ./run-local.sh experiments/<experiment>.sbatch instead.
set -eo pipefail
source "$(dirname "$0")/config.sh"

usage() {
    echo "usage: $0 [--<sbatch option>=<value> ...] experiments/<experiment>.sbatch [<arguments>]" >&2
    exit 1
}

declare -A machine_options
case $MACHINE in
    musica)
        # GPU nodes, see https://docs.asc.ac.at/musica/musica_queues.html (short test runs: --qos=dev_zen4_0768_h100x4).
        # MUSICA's comfort mode (the default without the ASC vanilla directive in the batch file, see README): only
        # nodes and GPUs are requested, MUSICA sets the rest. For full GPU nodes (-N X --gres=gpu:4) that is 4 tasks
        # per node, 22 CPUs per task (more than one core per rank matters: bound to a single core, CoMMA's polling
        # thread competes with NCCL's proxy thread), exclusive nodes, all memory; the same options as the working
        # quick run (job 1863239). Every rank sees all GPUs of its node via --gpu-bind=none in LAUNCHER (config.sh).
        # The applied settings are in the job output and in job-info.txt.
        # TODO (TODO.md): switch to the explicit options and vanilla mode. MUSICA finds the vanilla directive anywhere
        # in a batch file, even in a comment, so the batch files must not mention it literally while in comfort mode
        machine_options=([partition]=zen4_0768_h100x4 [qos]=zen4_0768_h100x4 [gres]=gpu:4)
        ;;
    spark) echo "no Slurm on $MACHINE: ./run-local.sh experiments/<experiment>.sbatch" >&2; exit 1 ;;
    *) echo "unknown MACHINE $MACHINE" >&2; exit 1 ;;
esac

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
