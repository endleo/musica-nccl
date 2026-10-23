#!/bin/bash
# Runs one benchmark process with a profiler plugin; started by the launcher once per rank (see experiment.sh):
#   profiled.sh <profiler> <output directory> <benchmark command...>
# Every node writes into <output directory>/<hostname>, since CoMMA's file names (latency-<pid>.txt) are
# only unique per node. The environment the benchmark runs with is saved there as env-rank<N>.txt.
source "$(dirname "$0")/config.sh"

profiler=$1
host="${SLURMD_NODENAME:-$(hostname -s)}"
[ -n "$host" ] || { echo "profiled.sh: unknown host name" >&2; exit 1; }
out="$2/$host"
shift 2

mkdir -p "$out"
profiler_env "$profiler" "$out" || exit 1
export LD_LIBRARY_PATH="$(runtime_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
print_environment > "$out/env-rank${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-$$}}.txt"
exec "$@"
