#!/bin/bash -l
# The experiment: one iteration per array task, each runs every benchmark once per profiler on the job's nodes.
# Submit from the repository root (after ./build.sh):
#   sbatch run.sbatch                     results in results/<array job id>/<task id>/
#   sbatch --array=1-5%1 run.sbatch       other number of iterations
#SBATCH --job-name=musica-nccl
#SBATCH --array=1-3%1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out
# one rank (and GPU) per node, as on Chameleon. More than one core per rank: bound to a single core, CoMMA's
# polling thread competes with NCCL's proxy thread (suspected cause of the slow CoMMA runs on Chameleon)
#SBATCH --cpus-per-task=8
# TODO: fill in (GPU request: e.g. --gres=gpu:1 or --gpus-per-node=1, whatever the cluster uses)
#SBATCH --partition=
#SBATCH --account=
#SBATCH --gres=

cd "$SLURM_SUBMIT_DIR"
source ./config.sh
load_modules

OUT="$RESULTS_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/$(printf %02d "${SLURM_ARRAY_TASK_ID:-1}")"
mkdir -p "$OUT"
echo "Results in $OUT"

# versions and nodes, for reproducibility
{
    echo "job: $SLURM_JOB_ID (array job ${SLURM_ARRAY_JOB_ID:--}, task ${SLURM_ARRAY_TASK_ID:--})"
    echo "date: $(date -Is)"
    echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
    echo "NCCL: $(nccl_version) ($NCCL_HOME)"
    for repo in nccl nccl-tests lib-ncclbench CoMMA ncclsee-cpp; do
        echo "$repo: $(git -C "$BUILD_DIR/$repo" describe --tags --always)"
    done
    echo; echo "GPUs:"
    srun --ntasks-per-node=1 bash -c 'echo "$(hostname -s): $(nvidia-smi -L | paste -sd" ")"'
    echo; echo "modules:"
    module list 2>&1
} > "$OUT/job-info.txt"

failed=0

run() {  # run <profiler> <test> <benchmark command...>; output in <test>_<profiler>.out
    local profiler=$1 test=$2
    shift 2
    echo "[$(date +%T)] $test ($profiler)"
    $LAUNCHER ./profiled.sh "$profiler" "$OUT/profiler-out/$profiler/$test" "$@" > "$OUT/${test}_$profiler.out" \
        || { echo "FAILED: $test ($profiler)"; failed=1; }
}

for profiler in "${PROFILERS[@]}"; do
    for test in "${NCCL_TESTS[@]}"; do
        run "$profiler" "$test" "$NCCL_TESTS_BIN/${test}_perf" $NCCL_TESTS_ARGS
    done
    for test in $(printf '%s\n' "${!NCCLBENCH_TESTS[@]}" | sort); do
        run "$profiler" "$test" "$NCCLBENCH_BIN" --operation "${NCCLBENCH_TESTS[$test]}" $NCCLBENCH_ARGS
    done
done

# one archive per iteration, as on Chameleon (the analysis extracts it)
tar czf "$OUT/profiler-out.tar.gz" -C "$OUT" profiler-out && rm -rf "$OUT/profiler-out"

echo "[$(date +%T)] done"
exit $failed
