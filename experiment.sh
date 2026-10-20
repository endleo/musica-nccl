# Shared part of the experiments in experiments/*.sbatch (sourced, not executed). An experiment sets
#   EXPERIMENT        name, results go to results/<EXPERIMENT>/<array job id>/<array task id>/
#   PROFILERS         e.g. (comma ncclsee)
#   NCCL_TESTS        nccl-tests binaries <test>_perf, e.g. (all_reduce broadcast); NCCL_TESTS_ARGS
#   NCCLBENCH_TESTS   associative array <test name> -> ncclbench operation; NCCLBENCH_ARGS (without --operation)
# and then calls run_experiment, which runs every test once per profiler.

source ./config.sh

# powers of two from $1 to $2, e.g. `powers_of_two 8 32` -> 8 16 32
powers_of_two() {
    local size
    for ((size = $1; size <= $2; size *= 2)); do
        printf '%s ' "$size"
    done
}

# run <profiler> <test> <benchmark command...>; output in $OUT/<test>_<profiler>.out
run() {
    local profiler=$1 test=$2
    shift 2
    echo "[$(date +%T)] $test ($profiler)"
    $LAUNCHER ./profiled.sh "$profiler" "$OUT/profiler-out/$profiler/$test" "$@" > "$OUT/${test}_$profiler.out" \
        || { echo "FAILED: $test ($profiler)"; failed=1; }
}

write_job_info() {
    echo "experiment: $EXPERIMENT"
    echo "job: $SLURM_JOB_ID (array job ${SLURM_ARRAY_JOB_ID:--}, task ${SLURM_ARRAY_TASK_ID:--})"
    echo "date: $(date -Is)"
    echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
    echo "tasks: $SLURM_NTASKS ($SLURM_NTASKS_PER_NODE per node)"
    echo "profilers: ${PROFILERS[*]}"
    echo "nccl-tests: ${NCCL_TESTS[*]}, args: $NCCL_TESTS_ARGS"
    echo "ncclbench: ${!NCCLBENCH_TESTS[*]}, args: $NCCLBENCH_ARGS"
    echo "NCCL: $(nccl_version) ($NCCL_HOME)"
    for repo in nccl nccl-tests lib-ncclbench CoMMA ncclsee-cpp; do
        echo "$repo: $(git -C "$BUILD_DIR/$repo" describe --tags --always)"
    done
    echo; echo "GPUs:"
    srun --ntasks-per-node=1 bash -c 'echo "$(hostname -s): $(nvidia-smi -L | paste -sd" ")"'
    echo; echo "modules:"
    module list 2>&1
}

run_experiment() {
    load_modules
    check_nccl_version

    OUT="$RESULTS_DIR/$EXPERIMENT/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/$(printf %02d "${SLURM_ARRAY_TASK_ID:-1}")"
    mkdir -p "$OUT"
    echo "Results in $OUT"
    write_job_info > "$OUT/job-info.txt"

    failed=0
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
    return $failed
}
