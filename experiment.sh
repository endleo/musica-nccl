# Shared part of the experiments in experiments/*.sbatch (sourced, not executed). An experiment sets
#   EXPERIMENT        name, results go to results/<EXPERIMENT>/<array job id>/<array task id>/
#   PROFILERS         e.g. (comma ncclsee)
#   NCCL_TESTS        nccl-tests binaries <test>_perf, e.g. (all_reduce broadcast), or () for none; NCCL_TESTS_ARGS
#   NCCLBENCH_TESTS   associative array <test name> -> ncclbench operation; NCCLBENCH_ARGS (without --operation)
# optionally
#   LOG_NCCL          true: NCCL debug log and topology/graph dumps per rank (nccl_logging_env in config.sh)
#   VARIANTS          names of variants, e.g. (default ring); every variant runs all tests, in ROUNDS rounds inside
#                     the job (the order rotated by one per round, so no variant always runs first). Results then go
#                     to results/<EXPERIMENT>/<job id>/<round>/<variant>/, the job info to results/<EXPERIMENT>/<job id>/
#   VARIANT_ENV       associative array <variant> -> environment of its runs, e.g. ([ring]="NCCL_ALGO=allreduce:ring");
#                     a variant without an entry runs with the job's environment
#   ROUNDS            number of rounds (default 1)
# and then calls run_experiment, which runs every test once per profiler (and variant).

source ./config.sh

# powers of two from $1 to $2, e.g. `powers_of_two 8 32` -> 8 16 32
powers_of_two() {
    local size
    for ((size = $1; size <= $2; size *= 2)); do
        printf '%s ' "$size"
    done
}

# run <profiler> <test> <benchmark command...>; output in $OUT/<test>_<profiler>.out, with the environment of the
# current variant ($variant_env, word-split into assignments for env)
run() {
    local profiler=$1 test=$2
    shift 2
    echo "[$(date +%T)] $test ($profiler${variant:+, $variant})"
    env $variant_env $LAUNCHER ./profiled.sh "$profiler" "$OUT/profiler-out/$profiler/$test" "$@" \
        > "$OUT/${test}_$profiler.out" || { echo "FAILED: $test ($profiler${variant:+, $variant})"; failed=1; }
}

write_job_info() {
    echo "experiment: $EXPERIMENT"
    echo "job: $SLURM_JOB_ID (array job ${SLURM_ARRAY_JOB_ID:--}, task ${SLURM_ARRAY_TASK_ID:--})"
    echo "date: $(date -Is)"
    echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
    echo "tasks: $SLURM_NTASKS (per node: $SLURM_TASKS_PER_NODE), CPUs per task: ${SLURM_CPUS_PER_TASK:--}"
    echo "profilers: ${PROFILERS[*]}"
    echo "nccl-tests: ${NCCL_TESTS[*]}, args: $NCCL_TESTS_ARGS"
    echo "ncclbench: ${!NCCLBENCH_TESTS[*]}, args: $NCCLBENCH_ARGS"
    echo "NCCL logging: $LOG_NCCL"
    if [ ${#VARIANTS[@]} -gt 0 ]; then
        echo "rounds: ${ROUNDS:-1}"
        echo "variants:"
        for v in "${VARIANTS[@]}"; do
            echo "  $v: ${VARIANT_ENV[$v]:-(job environment)}"
        done
    fi
    echo "NCCL: $(nccl_version) ($NCCL_HOME)"
    for repo in nccl nccl-tests lib-ncclbench CoMMA ncclsee-cpp; do
        echo "$repo: $(git -C "$BUILD_DIR/$repo" describe --tags --always)"
    done
    # as the benchmarks see them: same launcher and number of tasks (a step with fewer tasks may get fewer GPUs,
    # e.g. one per task in MUSICA's comfort mode)
    echo; echo "GPUs per node:"
    $LAUNCHER bash -c '[ "$SLURM_LOCALID" != 0 ] || echo "$(hostname -s): $(nvidia-smi -L | paste -sd" ")"' 2>/dev/null | sort
    echo; echo "GPUs visible per rank (CUDA_VISIBLE_DEVICES, nvidia-smi):"
    $LAUNCHER bash -c 'echo "$(hostname -s) local rank $SLURM_LOCALID: ${CUDA_VISIBLE_DEVICES:-unset}, $(nvidia-smi -L | wc -l) GPUs"' 2>/dev/null | sort
    echo; echo "modules:"
    module list 2>&1
    # everything Slurm applied (incl. what MUSICA's comfort mode set)
    echo; echo "slurm job:"
    scontrol show job "$SLURM_JOB_ID"
    # of the job, after loading the modules; the benchmark processes additionally get the profiler settings, the
    # variant's environment and the library path (profiled.sh), see env-rank<N>.txt in the profiler outputs
    echo; echo "environment:"
    print_environment
}

# all tests once per profiler, output in $OUT
run_tests() {
    mkdir -p "$OUT"
    echo "Results in $OUT"
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
}

run_experiment() {
    load_modules
    check_nccl_version
    # read by profiled.sh on every rank
    export LOG_NCCL="${LOG_NCCL:-false}"

    failed=0
    variant="" variant_env=""
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        # one iteration per array task
        OUT="$RESULTS_DIR/$EXPERIMENT/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/$(printf %02d "${SLURM_ARRAY_TASK_ID:-1}")"
        mkdir -p "$OUT"
        write_job_info > "$OUT/job-info.txt"
        run_tests
    else
        # all rounds and variants in this job, on the same nodes
        local base="$RESULTS_DIR/$EXPERIMENT/$SLURM_JOB_ID" round i n=${#VARIANTS[@]}
        mkdir -p "$base"
        write_job_info > "$base/job-info.txt"
        for ((round = 1; round <= ${ROUNDS:-1}; round++)); do
            for ((i = 0; i < n; i++)); do
                variant=${VARIANTS[(i + round - 1) % n]}
                variant_env=${VARIANT_ENV[$variant]:-}
                OUT="$base/$(printf %02d "$round")/$variant"
                run_tests
                echo "$variant_env" > "$OUT/variant-env.txt"
            done
        done
    fi

    echo "[$(date +%T)] done"
    return $failed
}
