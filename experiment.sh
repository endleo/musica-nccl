# Shared part of the experiments in experiments/*.sbatch (sourced, not executed). An experiment sets
#   EXPERIMENT        name, results go to results/<EXPERIMENT>/<run id>/<iteration>/; run id and iteration:
#                     <machine>-<array job id> and the task id (Slurm; the machine, since job ids of different
#                     clusters can be equal), or RUN_ID and ITERATION (set by run-local.sh, without Slurm)
#   PROFILERS         e.g. (comma ncclsee)
#   NCCL_TESTS        nccl-tests binaries <test>_perf, e.g. (all_reduce broadcast), or () for none; NCCL_TESTS_ARGS
#   NCCLBENCH_TESTS   associative array <test name> -> ncclbench operation; NCCLBENCH_ARGS (without --operation)
# optionally
#   LOG_NCCL          true: NCCL debug log and topology/graph dumps per rank (nccl_logging_env in config.sh)
#   VARIANTS          names of variants, e.g. (default ring); every variant runs all tests, in ROUNDS rounds inside
#                     the job (the order rotated by one per round, so no variant always runs first). Results then go
#                     to results/<EXPERIMENT>/<run id>/<round>/<variant>/, the job info to
#                     results/<EXPERIMENT>/<run id>/
#   VARIANT_ENV       associative array <variant> -> environment of its runs, e.g. ([ring]="NCCL_ALGO=allreduce:ring");
#                     a variant without an entry runs with the job's environment
#   ROUNDS            number of rounds (default 1)
#   STEP_TIMEOUT      time limit of every benchmark run (timeout(1) syntax, e.g. 10m), so a hanging run fails on its
#                     own instead of using up the job; unset: no limit
# Variants that pin a protocol of the machine's UNSUPPORTED_PROTOCOLS (machines/*.sh) are skipped (and listed in
# job-info.txt); an experiment that needs such a protocol checks protocol_supported itself.
# A failed benchmark run (non-zero exit, or STEP_TIMEOUT) is listed in failed.txt of its output folder.
# and then calls run_experiment, which runs every test once per profiler (and variant).

source ./config.sh || exit 1

# powers of two from $1 to $2, e.g. `powers_of_two 8 32` -> 8 16 32
powers_of_two() {
    local size
    for ((size = $1; size <= $2; size *= 2)); do
        printf '%s ' "$size"
    done
}

# protocols an NCCL_PROTO value enables explicitly, lower case, one per line: "allreduce:LL;broadcast:LL128" -> ll,
# ll128 (syntax: experiments/tuning.sbatch). Exclusions ("^LL128") enable nothing explicitly and are left out
pinned_protocols() {
    local entry list entries
    IFS=';' read -ra entries <<< "$1"
    for entry in "${entries[@]}"; do
        list=${entry##*:}
        [[ $list == ^* ]] && continue
        tr ',' '\n' <<< "${list,,}"
    done
}

# protocol_supported <protocol>: true unless the protocol is in the machine's UNSUPPORTED_PROTOCOLS (any case)
protocol_supported() {
    local p
    for p in "${UNSUPPORTED_PROTOCOLS[@]}"; do
        [ "${p,,}" = "${1,,}" ] && return 1
    done
    return 0
}

# the unsupported protocols (UNSUPPORTED_PROTOCOLS) an environment pins, space-separated, empty if none; the
# environment as in VARIANT_ENV: "NCCL_ALGO=... NCCL_PROTO=..."
unsupported_pins() {
    local assignment p out=()
    for assignment in $1; do
        [[ $assignment == NCCL_PROTO=* ]] || continue
        for p in $(pinned_protocols "${assignment#NCCL_PROTO=}"); do
            protocol_supported "$p" || out+=("$p")
        done
    done
    [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]}" | sort -u | paste -sd' '
}

# run <profiler> <test> <benchmark command...>; output in $OUT/<test>_<profiler>.out, with the environment of the
# current variant ($variant_env, word-split into assignments for env). Set per rank, behind the launcher: mpirun
# (Sparks) does not pass the environment on to the other nodes. With STEP_TIMEOUT, timeout(1) stops the launcher
# (srun and mpirun pass SIGTERM on to the ranks; SIGKILL a minute later if that does not end them). A failure is
# appended to $OUT/failed.txt (the analysis leaves such variants out)
run() {
    local profiler=$1 test=$2 status=0 guard=() reason
    shift 2
    [ -n "$STEP_TIMEOUT" ] && guard=(timeout --kill-after=60 "$STEP_TIMEOUT")
    echo "[$(date +%T)] $test ($profiler${variant:+, $variant})"
    "${guard[@]}" $LAUNCHER env LOG_NCCL="$LOG_NCCL" $variant_env \
        ./profiled.sh "$profiler" "$OUT/profiler-out/$profiler/$test" "$@" > "$OUT/${test}_$profiler.out" \
        || status=$?
    if [ "$status" -ne 0 ]; then
        reason="exit status $status"
        [ "$status" -eq 124 ] && reason="timed out after $STEP_TIMEOUT (STEP_TIMEOUT)"
        echo "FAILED: $test ($profiler${variant:+, $variant}): $reason"
        echo "$test $profiler: $reason" >> "$OUT/failed.txt"
        failed=1
    fi
}

write_job_info() {
    echo "experiment: $EXPERIMENT"
    echo "machine: $MACHINE"
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "job: $SLURM_JOB_ID (array job ${SLURM_ARRAY_JOB_ID:--}, task ${SLURM_ARRAY_TASK_ID:--})"
        echo "date: $(date -Is)"
        echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
        echo "tasks: $SLURM_NTASKS (per node: $SLURM_TASKS_PER_NODE), CPUs per task: ${SLURM_CPUS_PER_TASK:--}"
    else
        echo "run: $RUN_ID (iteration ${ITERATION:--}), no Slurm"
        echo "date: $(date -Is)"
        echo "launcher: $(echo $LAUNCHER)"
    fi
    echo "profilers: ${PROFILERS[*]}"
    echo "nccl-tests: ${NCCL_TESTS[*]}, args: $NCCL_TESTS_ARGS"
    echo "ncclbench: ${!NCCLBENCH_TESTS[*]}, args: $NCCLBENCH_ARGS"
    echo "NCCL logging: $LOG_NCCL"
    echo "unsupported protocols: ${UNSUPPORTED_PROTOCOLS[*]:--}"
    echo "step timeout: ${STEP_TIMEOUT:--}"
    if [ ${#SKIPPED_VARIANTS[@]} -gt 0 ]; then
        echo "skipped variants: ${SKIPPED_VARIANTS[*]}"
    fi
    if [ ${#VARIANTS[@]} -gt 0 ]; then
        echo "rounds: ${ROUNDS:-1}"
        echo "variants:"
        for v in "${VARIANTS[@]}"; do
            echo "  $v: ${VARIANT_ENV[$v]:-(job environment)}"
        done
    fi
    echo "NCCL: $(nccl_version) ($NCCL_HOME)"
    for repo in nccl nccl-tests lib-ncclbench CoMMA ncclsee-cpp; do
        echo "$repo: $(git -C "$BUILD_DIR/$repo" describe --tags --always 2>/dev/null || echo -)"
    done
    # local patch of ncclsee-cpp (build.sh): applied in the checkout? (the plugin is built from the checkout)
    git -C "$BUILD_DIR/ncclsee-cpp" apply --reverse --check "$REPO_DIR/files/ncclsee/tree-mutex.patch" 2>/dev/null \
        && echo "ncclsee-cpp patch: tree-mutex.patch" || echo "ncclsee-cpp patch: none"
    # as the benchmarks see them: same launcher and number of tasks (a step with fewer tasks may get fewer GPUs,
    # e.g. one per task in MUSICA's comfort mode)
    echo; echo "GPUs per node:"
    # local rank from Slurm or OpenMPI
    $LAUNCHER bash -c '[ "${SLURM_LOCALID:-$OMPI_COMM_WORLD_LOCAL_RANK}" != 0 ] || echo "$(hostname -s): $(nvidia-smi -L | paste -sd" ")"' 2>/dev/null | sort
    echo; echo "GPUs visible per rank (CUDA_VISIBLE_DEVICES, nvidia-smi):"
    $LAUNCHER bash -c 'echo "$(hostname -s) local rank ${SLURM_LOCALID:-$OMPI_COMM_WORLD_LOCAL_RANK}: ${CUDA_VISIBLE_DEVICES:-unset}, $(nvidia-smi -L | wc -l) GPUs"' 2>/dev/null | sort
    # the CPUs each rank may run on (binding of the launcher; one line per rank, but not "local rank", which
    # profiler_results._job_info_ranks counts)
    echo; echo "CPU affinity per rank (taskset):"
    $LAUNCHER bash -c 'echo "$(hostname -s) rank ${SLURM_PROCID:-$OMPI_COMM_WORLD_RANK}: $(taskset -pc $$ | sed "s/.*: //")"' 2>/dev/null | sort
    echo; echo "modules:"
    list_modules
    # everything Slurm applied (incl. what MUSICA's comfort mode set)
    if [ -n "$SLURM_JOB_ID" ]; then
        echo; echo "slurm job:"
        scontrol show job "$SLURM_JOB_ID"
    fi
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

    # one archive per iteration, as on Chameleon (the analysis extracts it).
    # DGX Sparks: dgx1 is the NFS server of /home, and dgx2/3 can still hold NFSv4 delegations on the files their
    # ranks wrote. tar opens with O_NONBLOCK, so on the server that fails with EAGAIN ("Resource temporarily
    # unavailable") instead of waiting for the delegation to be returned. Reading every file first with a normal
    # (blocking) open recalls the delegations and waits for them.
    find "$OUT/profiler-out" -type f -exec cat {} + >/dev/null
    tar czf "$OUT/profiler-out.tar.gz" -C "$OUT" profiler-out && rm -rf "$OUT/profiler-out"
}

run_experiment() {
    load_modules
    check_nccl_version
    # read by profiled.sh on every rank
    export LOG_NCCL="${LOG_NCCL:-false}"
    # profilers whose plugin is not built (CoMMA on the Sparks) are left out
    local p available=()
    for p in "${PROFILERS[@]}"; do
        if [ -f "$(profiler_plugin "$p")" ]; then
            available+=("$p")
        else
            echo "WARNING: no plugin for $p ($(profiler_plugin "$p")), skipped" >&2
        fi
    done
    PROFILERS=("${available[@]}")
    # results/<EXPERIMENT>/<run id>: <machine>-<Slurm job id> (job ids of different clusters can clash), or without
    # Slurm RUN_ID from run-local.sh (<machine>-<date>). Older Slurm runs have the bare job id
    local job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    local run_id="${RUN_ID:-${job_id:+$MACHINE-$job_id}}"
    [ -n "$run_id" ] || { echo "no run id: neither a Slurm job nor RUN_ID (run-local.sh)" >&2; return 1; }

    # pinning a protocol NCCL disables on this machine: refused in the job environment, variants that do it skipped
    local v bad kept=()
    if bad=$(unsupported_pins "NCCL_PROTO=${NCCL_PROTO:-}") && [ -n "$bad" ]; then
        echo "NCCL_PROTO=$NCCL_PROTO in the job environment pins $bad, unsupported on $MACHINE" \
             "(UNSUPPORTED_PROTOCOLS, machines/$MACHINE.sh)" >&2
        return 1
    fi
    SKIPPED_VARIANTS=()
    for v in "${VARIANTS[@]}"; do
        bad=$(unsupported_pins "${VARIANT_ENV[$v]:-}")
        if [ -n "$bad" ]; then
            echo "WARNING: variant $v skipped, it pins $bad, unsupported on $MACHINE (UNSUPPORTED_PROTOCOLS," \
                 "machines/$MACHINE.sh)" >&2
            SKIPPED_VARIANTS+=("$v")
        else
            kept+=("$v")
        fi
    done
    if [ ${#VARIANTS[@]} -gt 0 ] && [ ${#kept[@]} -eq 0 ]; then
        echo "every variant pins a protocol unsupported on $MACHINE, nothing to run" >&2
        return 1
    fi
    VARIANTS=("${kept[@]}")

    failed=0
    variant="" variant_env=""
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        # one iteration per array task
        OUT="$RESULTS_DIR/$EXPERIMENT/$run_id/$(printf %02d "${ITERATION:-${SLURM_ARRAY_TASK_ID:-1}}")"
        mkdir -p "$OUT"
        write_job_info > "$OUT/job-info.txt"
        run_tests
    else
        # all rounds and variants in this job, on the same nodes
        local base="$RESULTS_DIR/$EXPERIMENT/$run_id" round i n=${#VARIANTS[@]}
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
