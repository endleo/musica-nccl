# TODO

Open items per experiment (and general ones first), roughly by priority; done items at the bottom with the date.

## General (all experiments)

### NCCL version / Inspector
- [ ] Decide between the module's NCCL (2.27.7) and building NCCL ourselves:
  - module: no NCCL Inspector (needs profiler API v5, NCCL 2.27 loads up to v4), and a different NCCL version
    than on Chameleon (2.32.3), but possibly optimized for the machine (build options, patches, defaults in the
    module environment)
  - own build (e.g. v2.32.3, as on Chameleon): all three profilers and the same NCCL as on Chameleon, but not
    tuned beyond environment variables
  - possibly both: compare the module's NCCL with an own build of 2.27.7 without profilers first, to see if
    the module is tuned at all; check the module's environment (`module show`, `NCCL_*` variables it sets)
- [ ] With NCCL >= 2.29.2: check that the broadcast workaround (`NCCL_ALLGATHERV_ENABLE=0`) is still needed

### Slurm
- [ ] Switch all batch files in `experiments/` from MUSICA's comfort mode to explicit
      options with `#ASC --vanilla`, so the batch files state their options without MUSICA-specific defaults;
      take the values from what comfort mode applied (`scontrol show job` in `job-info.txt`). Known from jobs
      1863221/1863239: 4 tasks per node, 22 CPUs per task, exclusive, one GPU per task (-> `--gpu-bind=none` in
      `LAUNCHER`). The explicit options full.sbatch had before (untested): `--ntasks-per-node=4 --exclusive
      --cpus-per-task=8 --gres=gpu:4`, no per-task GPU binding

## Profiler experiment (`full.sbatch`, `quick.sbatch`)

### Experiment
- [ ] Baseline run without any profiler, to quantify the profiler overhead: the benchmark times already differ
      between the profilers' runs (8 B Broadcast, ncclbench: about 21 µs in the CoMMA runs, 23-29 µs in the ncclsee
      runs of 1863257). Caution: Broadcast/LL up to 16 KiB is bimodal (clusters at about 20 and 28 µs at 8 B, tuning
      run 1893400), so these medians may differ by the cluster they fall into rather than by profiler overhead
- [ ] **CoMMA has no times for ranks that only communicate within their node** (full run 1863257: only 60-69 % of
      the calls have a time, ncclsee ~100 %). CoMMA records every call, but its time (`child_dur`) comes from child
      events, by default only proxy ops (network transfers). Ranks without a network transfer in an op never get a
      child and the op is reclaimed after the timeout. Small sizes: AllReduce only ranks 0/4 (tree), Broadcast only
      ranks 2/4 (ring) have times; with more channels more ranks get times, 100 % from 256 KiB for AllReduce (both
      benchmarks) and from 128 KiB (ncclbench) or 2 MiB (nccl-tests) for Broadcast. Not visible on Chameleon (1 GPU
      per node, every rank on the network). Options:
  - `NCCL_PROFILER_TRACK_KERNEL_CH=true` (CoMMA option, default false): kernel channels as children too, so every
    rank gets a time, but `child_dur` then means something else than on Chameleon (kernel instead of proxy time)
  - keep it, and only compare CoMMA on the ranks it has times for (the max over ranks in the violin plots is then
    over a subset of the ranks)
- [ ] CoMMA: raise `NCCL_PROFILER_NCCLOP_TIMEOUT` (default 10s, e.g. `60s`); on Chameleon nccl-tests lost large
      ops that were reclaimed without `child_dur` (see Chameleon TODO). Not needed on MUSICA so far (1863257: CoMMA
      has times for 100 % of the calls from 4 MiB on, nothing lost to the timeout), but could come up again with
      larger messages or a busier network
- [ ] nccl-tests: more warmups/iterations (`-w 5 -n 50`) for comparability with ncclbench

### Analysis
- [ ] Run 1863257, `iteration_consistency.pdf`: ncclsee run differs between iterations by up to 28 % (benchmark) /
      39 % (profiler) below 4 MiB (it 01 Broadcast 4-16 KiB higher, it 02 Broadcast up to 4 KiB lower, alternating
      sizes, it 03 AllReduce up to 2 MiB higher); CoMMA run within 5 % (benchmark) / 12.5 % (CoMMA, it 03 AllReduce
      small sizes). Same nodes in all iterations -> look at the per-iteration distributions. The Broadcast part
      matches the bimodal Broadcast/LL times of the tuning run (rounds up to 23 % apart, `report/tuning/summary.typ`)
- [ ] ncclsee, AllReduce 2 MiB: profiler and benchmark time about equal (ratio 1.01; 0.80 at 1 MiB, 0.89 at 4 MiB);
      NCCL uses ring instead of tree at that size and *only* that size (both profilers record it). CoMMA shows no
      jump there (0.81, 0.85, 0.89), so it is specific to how ncclsee measures the ring case
      -> why NCCL picks Ring there: cost model, confirmed in the logging experiment (Tree bandwidth correction
      factor 0.6 up to 2 MiB, 0.8 from 4 MiB); Ring is actually faster there (tuning run 1893400: 15 % at 2 MiB, also
      at 1.5, 3 and 4 MiB)
- [ ] Tests for the analysis (the Chameleon project has a pytest suite for `profiler_results.py`; this copy
      only differs in `_profiler_dirs`)
- [ ] CoMMA's coverage differs between nccl-tests and ncclbench (AllReduce 8-128 KiB, Broadcast 64 KiB-1 MiB,
      `recorded_by_size.pdf`), although NCCL chose the same algorithm, protocol and channels in both; unexplained
      (call pattern? nccl-tests enqueues 20 calls back to back)
      Also: the summary's guess for ncclbench's 25 % at 32-64 KiB (all trees cross the nodes at the same GPUs) is
      ruled out by the logging run: the 4 tree channels cross at 4 different GPU pairs
- [ ] ncclsee: GROUP and KERNEL_CH events carry a wrong `rank` (0 in the file of rank 1; only COLL events have the
      right one, which is the one the analysis uses) -> possibly report upstream
- [ ] Compare runs 1863257 and 1874191 (which ran on different nodes) to see how consistent latencies are (ideally
      also repeat this with more future runs)


## Logging experiment (`logging.sbatch`)

ncclbench + ncclsee, 8 B - 1 GiB, one iteration, `NCCL_DEBUG=INFO` with `INIT,ENV,GRAPH,TUNING,NET` and the
topology/graph dumps (`nccl_logging_env` in `config.sh`); the logs are in the profiler outputs
(`nccl-rank<N>.log`, `topo-rank0.xml`, `graph-rank0.xml` next to `env-rank<N>.txt`). Run 1889653 evaluated in
`report/logging/summary.typ`; done items below (2026-10-29).

- [ ] `module show NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1` in `job-info.txt` (the log shows that no `NCCL_*`
      setting reached NCCL from outside, but not what else the module sets)
- [ ] PXN and CoMMA: rank 2's network send of the 8 B Broadcast is done by the proxy thread of rank 0's process
      (`NET/IB/4(0)`), yet CoMMA records the transfer for rank 2. Check how proxy events of a PXN send are attributed
      (profiler API v4: the proxy op descriptor carries the originating pid/rank?), and whether this is related to
      the longer enqueue on ranks 0 and 2 (profiler summary, appendix)
- [ ] Optional: the hidden `mlx5_1` (probably the inactive second port of the 25 Gb RoCE card `mlx5_0`); check with
      `ibv_devices` / `ibstat` / `rdma link` on a compute node
- [ ] Later: pass-through tuner plugin (tuner API v4 for NCCL 2.27) that logs the cost table NCCL computed
      (`collCostTable` in `getCollInfo`, 7 algorithms x 3 protocols, -1 = not available) once per (operation, size,
      `numPipeOps`) and changes nothing: the estimates of the options NCCL did not choose, without recomputing
      the model. Leave `*nChannels` at 0 (in 2.27 the plugin's value replaces NCCL's; the bundled example sets 1
      for every call without a matching config line); log through the logger NCCL passes to `init`; with dedup per
      key the per-call cost is a lookup. In 2.27 the plugin runs after NCCL disabled what `NCCL_ALGO`/`NCCL_PROTO`
      exclude, so in pinned runs it only sees the allowed options. Less urgent now: the offline calculation from the
      TUNING table already reproduces every choice

## Tuning experiment (`tuning.sbatch`)

ncclbench + ncclsee, 8 B - 1 GiB plus 1.5 and 3 MiB; `default` (NCCL's choice) and every available
algorithm/protocol combination pinned with `NCCL_ALGO` + `NCCL_PROTO` (AllReduce `tree_*`/`ring_*` x LL/LL128/Simple,
Broadcast's protocol pinned in the `ring_*` variants), i.e. a measured version of NCCL's cost table; in 3 rounds in
one job on the same nodes, order rotated per round; results in `results/tuning/<job>/<round>/<variant>/`.
42 runs, estimated 15-20 min (logging run: about 15 s per benchmark; LL variants slower at large sizes).

Run 1893400 evaluated in `report/tuning/summary.typ`; done items below (2026-10-31).

- [ ] Bimodal Broadcast/LL times up to 16 KiB (8 B: clusters at about 20 and 28 µs, the median jumps between them,
      rounds up to 23 % apart; LL128/Simple are unimodal): where the clusters come from (which rank is last,
      ncclsee's per-rank kernel times of the slow vs. fast calls), and whether they also occur without a profiler
- [ ] AllReduce Ring/LL, round 03 10-21 % slower (64 B - 1 MiB) and Tree/LL128, round 01 5-7 % slower
      (128 KiB - 4 MiB) than the other rounds: single slow runs, or a pattern (needs more rounds/runs)
- [ ] Second tuning run on other nodes: are the wins over NCCL's choice (AllReduce 4 MiB Ring/LL128, 32-64 MiB
      Tree/Simple; Broadcast 256-512 KiB LL128, 128 MiB LL128; 4-16 %) a property of the system or of n3010-020/021?
- [ ] The model's estimates are far from the measured times for combinations NCCL does not pick (Tree/LL above
      128 KiB up to 3.5x, Broadcast LL128/Simple latencies 1.7x/2.7x at 8 B, Broadcast 64 KiB - 32 MiB 0.6-0.85x):
      the TUNING table is printed with one decimal and was taken from the logging run on other nodes; check with a
      logging run on the tuning nodes, or the tuner plugin below (exact cost table)
- [ ] Also with nccl-tests (20 calls queued without sync per measurement): does the difference between pinned and
      default change when launch/host overhead overlaps with the GPU work?
- [ ] Also without ncclsee, for the profiler's overhead on the differences (needs a `none` case in `profiler_env`);
      keep ncclsee in the first runs to confirm what NCCL actually did
- [ ] Broadcast runs in all variants, also in the `tree_*` ones where only AllReduce is pinned (extra default
      samples, but run time); per-variant test lists if run time matters

## Done

### 2026-10-31
- [x] Tuning run 1893400 (3 rounds x 7 variants, 42 runs, n3010-020/021) evaluated (`analysis/tuning.py`, called by
      `analyze.py` for runs with variants; `report/tuning/summary.typ`): all runs complete, ncclsee has all
      440 calls per round/variant/operation/size, every pinned call used the pinned algorithm/protocol (channels and
      warps follow the pinned combination's thresholds); NCCL's choice identical to the logging run and to the cost
      model (Ring at 1.5-3 MiB as predicted)
- [x] Measured vs. estimated (cost model with the logging run's TUNING table): estimates are mostly outside the
      measured distribution (179 of 270 below or above every call); NCCL's choice 0.76-0.93 of the measured median
      up to 64 KiB (host overhead not in the model), 0.87-1.06 at 128 KiB - 64 MiB, 1.21-1.30 from 128 MiB (AllReduce)
- [x] Pinned vs. default with a noise floor from two runs of the same configuration: NCCL's choice is the fastest
      (within noise) at 27 of 30 sizes per operation. Ring vs. Tree around 2 MiB: Ring faster at 1.5-4 MiB (NCCL:
      1.5-3 MiB). Close calls of the logging run: NCCL right at AllReduce 1 MiB, 128 MiB, Broadcast 64 MiB; wrong at
      AllReduce 64 MiB (Simple 14 % faster) and Broadcast 512 KiB (LL128 16 % faster)
- [x] Consistency of the rounds (`round_consistency.pdf`): within about 2 % from 1 MiB, except Broadcast/LL at small
      sizes (bimodal) and two single slow rounds (open items above)

### 2026-10-30
- [x] Tuning variants from the logging run: the full grid of available combinations (`NCCL_ALGO` and `NCCL_PROTO`
      together leave exactly one; per-operation prefixes only change that operation, `parseList` in NCCL 2.27's
      `src/graph/tuning.cc`) instead of `allreduce_ring`/`allreduce_tree`; covers Ring vs. Tree around 2 MiB and
      all close calls. Sizes 1.5 and 3 MiB added
- [x] Correctness of pinned combinations: not a concern on MUSICA. LL128 is enabled by default there (TUNING table
      of run 1889653), so pinning it does not override NCCL's hardware check; all pinned combinations are ones NCCL
      may choose itself
- [x] Dry run of `tuning.sbatch` (local, with stubs for `srun`/`module`/`scontrol`/`nvidia-smi` and ncclbench): 42
      runs, rotation and result layout as intended, `NCCL_ALGO`/`NCCL_PROTO` (incl. `;`) reach every rank's
      environment. Not covered: NCCL itself, srun's environment passing (shown by the logging run: `LOG_NCCL`
      reached all ranks)

### 2026-10-29
- [x] Logging run 1889653 evaluated (`report/logging/summary.typ`): all ranks wrote their logs, rank 0 the dumps; no
      `NCCL_*` setting from outside the repo (environment, conf files), internal tuner, IB with GDR on all
      connections (4 x 200 Gb/s, one NIC per GPU; the 25 Gb/s RoCE card is unused); NCCL's per-call log agrees with
      ncclsee on every call and with the tables of the profiler summary; rings/trees per channel confirm the
      per-rank explanations of the profiler summary (correction: PXN, rank 2's send goes through rank 0's proxy);
      NCCL's cost model (TUNING table) reproduces every choice, incl. Ring at 2 MiB
- [x] Report split per experiment: `report/profilers/summary.typ` (with a note on the logging results),
      `report/logging/summary.typ`
- [x] Logging and tuning experiments (`experiments/logging.sbatch`, `experiments/tuning.sbatch`; variants and
      rounds in `experiment.sh`, NCCL logging in `config.sh`/`profiled.sh`), not run yet
- [x] The three iterations of the full experiment tend to get the same nodes assigned every time, but this is not
      guaranteed. Different experiment runs (e.g., 1863257 vs. 1874191) can get different nodes

### 2026-10-23
- [x] NCCL module: found via `EBROOTNCCL`; its include directory only has `nccl.h` (no profiler headers), so the own
      NCCL checkout stays necessary for CoMMA's headers
- [x] Environment logged: of the job in `job-info.txt`, per rank (with the profiler settings) as `env-rank<N>.txt`
      in the profiler outputs; names that look like credentials are left out

### 2026-10-21
- [x] Latency over all sizes (`latency_ratio.pdf`; the difference version was dropped, the ratio shows the convergence
      better) and consistency of the iterations (`iteration_consistency.pdf`)
- [x] `report/summary.typ`: two-page summary of run 1863257 (Typst); figure titles name runs relative to `results/`
- [x] Plot of the recorded calls per message size (`recorded_by_size.pdf`, `profiler_results.plot_recorded_by_size`):
      one panel per operation, color + marker per profiler, line style per benchmark
- [x] Full experiment 1863257 (3 iterations, CoMMA + ncclsee, 8 B - 1 GiB): all steps succeeded, nccl-tests validation
      OK, ncclbench complete; ncclsee recorded ~100 % of the calls, CoMMA see the open item above
- [x] CPU binding (`--cpu-bind=verbose` in the job output): 22 cores per rank in comfort mode (job 1863239)
- [x] `experiments/full.sbatch` uses the same Slurm options as the successful quick run (comfort mode,
      `-N 2 --gres=gpu:4`), regular QoS, so it can be queued as is
- [x] Quick run 1863221 failed: comfort mode binds one GPU per task, but both benchmarks pick their GPU by local rank
      among all visible GPUs ("invalid device ordinal"); `--gpu-bind=none` in `LAUNCHER`. If MUSICA restricts the
      GPUs per task via cgroups, that is not enough -> switch quick to the explicit options of full.sbatch
- [x] EESSI modules set no `LD_LIBRARY_PATH` (`libcudart`/`libmpi` not found, `CXXABI_1.3.15` from the old system
      libstdc++): `runtime_library_path` in `config.sh`, set for the benchmark processes in `profiled.sh` and for the
      library check in `build.sh`
- [x] `libcudart.so.13` not found at runtime: binaries from an earlier build against another CUDA (make/cmake do not
      rebuild after module changes); `./build.sh clean`, and `compile` warns about libraries `ldd` cannot find
- [x] libclang for the CoMMA build: `LLVM/20.1.8-GCCcore-14.3.0` module, `LIBCLANG_PATH` set in `build.sh` (otherwise
      bindgen picks the incomplete system LLVM 21 in `/usr/lib64/llvm21`, whose libLLVM is missing)
- [x] Module loads (NCCL 2.27.7 / CUDA 12.9.1, gompi 2025b, protobuf) and `CUDA_ARCH=90` in `config.sh`
- [x] lib-ncclbench build with CMake 4: `CMAKE_POLICY_VERSION_MINIMUM=3.5` for yaml-cpp 0.8.0, which `build.sh fetch`
      now clones (no download while configuring)
- [x] Slurm partition, account and QoS filled in: GPU nodes `zen4_0768_h100x4` for the experiments (dev QoS for
      the quick experiment), CPU nodes `zen4_0768` for the build job
- [x] `experiments/quick.sbatch` in MUSICA's comfort mode (`-N 2 --gres=gpu:4`) for the first test run;
      `full.sbatch` and `build.sh` explicit with `#ASC --vanilla`; `job-info.txt` records `scontrol show job` and
      `CUDA_VISIBLE_DEVICES` per rank
- [x] NCCL_TAG v2.27.7-1 (version of the module); `build.sh` and the experiments warn if the loaded NCCL differs
- [x] Inspector skipped (not loadable with NCCL 2.27, see above)
- [x] 4 ranks per node (one per GPU), every rank sees all GPUs of its node (both benchmarks pick by local rank)
- [x] Experiment parameters in one batch file per experiment (`experiments/`), shared logic in `experiment.sh`;
      quick experiment (ncclsee only, 8 B - 1 MiB, 1 iteration)
- [x] `analyze.py` takes profilers, tests, sizes and ncclbench settings from the run

### 2026-10-20
- [x] Project scaffolding: builds against the cluster's modules, Slurm array job, `fetch.sh` (rsync), analysis of
      the Chameleon notebook as a script (verified on a converted Chameleon run: identical tables and counts)
