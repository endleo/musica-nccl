# TODO

Open items first, roughly by priority; done items at the bottom with the date.

## Open

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

### Experiment
- [ ] Baseline run without any profiler, to quantify the profiler overhead: the benchmark times already differ
      between the profilers' runs (8 B Broadcast, ncclbench: about 21 µs in the CoMMA runs, 23-29 µs in the ncclsee
      runs of 1863257)
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
- [ ] Check the rings/trees NCCL builds (which GPU crosses the nodes in which channel) with
      `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH`, e.g. in one quick run; this would confirm the per-rank
      explanations in the summary appendix (e.g. rank 2 as the network sender of the 8 B Broadcast), which are
      inferred from the data
- [ ] All three iterations of 1863257 ran on the same two nodes (n3010-021, n3015-015); for node-to-node variation
      (as on Chameleon, fresh servers per run) exclude previously used nodes or submit iterations separately
- [ ] CoMMA: raise `NCCL_PROFILER_NCCLOP_TIMEOUT` (default 10s, e.g. `60s`); on Chameleon nccl-tests lost large
      ops that were reclaimed without `child_dur` (see Chameleon TODO). Not needed on MUSICA so far (1863257: CoMMA
      has times for 100 % of the calls from 4 MiB on, nothing lost to the timeout), but could come up again with
      larger messages or a busier network
- [ ] nccl-tests: more warmups/iterations (`-w 5 -n 50`) for comparability with ncclbench
- [ ] Switch `experiments/quick.sbatch` and `experiments/full.sbatch` from MUSICA's comfort mode to explicit
      options with `#ASC --vanilla`, so the batch files state their options without MUSICA-specific defaults;
      take the values from what comfort mode applied (`scontrol show job` in `job-info.txt`). Known from jobs
      1863221/1863239: 4 tasks per node, 22 CPUs per task, exclusive, one GPU per task (-> `--gpu-bind=none` in
      `LAUNCHER`). The explicit options full.sbatch had before (untested): `--ntasks-per-node=4 --exclusive
      --cpus-per-task=8 --gres=gpu:4`, no per-task GPU binding

### Analysis
- [ ] Run 1863257, `iteration_consistency.pdf`: ncclsee run differs between iterations by up to 28 % (benchmark) /
      39 % (profiler) below 4 MiB (it 01 Broadcast 4-16 KiB higher, it 02 Broadcast up to 4 KiB lower, alternating
      sizes, it 03 AllReduce up to 2 MiB higher); CoMMA run within 5 % (benchmark) / 12.5 % (CoMMA, it 03 AllReduce
      small sizes). Same nodes in all iterations -> look at the per-iteration distributions
- [ ] ncclsee, AllReduce 2 MiB: profiler and benchmark time about equal (ratio 1.01; 0.80 at 1 MiB, 0.89 at 4 MiB);
      NCCL uses ring instead of tree at that size and *only* that size (both profilers record it). CoMMA shows no
      jump there (0.81, 0.85, 0.89), so it is specific to how ncclsee measures the ring case
- [ ] Tests for the analysis (the Chameleon project has a pytest suite for `profiler_results.py`; this copy
      only differs in `_profiler_dirs`)
- [ ] CoMMA's coverage differs between nccl-tests and ncclbench (AllReduce 8-128 KiB, Broadcast 64 KiB-1 MiB,
      `recorded_by_size.pdf`), although NCCL chose the same algorithm, protocol and channels in both; unexplained
      (call pattern? nccl-tests enqueues 20 calls back to back)
- [ ] ncclsee: GROUP and KERNEL_CH events carry a wrong `rank` (0 in the file of rank 1; only COLL events have the
      right one, which is the one the analysis uses) -> possibly report upstream

## Done

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
