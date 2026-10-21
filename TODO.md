# TODO

Open items first, roughly by priority; done items at the bottom with the date.

## Open

### Before the first run
- [ ] EESSI: the CUDA runtime loads the GPU driver (`libcuda.so`) at runtime; EESSI programs only find the host's
      driver if the admins set up EESSI's `host_injections` for it. If the quick run fails with "CUDA driver
      version is insufficient" / "no CUDA-capable device", that is the place to look (or ask the admins)
- [ ] Check that the module sets `NCCL_HOME` (or `NCCL_ROOT`) and whether its include directory has profiler
      headers (`module show <nccl>`, then `ls <prefix>/include`; NCCL's own install only ships `nccl.h`)
- [ ] First build: does `build.sh fetch` work on the login node, does CoMMA build against the NCCL source checkout
      (on Chameleon it was a built NCCL tree), does it build with the modules in `config.sh`?
- [ ] Switch `experiments/quick.sbatch` and `experiments/full.sbatch` from MUSICA's comfort mode to explicit
      options with `#ASC --vanilla`, so the batch files state their options without MUSICA-specific defaults;
      take the values from what comfort mode applied (`scontrol show job` in `job-info.txt`). Known from jobs
      1863221/1863239: 4 tasks per node, 22 CPUs per task, exclusive, one GPU per task (-> `--gpu-bind=none` in
      `LAUNCHER`). The explicit options full.sbatch had before (untested): `--ntasks-per-node=4 --exclusive
      --cpus-per-task=8 --gres=gpu:4`, no per-task GPU binding
- [ ] Check that `#ASC --vanilla` with the explicit options of `build.sh` is accepted (MUSICA checks for required
      flags)
- [ ] First `experiments/quick.sbatch` run: does `srun` launch the MPI ranks (PMI support of the MPI module;
      otherwise `LAUNCHER="mpirun"` in `config.sh`, which needs `-x` for the environment), do the 4 ranks per
      node get different GPUs (nccl-tests prints the device per rank), does it finish in < 10 min?

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
- [ ] **CoMMA has no times for ranks that only communicate within their node** (full run 1863257: only 60-69 % of
      the calls have a time, ncclsee ~100 %). CoMMA records every call, but its time (`child_dur`) comes from child
      events, by default only proxy ops (network transfers). Ranks without a network transfer in an op never get a
      child and the op is reclaimed after the timeout. Small sizes: AllReduce only ranks 0/4 (tree), Broadcast only
      ranks 2/4 (ring) have times; from 128-256 KiB on all ranks (more channels). Not visible on Chameleon (1 GPU per
      node, every rank on the network). Options:
  - `NCCL_PROFILER_TRACK_KERNEL_CH=true` (CoMMA option, default false): kernel channels as children too, so every
    rank gets a time, but `child_dur` then means something else than on Chameleon (kernel instead of proxy time)
  - 1 rank per node (as on Chameleon; e.g. `--gres=gpu:1` per node), same semantics as on Chameleon
  - keep it, and only compare CoMMA on the ranks it has times for (the max over ranks in the violin plots is then
    over a subset of the ranks)
- [ ] All three iterations of 1863257 ran on the same two nodes (n3010-021, n3015-015); for node-to-node variation
      (as on Chameleon, fresh servers per run) exclude previously used nodes or submit iterations separately
- [ ] CoMMA: raise `NCCL_PROFILER_NCCLOP_TIMEOUT` (default 10s, e.g. `60s`); on Chameleon nccl-tests lost large
      ops that were reclaimed without `child_dur` (see Chameleon TODO)
- [ ] Optional: baseline run without any profiler, to quantify profiler overhead
- [ ] nccl-tests: more warmups/iterations (`-w 5 -n 50`) for comparability with ncclbench

### Analysis
- [ ] Decide between `latency_ratio.pdf` and `latency_difference.pdf` (same figure, profiler vs. benchmark as ratio or
      as difference in µs)
- [ ] Run 1863257, ncclsee run: iteration 02 has 20-35 % lower Broadcast medians at sizes up to 32 KiB (alternating
      sizes), iteration 03 5-25 % higher AllReduce medians up to 16 MiB (`iteration_consistency.pdf`); the CoMMA run
      is within a few %. Same nodes in all iterations -> look at the per-iteration distributions
- [ ] ncclsee, AllReduce 2 MiB: profiler and benchmark time about equal (ratio ~1, elsewhere ~0.8); NCCL uses ring
      instead of tree at that size (CoMMA records)
- [ ] Tests for the analysis (the Chameleon project has a pytest suite for `profiler_results.py`; this copy
      only differs in `_profiler_dirs`)

## Done

### 2026-10-21
- [x] Latency over all sizes (`latency_ratio.pdf` / `latency_difference.pdf`) and consistency of the iterations
      (`iteration_consistency.pdf`)
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
