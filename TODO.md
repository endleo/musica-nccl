# TODO

Open items first, roughly by priority; done items at the bottom with the date.

## Open

### Before the first run
- [ ] CoMMA build: bindgen needs libclang; no Clang/LLVM module is loaded in `config.sh` yet (system libclang or
      e.g. a `Clang/...-GCCcore-14.3.0` module); check where the CMake 4 comes from (system or a module)
- [ ] Check that the module sets `NCCL_HOME` (or `NCCL_ROOT`) and whether its include directory has profiler
      headers (`module show <nccl>`, then `ls <prefix>/include`; NCCL's own install only ships `nccl.h`)
- [ ] First build: does `build.sh fetch` work on the login node, does CoMMA build against the NCCL source checkout
      (on Chameleon it was a built NCCL tree), are clang/libclang and protoc available as modules?
- [ ] After the first quick run: switch `experiments/quick.sbatch` from MUSICA's comfort mode to explicit options
      with `#ASC --vanilla` (as `full.sbatch`), so the batch files state their options without MUSICA-specific
      defaults; take the values from what comfort mode applied (`scontrol show job` in `job-info.txt`: tasks per
      node, CPUs per task, exclusive, memory, GPU binding)
- [ ] Check that `#ASC --vanilla` with the explicit options of `full.sbatch` and `build.sh` is accepted (MUSICA
      checks for required flags)
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
- [ ] CoMMA: raise `NCCL_PROFILER_NCCLOP_TIMEOUT` (default 10s, e.g. `60s`); on Chameleon nccl-tests lost large
      ops that were reclaimed without `child_dur` (see Chameleon TODO)
- [ ] Optional: baseline run without any profiler, to quantify profiler overhead
- [ ] nccl-tests: more warmups/iterations (`-w 5 -n 50`) for comparability with ncclbench
- [ ] Check the CPU binding in the job output (`--cpu-bind=verbose`) and whether `--cpus-per-task=8` fits the nodes

### Analysis
- [ ] Tests for the analysis (the Chameleon project has a pytest suite for `profiler_results.py`; this copy
      only differs in `_profiler_dirs`)

## Done

### 2026-10-21
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
