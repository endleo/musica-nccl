# musica-nccl

The NCCL benchmark vs. profiler experiment of the Chameleon project (`chameleon-nccl`), on the MUSICA HPC
cluster: run nccl-tests and lib-ncclbench once per NCCL profiler plugin (CoMMA, NCCL Inspector, ncclsee) and
compare the latencies the profilers report with the times the benchmarks measure.

CUDA, NCCL and MPI come from the cluster's modules; the benchmarks and profiler plugins are built from source
(same versions as on Chameleon). The experiment is a Slurm array job; the analysis runs locally.

## Layout

```
config.sh             settings for build and run: modules (TODO), GPU arch (TODO), versions, paths,
                      benchmark parameters, profiler environment
build.sh              clones and builds everything into build/
run.sbatch            the experiment (one iteration per array task)
profiled.sh           wrapper started per rank: sets the profiler environment, then runs the benchmark
fetch.sh              copies results/ and logs/ from the cluster into a local clone (rsync over ssh)
analysis/
  analyze.py          plots and tables of a run (the evaluation cells of the Chameleon notebook)
  profiler_results.py parsing of benchmark and profiler outputs (copied from the Chameleon project)
files/comma/Cargo.lock  pinned crate versions for the CoMMA build (its repo has none)
results/<job>/<NN>/   output of each array task
logs/                 Slurm job output (not in git)
```

## Before the first run

Fill in the TODOs:

- `config.sh`: `module load` lines in `load_modules` (CUDA, NCCL, MPI, cmake; clang/libclang and protobuf for
  CoMMA), `CUDA_ARCH`; check that `NCCL_HOME`/`CUDA_HOME`/`MPI_HOME` are found (`build.sh compile` prints them)
  and that `NCCL_TAG` matches the module's NCCL version
- `run.sbatch` and `build.sh`: `#SBATCH` partition, account, GPU request; maybe `--time`

## Usage

On the cluster (from the repository root, everything relative to it):

```sh
git clone <this repo> musica-nccl && cd musica-nccl
./build.sh                  # or: ./build.sh fetch && sbatch build.sh compile
sbatch run.sbatch           # 3 iterations, one after another; --array=1-5%1 for more
```

Locally:

```sh
./fetch.sh                  # default remote musica:musica-nccl (ssh config entry "musica")
uv run analysis/analyze.py  # newest run; or: uv run analysis/analyze.py results/<job>
```

## Experiment

Per array task (= iteration, on `--nodes=2` nodes with one rank and GPU each, as on Chameleon), for every
profiler: nccl-tests `all_reduce_perf` / `broadcast_perf` (`-b 8 -e 1G -f 2 -g 1`) and lib-ncclbench
`ncclAllReduce` / `ncclBroadcast` (8 B - 1 GiB, 5 warmups, 50 blocking iterations), with
`NCCL_ALLGATHERV_ENABLE=0` (workaround for the NCCL broadcast profiler bug, see the Chameleon project).

Differences to Chameleon:

- NCCL from the module instead of built from source; the NCCL checkout in `build/nccl` is only used to build
  the Inspector and as headers for CoMMA
- launched with `srun` (`LAUNCHER` in `config.sh`) instead of `mpirun`, with `--cpus-per-task=8`: when bound to
  one core, CoMMA's polling thread competes with NCCL's proxy thread (suspected cause of the slow CoMMA runs on
  Chameleon with 2 nodes); `--cpu-bind=verbose` logs the binding in the job output
- profiler outputs per node in `profiler-out/<profiler>/<test>/<host>/` (shared filesystem; CoMMA's file names
  are only unique per node), so `profiler_results._profiler_dirs` differs from the Chameleon version

### Results of an iteration

```
results/<array job id>/<task id>/
  job-info.txt                   job, nodes, GPUs, NCCL version, commits of the builds, loaded modules
  <test>_<profiler>.out          benchmark output (nccl-tests table / ncclbench csv)
  profiler-out.tar.gz            profiler outputs of all nodes (extracted by profiler_results.load_run)
```

`analyze.py` writes `violins.pdf` per iteration and `recorded_operations.pdf` and `reported_times.txt` per run.
