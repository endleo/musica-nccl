# musica-nccl

The NCCL benchmark vs. profiler experiment of the Chameleon project (`chameleon-nccl`), on the MUSICA HPC
cluster: run nccl-tests and lib-ncclbench once per NCCL profiler plugin (CoMMA, ncclsee; no Inspector, see below) and
compare the latencies the profilers report with the times the benchmarks measure.

Open tasks and progress: [TODO.md](TODO.md).

CUDA, NCCL (2.27.7) and MPI come from the cluster's modules; the benchmarks and profiler plugins are built from
source (same versions as on Chameleon). The experiments are Slurm array jobs; the analysis runs locally.

**No NCCL Inspector:** NCCL 2.27 loads profiler plugins up to API v4; the Inspector only implements v5 (and only
exists since NCCL 2.29.7). CoMMA and ncclsee use v4 and work.

## Layout

```
config.sh             shared settings: modules, GPU arch, versions, paths, profiler environment
build.sh              clones and builds everything into build/
experiments/          one Slurm batch file per experiment, with its parameters (profilers, tests, sizes, ...)
  full.sbatch         the Chameleon experiment: CoMMA + ncclsee, 8 B - 1 GiB, 3 iterations
  quick.sbatch        setup check: ncclsee only, 8 B - 1 MiB, 1 iteration (< 10 min)
  logging.sbatch      NCCL's debug log (incl. per-call TUNING) and topology/graph dumps: ncclbench + ncclsee, 1 iteration
  tuning.sbatch       NCCL's choice vs. pinned NCCL_ALGO/NCCL_PROTO: ncclbench + ncclsee, variants interleaved in
                      3 rounds in one job (results/tuning/<job>/<round>/<variant>/)
experiment.sh         shared part of the experiments (run every test once per profiler, job info, packing)
profiled.sh           wrapper started per rank: sets the profiler environment, then runs the benchmark
fetch.sh              copies results/ and logs/ from the cluster into a local clone (rsync over ssh)
TODO.md               open tasks and progress
report/               summaries (Typst), one folder per experiment
  profilers/summary.typ  profiler experiment: summary of a run with the figures
  logging/summary.typ    logging experiment: NCCL's environment, layout, choices and cost model
analysis/
  analyze.py          plots and tables of a run (the evaluation cells of the Chameleon notebook)
  profiler_results.py parsing of benchmark and profiler outputs (copied from the Chameleon project)
files/comma/Cargo.lock  pinned crate versions for the CoMMA build (its repo has none)
results/<experiment>/<job>/<NN>/   output of each array task
logs/                 Slurm job output (not in git)
```

## Before the first run

Check the settings (filled in for MUSICA):

- `config.sh`: `module load` lines in `load_modules` (CUDA, NCCL, MPI, cmake; clang/libclang and protobuf for
  CoMMA), `CUDA_ARCH`; check that `NCCL_HOME`/`CUDA_HOME`/`MPI_HOME` are found (`build.sh compile` prints them)
  and that `NCCL_TAG` matches the module's NCCL version (`build.sh` and the experiments warn if not)
- `experiments/*.sbatch` and `build.sh`: maybe `--time` (partition, account and QoS are set, see
  [MUSICA queues](https://docs.asc.ac.at/musica/musica_queues.html): GPU nodes `zen4_0768_h100x4` for the
  experiments, with QoS `dev_zen4_0768_h100x4` for `quick.sbatch`; CPU nodes `zen4_0768` for `build.sh`)

## Usage

On the cluster (from the repository root, everything relative to it):

```sh
git clone <this repo> musica-nccl && cd musica-nccl
./build.sh                  # or: ./build.sh fetch && sbatch build.sh compile
                            # after changing modules: ./build.sh clean && ./build.sh compile
sbatch experiments/quick.sbatch
sbatch experiments/full.sbatch
```

Locally:

```sh
./fetch.sh                  # default remote musica:musica-nccl (ssh config entry "musica")
uv run analysis/analyze.py  # newest run; or: uv run analysis/analyze.py results/<experiment>/<job>
```

## Experiments

An experiment runs every test once per profiler: nccl-tests `all_reduce_perf` / `broadcast_perf` and
lib-ncclbench `ncclAllReduce` / `ncclBroadcast`, on 2 nodes with 4 ranks each (one per GPU).

Iterations are Slurm array tasks: `#SBATCH --array=1-3%1` in `full.sbatch` submits the experiment as 3
separate tasks (iterations 01, 02, 03), each running the whole experiment (all profilers and tests) on the nodes it
gets; `%1` lets only one of them run at a time. Other counts on the command line, e.g.
`sbatch --array=1-5%1 experiments/full.sbatch`. This is the counterpart of the repeated `run_benchmark` calls with
fresh servers on Chameleon.

A new experiment is a copy of one of the batch files with other parameters and another `EXPERIMENT` name (the
results folder); `analyze.py` takes the profilers, tests, sizes and ncclbench settings from the results.

Differences to Chameleon:

- NCCL 2.27.7 from the module instead of 2.32.3 built from source; the NCCL checkout in `build/nccl` (same version)
  only provides the profiler headers for CoMMA. The module only ships `nccl.h` (NCCL installs no profiler
  headers), so the checkout is needed. No Inspector (see above); `NCCL_ALLGATHERV_ENABLE=0` (broadcast bug
  workaround) is still set, but NCCL 2.27 does not have this bug or variable
- 4 ranks per node instead of 1; both benchmarks pick their GPU by the local rank, so every rank has to see all
  GPUs of its node (no per-task GPU binding)
- launched with `srun` (`LAUNCHER` in `config.sh`) instead of `mpirun`, with several cores per rank (22 in comfort mode): when bound to
  one core, CoMMA's polling thread competes with NCCL's proxy thread (suspected cause of the slow CoMMA runs on
  Chameleon with 2 nodes); `--cpu-bind=verbose` logs the binding in the job output
- profiler outputs per node in `profiler-out/<profiler>/<test>/<host>/` (shared filesystem; CoMMA's file names
  are only unique per node), so `profiler_results._profiler_dirs` differs from the Chameleon version

### MUSICA Slurm modes

MUSICA has two [Slurm modes](https://docs.asc.ac.at/musica/slurm_modes.html); partition, account, QoS and time
limit have to be set in both:

- comfort mode (default): for some request shapes MUSICA sets the other options itself. Full GPU nodes: only
  `-N X --gres=gpu:4`, which gets 4 tasks per node, 22 CPUs per task, exclusive nodes, all memory. Slurm prints the
  applied settings in the job output.
- vanilla mode (`#ASC --vanilla` in the batch file): plain Slurm, the options apply as written (MUSICA only checks
  that the required ones are set). MUSICA finds the directive anywhere in the file, even inside a comment, so
  batch files in comfort mode must not mention it literally.

The experiments (`quick.sbatch`, `full.sbatch`) use comfort mode for now, since that is what worked in the first
runs; they should switch to vanilla mode with explicit options (TODO.md), so that the batch files state what they
use without knowledge of MUSICA's defaults. `build.sh` uses vanilla mode with explicit options.
Either way, `job-info.txt` records what Slurm actually applied (`scontrol show job`, `CUDA_VISIBLE_DEVICES` per
rank).

### Results of an iteration

```
results/<experiment>/<array job id>/<task id>/
  job-info.txt                   experiment and its parameters, job, nodes, GPUs (visible per rank), NCCL version,
                                 commits of the builds, loaded modules, `scontrol show job`, environment of the job
  <test>_<profiler>.out          benchmark output (nccl-tests table / ncclbench csv)
  profiler-out.tar.gz            profiler outputs of all nodes (extracted by profiler_results.load_run), and per
                                 rank the environment the benchmark ran with (profiler settings, library path):
                                 profiler-out/<profiler>/<test>/<host>/env-rank<N>.txt
```

Short summaries to send around, one per experiment, compiled from the repository root:
`typst compile --root . report/profilers/summary.typ` (text written for run 1863257, figures from the analysis of the
run; `--input run=results/<experiment>/<job>` for another run) writes `report/profilers/summary.pdf`;
`report/logging/summary.typ` (run 1889653) has no figures, its numbers are taken from the logs.

`analyze.py` writes `violins.pdf` per iteration and per run:

- `recorded_operations.pdf`, `recorded_by_size.pdf`: share of the NCCL calls with a recorded time (total, per size)
- `latency_ratio.pdf`: ncclbench latencies over all sizes, and the ratio profiler / benchmark time of the same call
- `iteration_consistency.pdf`: per size, how far each iteration's median is from the median over all iterations,
  against the spread within an iteration
- `reported_times.txt`: mean times per size, as the tables in the Chameleon notebook
