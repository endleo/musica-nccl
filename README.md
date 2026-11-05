# musica-nccl

The NCCL benchmark vs. profiler experiment of the Chameleon project (`chameleon-nccl`), on the MUSICA HPC
cluster: run nccl-tests and lib-ncclbench once per NCCL profiler plugin (CoMMA, ncclsee; no Inspector, see below) and
compare the latencies the profilers report with the times the benchmarks measure.

Open tasks and progress: [TODO.md](TODO.md).

CUDA, NCCL (2.27.7) and MPI come from the cluster's modules; the benchmarks and profiler plugins are built from
source (same versions as on Chameleon). The experiments are Slurm array jobs; the analysis runs locally.

The same experiments also run on three DGX Sparks (dgx1-3, aarch64, one GB10 each, no Slurm), see
[DGX Spark cluster](#dgx-spark-cluster). `MACHINE` in `config.sh` (detected from the CPU architecture) selects the
settings.

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
run-local.sh          runs a batch file without Slurm (DGX Sparks), the iterations one after the other
experiment.sh         shared part of the experiments (run every test once per profiler, job info, packing)
profiled.sh           wrapper started per rank: sets the profiler environment, then runs the benchmark
fetch.sh              copies results/ and logs/ from the cluster into a local clone (rsync over ssh)
TODO.md               open tasks and progress
report/               summaries (Typst), one folder per experiment
  profilers/summary.typ  profiler experiment: summary of a run with the figures
  logging/summary.typ    logging experiment: NCCL's environment, layout, choices and cost model
  tuning/summary.typ     tuning experiment: pinned vs. NCCL's choice, NCCL's estimates vs. measured times
analysis/
  analyze_full.py     profiler experiment (full, also quick): plots and tables of a run (the evaluation cells of
                      the Chameleon notebook)
  analyze_tuning.py   tuning experiment, with a logging run of the same setup (choices per variant, cost model,
                      rounds, pinned vs. default)
  analyze_logging.py  logging experiment: stub, no analysis yet
  common.py           shared by the analyze_*.py scripts (results folder, command line, choice of the run)
  profiler_results.py parsing of benchmark and profiler outputs (copied from the Chameleon project)
  tests/              tests of the scripts (pytest, see "Tests of the analysis"); reference/: the outputs of the
                      regression runs
files/comma/Cargo.lock  pinned crate versions for the CoMMA build (its repo has none)
files/ncclsee/tree-mutex.patch  fix of a race in ncclsee-cpp's event trees, applied by build.sh (see the patch)
results/<experiment>/<job>/<NN>/   output of each array task (<job>: Slurm job id, or <machine>-<date> without Slurm)
logs/                 Slurm job output, run-local.sh output (not in git)
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
uv run analysis/analyze_full.py     # newest run in results/full/, or: ... analyze_full.py results/full/<job>
uv run analysis/analyze_tuning.py   # newest run in results/tuning/, with the newest logging run that fits it
                                    # (or: ... analyze_tuning.py results/tuning/<job> --logging results/logging/<job>)
uv run analysis/analyze_logging.py  # stub, no analysis yet
uv run pytest                       # tests of the analysis scripts (about 1 min; -m "not slow": a few seconds)
typst compile --root . report/<experiment>/summary.typ    # summary with the figures of the run (see below)
```

## DGX Spark cluster

Three DGX Sparks (dgx1, dgx2, dgx3; Ubuntu 24.04, aarch64, one GB10 = sm_121 each), started from dgx1. Differences
to MUSICA, all in `config.sh` (`MACHINE=spark`) unless stated:

- no modules: OpenMPI 5.0.10 from the shared spack (`spack load`, `SPACK_ROOT=/opt/dgx-shared/spack/spack_git`),
  CUDA 13.0 from `/usr/local/cuda`; NCCL is built by `build.sh` from the checkout, pinned to v2.32.3 (as on
  Chameleon; not meant to be comparable with MUSICA's 2.27.7), into `build/nccl/build`. With 2.32 `build.sh` also
  builds the Inspector, the experiments do not use it yet
- no CoMMA for now: `build.sh` skips it (rust, libclang, protobuf), and the experiments leave out every profiler
  whose plugin is missing (warning in the output), so `full.sbatch` runs ncclsee only
- no Slurm: `./run-local.sh experiments/<experiment>.sbatch [<iterations>]` instead of `sbatch`, on dgx1 (in tmux:
  `full` takes about an hour). The `#SBATCH` lines are only comments then; the iterations default to the batch file's
  `--array` (`full`: 3) and run one after the other. Run ids are `spark-<date>-<time>`, e.g.
  `results/full/spark-20261104-153000/01/`, so they cannot clash with MUSICA's job ids in a `results/` fetched from
  both; the `analyze_*.py` scripts pick the newest run by the date in `job-info.txt`
- `LAUNCHER`: `mpirun` with one rank per node (`--prefix` PRRTE's install, so the other nodes find its daemon
  `prted`: spack's OpenMPI uses an external PRRTE). mpirun does not pass the environment on, so `experiment.sh`
  sets the variant's environment and `LOG_NCCL` per rank behind the launcher, and `-x` passes the library paths;
  `profiled.sh` sets the rest as before
- CPUs: each rank runs on the 10 performance cores (`SPARK_CPUS=5-9,15-19`, Cortex-X925; the others are Cortex-A725
  efficiency cores), pinned with `taskset` behind `mpirun --bind-to none`. With mpirun's default binding every rank
  was on CPU 0 alone, where the benchmark's busy-waiting main thread starved NCCL's proxy thread: ncclbench reached
  ~45 % of nccl-tests' bandwidth and took ms per small call (as with CoMMA on Chameleon, see below).
  `job-info.txt` records the affinity of every rank (`CPU affinity per rank`)
- network: the nodes are connected in a ring by direct links (two point-to-point /30 subnets per pair of nodes, no
  switch, 4 RoCE HCAs per node), so a node only reaches its two neighbors' addresses on them. MPI and NCCL's
  bootstrap use the university network (`enP7s7`), which connects all nodes; NCCL's data goes over the HCAs, with
  `NCCL_IB_SUBNET_AWARE_ROUTING=1` (NCCL >= 2.32) so that it picks the HCA on the link to the peer
- the home directory is NFS (exported by dgx1), so `build/` and `results/` are the same on all nodes

Before the first run on dgx1: passwordless ssh from dgx1 to dgx2 and dgx3 must work without a prompt, including
their host keys (`ssh dgx2 true`, `ssh dgx3 true`), since mpirun starts its daemons there over ssh.

```sh
git clone <this repo> musica-nccl && cd musica-nccl   # on dgx1, in the home directory
./build.sh
./run-local.sh experiments/quick.sbatch
```

Locally: `./fetch.sh dgx1:musica-nccl`.

## Experiments

An experiment runs every test once per profiler: nccl-tests `all_reduce_perf` / `broadcast_perf` and
lib-ncclbench `ncclAllReduce` / `ncclBroadcast`, on 2 nodes with 4 ranks each (one per GPU).

Iterations are Slurm array tasks: `#SBATCH --array=1-3%1` in `full.sbatch` submits the experiment as 3
separate tasks (iterations 01, 02, 03), each running the whole experiment (all profilers and tests) on the nodes it
gets; `%1` lets only one of them run at a time. Other counts on the command line, e.g.
`sbatch --array=1-5%1 experiments/full.sbatch`. This is the counterpart of the repeated `run_benchmark` calls with
fresh servers on Chameleon.

A new experiment is a copy of one of the batch files with other parameters and another `EXPERIMENT` name (the
results folder); `analyze_full.py` takes the profilers, tests, sizes and ncclbench settings from the results.

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
`report/logging/summary.typ` (run 1889653) has no figures, its numbers are taken from the logs;
`report/tuning/summary.typ` (run 1893400) reads its figures from `results/tuning/<job>/`, so run
`analyze_tuning.py` on the tuning run first (it also needs a logging run of the same setup in `results/`, for
NCCL's cost table; for 1893400 that is 1889653).

There is one analysis script per experiment, `analysis/analyze_<experiment>.py`; without an argument it analyzes
the newest run of its experiment (by the date in `job-info.txt`), otherwise the run folder given. It refuses runs of
other experiments and names the script for them. Each writes its figures and tables into the run folder.

| experiment | script | writes |
| --- | --- | --- |
| `full` (profilers), also `quick` | `analyze_full.py` | see below |
| `tuning` | `analyze_tuning.py` | see below |
| `logging` | `analyze_logging.py` | nothing yet (stub; `report/logging/summary.typ` is taken from the logs by hand) |

`analyze_full.py` (default: newest run in `results/full/`; quick runs only with their folder, `results/quick/<job>`)
writes `violins.pdf` per iteration and per run:

- `recorded_operations.pdf`, `recorded_by_size.pdf`: share of the NCCL calls with a recorded time (total, per size)
- `latency_ratio.pdf`: ncclbench latencies over all sizes, and the ratio profiler / benchmark time of the same call
- `iteration_consistency.pdf`: per size, how far each iteration's median is from the median over all iterations,
  against the spread within an iteration
- `reported_times.txt`: mean times per size, as the tables in the Chameleon notebook

`analyze_tuning.py` (tuning runs, `results/tuning/<job>/<round>/<variant>/`) needs a logging run as well: NCCL's
TUNING table in its log is the input of the cost model, and its choices are compared with the tuning run's. By
default it takes the newest logging run that fits the tuning run, `--logging results/logging/<job>` names one.
Before anything is written it checks, and aborts with the list of problems:

- that the logging run has the same setup as the tuning run: machine, NCCL version, nodes and GPUs per node, ranks
  per node, the `NCCL_*` variables of the job (not logging and paths); the nodes themselves may differ
- that the NCCL version has a cost model (`COST_MODELS` in `analyze_tuning.py`: so far 2.27.7; the model is NCCL's
  `ncclTopoGetAlgoTime`, so another version needs its source checked and an entry there)
- the tuning run: the 7 variants of `tuning.sbatch`, at least 2 rounds with all variants, ncclbench with
  `--blocking` and the sizes of `model_violins.pdf`
- after loading: the communicator the model uses besides the TUNING table (ranks, nodes, channels from the logging
  run's NCCL log, compute capability from its topology dump) against the ranks and channels `ncclsee` recorded in
  the tuning run, and that the table offers the combinations the variants pin

It extracts the profiler outputs of every variant and writes into the run folder (ncclbench times, all rounds
together unless stated):

- `choices.txt`: algorithm, protocol, channels and warps `ncclsee` recorded per variant and size, checked against
  what was pinned; NCCL's choice compared with the logging run and with the cost model
- `model_vs_measured.txt`, `model_vs_measured.pdf`, `model_violins.pdf`: NCCL's estimate per combination (cost
  model with the TUNING table of the logging run) against the measured distribution
- `round_consistency.pdf`: per variant, how far each round's median is from the median over the rounds
- `pinned_vs_default.pdf`, `best_vs_default.txt`: every pinned combination against NCCL's choice, with the noise
  floor from two runs of the same configuration

### Tests of the analysis

`uv run pytest` (configuration in `pyproject.toml`, tests in `analysis/tests/`, run in parallel with pytest-xdist):

- `test_cli.py`, `test_tuning_checks.py` (a few seconds, no data needed): the scripts refuse runs of other
  experiments, missing folders and folders without `job-info.txt`, and take the newest run of their experiment by
  default; `analyze_tuning.py` accepts other setups (nodes, GPUs, machine, rounds) when the logging run has the
  same, picks the newest logging run that fits, and aborts, before writing anything, on every problem (unknown NCCL
  version, differences between the runs, variants, ncclbench settings, missing outputs, the communicator); NCCL
  2.27.7's model time. These run on fake runs (a `job-info.txt` and empty files).
- `test_regression.py` (marked `slow`, about 1 min in parallel, 3.5 min with `-n0`): each script on a copy of
  existing runs (`full/1863257`, `full/1874191`, both Spark full runs, `quick/1863239`, `tuning/1893400` with
  `logging/1889653`). Its outputs must equal `analysis/tests/reference/`: text identical, PDFs identical
  (`SOURCE_DATE_EPOCH=0`, no creation date), or identical when rendered (`pdftoppm`) if only the bytes differ.
  A test is skipped if its run is missing from `results/`. The copy links to the run's inputs
  (`ANALYSIS_RESULTS` points the scripts at it), so the outputs in `results/` are never written to.

`uv run pytest -m "not slow"` runs only the quick ones. After a deliberate change of the outputs:
`uv run pytest -m slow --update-reference`, then check the diff of `analysis/tests/reference/` before committing.
