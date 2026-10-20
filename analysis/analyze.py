"""Plots and tables of one run (the evaluation cells of nccl_bench.ipynb in the Chameleon project).

    uv run analysis/analyze.py [results/<experiment>/<job>]      default: newest run of any experiment

A run is one array job of an experiment; its iteration folders (01, 02, ...) are the array tasks. The profilers,
tests and ncclbench settings are taken from the run itself (output file names, job-info.txt). Writes into the run
folder: violins.pdf per iteration, recorded_operations.pdf and reported_times.txt.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only files, no windows
import matplotlib.pyplot as plt

import pandas as pd

import profiler_results as pr

RESULTS = Path(__file__).resolve().parent.parent / "results"

# message sizes of the violin plots (those the run has; the smallest and largest size if it has none of them)
VIOLIN_SIZES = [8, 2**10, 2**25, 2**30]


def run_settings(iteration_dir):
    """profilers, nccl-tests tests, ncclbench tests and ncclbench warmups/iterations of an iteration"""
    profilers, nccl_tests, bench_tests = set(), set(), set()
    for f in iteration_dir.glob("*.out"):
        test, profiler = f.stem.rsplit("_", 1)
        profilers.add(profiler)
        (bench_tests if test.endswith("_bench") else nccl_tests).add(test)
    # ncclbench arguments as recorded by experiment.sh
    info = (iteration_dir / "job-info.txt").read_text()
    warmups = int(re.search(r"--warmups (\d+)", info).group(1))
    iterations = int(re.search(r"--iterations (\d+)", info).group(1))
    return {"profilers": [p for p in pr.PROFILERS if p in profilers], "nccl_tests": sorted(nccl_tests),
            "nccl_bench_tests": sorted(bench_tests), "warmups": warmups, "iterations": iterations}


def plot_iteration(iteration_dir, settings):
    # benchmark vs. profiler latency per op; only the ncclbench tests report per-iteration times
    df = pr.load_run(iteration_dir, **settings)
    tests = settings["nccl_bench_tests"]
    run_sizes = sorted(df.loc[df["source"] == "benchmark", "size_bytes"].unique())
    sizes = [s for s in VIOLIN_SIZES if s in run_sizes] or [run_sizes[0], run_sizes[-1]]
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(len(tests), len(sizes), figsize=(4*len(sizes), 3*len(tests)), layout="constrained",
                                squeeze=False)
        for row, test in zip(axs, tests):
            operation = pr.OPERATION_LABELS.get(test.removesuffix("_bench"), test)
            for ax, size in zip(row, sizes):
                pr.plot_split_violins(df, test, size, profilers=settings["profilers"], ax=ax)
                ax.set_title(f"{operation}, {pr.size_label(int(size))}")
                if ax is not axs[0][0]:
                    ax.get_legend().remove()
        fig.suptitle(f"Benchmark vs. profiler latency ({iteration_dir})")
        fig.savefig(iteration_dir / "violins.pdf")
    plt.close(fig)


def plot_recorded_operations(run, iteration_dirs, settings):
    # share of the NCCL calls each benchmark issued (all ranks, incl. warm-ups) that each profiler recorded with a
    # usable time, summed over all iterations
    counts = pd.concat([pr.record_counts(d, profilers=settings["profilers"], nccl_tests=settings["nccl_tests"],
                                         nccl_bench_tests=settings["nccl_bench_tests"], warmups=settings["warmups"])
                        for d in iteration_dirs])
    print(counts.pivot_table(index=["benchmark", "operation"], columns=["profiler", "run"], values="recorded_pct")
                .round(1).to_string())

    ax = pr.plot_recorded_operations(counts, profilers=settings["profilers"])
    ax.set_title(f"{run} ({counts['run'].nunique()} iterations)", pad=30)
    ax.figure.savefig(run / "recorded_operations.pdf", bbox_inches="tight")
    plt.close(ax.figure)


def reported_time_tables(run, iteration_dirs, settings):
    # mean times per message size [us], per benchmark and operation, over all iterations: per profiler the time it
    # reported and the time the benchmark itself reported in the same run (each profiler has its own run).
    # nccl-tests reports the mean as well (per rank: elapsed time / iterations, then averaged over ranks), so all
    # profiler records of a size are averaged over all ranks; for nccl-tests they include its warm-up/sync/validation calls
    df_all = pr.load_runs(iteration_dirs, **settings)
    text = []
    for (benchmark, operation), table in pr.reported_time_tables(df_all, profilers=settings["profilers"]).items():
        caption = f"{pr.OPERATION_LABELS.get(operation, operation)} ({benchmark}), {run.name}, {len(iteration_dirs)} iteration(s)"
        text.append(f"{caption}\n{table.to_string(float_format=lambda v: f'{v:,.1f}', na_rep='-')}\n")
    print("\n".join(text))
    (run / "reported_times.txt").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="run folder (default: newest in results/*/)")
    args = parser.parse_args()
    # job ids increase, so the newest run is the one with the highest id
    run = args.run or max((p for p in RESULTS.glob("*/*") if p.name.isdigit()), key=lambda p: int(p.name))

    iteration_dirs = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and p.name.isdigit()):
        if (d / "profiler-out.tar.gz").exists():
            iteration_dirs.append(d)
        else:
            print(f"{d}: no profiler output (failed iteration?), skipped")
    if not iteration_dirs:
        raise SystemExit(f"{run}: no iterations with profiler output")
    settings = run_settings(iteration_dirs[0])
    print(f"{run}: {settings}")

    for d in iteration_dirs:
        print(f"{d}: violins.pdf")
        plot_iteration(d, settings)
    plot_recorded_operations(run, iteration_dirs, settings)
    reported_time_tables(run, iteration_dirs, settings)


if __name__ == "__main__":
    main()
