"""Plots and tables of one run (the evaluation cells of nccl_bench.ipynb in the Chameleon project).

    uv run analysis/analyze.py [results/<run>]      default: newest run in results/

A run is one array job of run.sbatch; its iteration folders (01, 02, ...) are the array tasks. Writes into the
run folder: violins.pdf per iteration, recorded_operations.pdf and reported_times.txt.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only files, no windows
import matplotlib.pyplot as plt

import pandas as pd

import profiler_results as pr

RESULTS = Path(__file__).resolve().parent.parent / "results"

# benchmark vs. profiler latency per op; only the ncclbench tests report per-iteration times (nccl-tests only averages)
TESTS = {"all_reduce_bench": "AllReduce", "broadcast_bench": "Broadcast"}
SIZES = {"8 B": 8, "1 KiB": 2**10, "32 MiB": 2**25, "1 GiB": 2**30}


def plot_iteration(iteration_dir):
    df = pr.load_run(iteration_dir)
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(len(TESTS), len(SIZES), figsize=(4*len(SIZES), 3*len(TESTS)), layout="constrained")
        for row, (test, operation) in zip(axs, TESTS.items()):
            for ax, (size_label, size) in zip(row, SIZES.items()):
                pr.plot_split_violins(df, test, size, ax=ax)
                ax.set_title(f"{operation}, {size_label}")
                if ax is not axs[0][0]:
                    ax.get_legend().remove()
        fig.suptitle(f"Benchmark vs. profiler latency ({iteration_dir})")
        fig.savefig(iteration_dir / "violins.pdf")
    plt.close(fig)


def plot_recorded_operations(run, iteration_dirs):
    # share of the NCCL calls each benchmark issued (all ranks, incl. warm-ups) that each profiler recorded with a
    # usable time, summed over all iterations
    counts = pd.concat([pr.record_counts(d) for d in iteration_dirs])
    print(counts.pivot_table(index=["benchmark", "operation"], columns=["profiler", "run"], values="recorded_pct")
                .round(1).to_string())

    ax = pr.plot_recorded_operations(counts)
    ax.set_title(f"{run} ({counts['run'].nunique()} iterations)", pad=30)
    ax.figure.savefig(run / "recorded_operations.pdf", bbox_inches="tight")
    plt.close(ax.figure)


def reported_time_tables(run, iteration_dirs):
    # mean times per message size [us], per benchmark and operation, over all iterations: per profiler the time it
    # reported and the time the benchmark itself reported in the same run (each profiler has its own run).
    # nccl-tests reports the mean as well (per rank: elapsed time / iterations, then averaged over ranks), so all
    # profiler records of a size are averaged over all ranks; for nccl-tests they include its warm-up/sync/validation calls
    df_all = pr.load_runs(iteration_dirs)
    text = []
    for (benchmark, operation), table in pr.reported_time_tables(df_all).items():
        caption = f"{pr.OPERATION_LABELS.get(operation, operation)} ({benchmark}), {run.name}, {len(iteration_dirs)} iteration(s)"
        text.append(f"{caption}\n{table.to_string(float_format=lambda v: f'{v:,.1f}', na_rep='-')}\n")
    print("\n".join(text))
    (run / "reported_times.txt").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="run folder (default: newest in results/)")
    args = parser.parse_args()
    run = args.run or max((p for p in RESULTS.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))

    iteration_dirs = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and p.name.isdigit()):
        if (d / "profiler-out.tar.gz").exists():
            iteration_dirs.append(d)
        else:
            print(f"{d}: no profiler output (failed iteration?), skipped")
    if not iteration_dirs:
        raise SystemExit(f"{run}: no iterations with profiler output")

    for d in iteration_dirs:
        print(f"{d}: violins.pdf")
        plot_iteration(d)
    plot_recorded_operations(run, iteration_dirs)
    reported_time_tables(run, iteration_dirs)


if __name__ == "__main__":
    main()
