"""Aggregate benchmark and profiler timings of run.sbatch runs into pandas DataFrames.

Copy of profiler_results.py from the Chameleon project; only the layout of the profiler outputs differs
(one directory per node instead of master + clush-copied worker directories, see _profiler_dirs).

Layout of a run folder (one iteration), as written by run.sbatch:
    <run>/<test>_<profiler>.out     benchmark stdout (nccl-tests table or ncclbench csv)
    <run>/profiler-out.tar.gz       profiler outputs of all nodes:
        profiler-out/<profiler>/<test>/<host>/...

Profiler output formats (derived from the profilers' source code):
    comma      latency-<pid>.txt, one JSON object per op; "child_dur" [us] = first child (proxy op /
               kernel channel) start to last child end, "dur" [us] = from collective enqueue to that end
    inspector  <host>-pid<pid>.log, one JSON object per op; coll_perf.coll_exec_time_us [us, integer] =
               max kernel execution time over all channels (GPU timer)
    ncclsee    tracer mode: comm_<hash>/<name>.json, chrome trace ("traceEvents"), begin event per op;
               args.runtime_us = collective start to last kernel channel end
               profiler mode: comm_<hash>/<name>.csv with per (op, count) totals only -> no distribution

All times in the resulting DataFrames are in seconds (column time_s).

Matching against lib-ncclbench (--blocking): every iteration is barrier -> one NCCL call -> stream sync,
and the csv row is the MAX over ranks of that iteration's wall time. No other NCCL calls are issued,
so the profilers see exactly warmups + iterations calls per (operation, size) and rank, which are
matched to the benchmark iterations by order. nccl-tests only reports per-size averages (no iterations).
"""

import json
import re
import tarfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROFILERS = ["comma", "inspector", "ncclsee"]
NCCL_TESTS = ["all_reduce", "broadcast"]
NCCL_BENCH_TESTS = ["all_reduce_bench", "broadcast_bench"]

PROFILER_LABELS = {"comma": "CoMMA", "inspector": "Inspector", "ncclsee": "ncclsee"}

# default per-op metric of each profiler (see module docstring)
PROFILER_METRICS = {"comma": "child_dur", "inspector": "exec_time", "ncclsee": "runtime"}

COLUMNS = ["run", "test", "benchmark", "profiler", "source", "metric", "operation",
           "size_bytes", "rank", "host", "iteration", "time_s"]

_NCCL_DTYPE_BYTES = {
    "ncclInt8": 1, "ncclUint8": 1, "ncclFloat8e4m3": 1, "ncclFloat8e5m2": 1,
    "ncclFloat16": 2, "ncclBfloat16": 2,
    "ncclInt32": 4, "ncclUint32": 4, "ncclFloat32": 4,
    "ncclInt64": 8, "ncclUint64": 8, "ncclFloat64": 8,
}

_OP_NAMES = {
    # CoMMA uses snake_case already; Inspector/ncclsee use NCCL's function names, ncclbench the API names
    "AllReduce": "all_reduce", "ncclAllReduce": "all_reduce",
    "Broadcast": "broadcast", "ncclBroadcast": "broadcast",
    "AllGather": "all_gather", "ncclAllGather": "all_gather",
    "ReduceScatter": "reduce_scatter", "ncclReduceScatter": "reduce_scatter",
    "Reduce": "reduce", "ncclReduce": "reduce",
    "Send": "send", "Recv": "recv",
}


def _op_name(name):
    return _OP_NAMES.get(name, name)


def _test_operation(test):
    return test.removesuffix("_bench")


def _json_lines(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------------------------
# profiler outputs
# ---------------------------------------------------------------------------------------------

def load_comma(directory, metric=PROFILER_METRICS["comma"]):
    """metric: "child_dur" (execution, default) or "dur" (from enqueue)"""
    rows = []
    for path in sorted(Path(directory).glob("latency-*.txt")):
        for rec in _json_lines(path):
            if rec.get("cat") != "COLL" or rec.get("reclaimed"):
                # reclaimed ops timed out without completing and have no child_dur
                continue
            if metric not in rec:
                continue
            rows.append({
                "operation": _op_name(rec["name"]),
                "size_bytes": rec["args"]["size"],
                "rank": rec["rank"],
                "seq": rec.get("seq_num"),
                "start": rec["ts"],
                "time_s": rec[metric] * 1e-6,
            })
    return pd.DataFrame(rows), metric


def load_inspector(directory, metric=PROFILER_METRICS["inspector"]):
    rows = []
    for path in sorted(Path(directory).glob("*.log")):
        for rec in _json_lines(path):
            perf = rec.get("coll_perf")
            if perf is None:
                continue
            rows.append({
                "operation": _op_name(perf["coll"]),
                "size_bytes": perf["coll_msg_size_bytes"],
                "rank": rec["header"]["rank"],
                "seq": perf["coll_sn"],
                "start": None,
                "time_s": perf["coll_exec_time_us"] * 1e-6,
            })
    return pd.DataFrame(rows), metric


def _ncclsee_trace_events(path):
    # parsed line by line: every event is on its own line, and this also works for traces whose
    # closing brackets are missing (the tracer only writes them when the process exits cleanly)
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line.startswith("{") and line not in ("{", "{}"):
                yield json.loads(line)


def load_ncclsee(directory, metric=PROFILER_METRICS["ncclsee"]):
    """metric: "runtime" (start to last kernel channel end, default) or "eventDur" (collective event only)"""
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("comm_*/*.json")):
        for ev in _ncclsee_trace_events(path):
            if ev.get("ph") != "b" or ev.get("cat") != "COLL":
                continue
            args = ev["args"]
            rows.append({
                "operation": _op_name(ev["name"]),
                "size_bytes": args["count"] * _NCCL_DTYPE_BYTES.get(args["datatype"], np.nan),
                "rank": args["rank"],
                "seq": args.get("seqNumber"),
                "start": ev["ts"],
                "time_s": args[f"{metric}_us"] * 1e-6,
            })
    if not rows and any(directory.glob("comm_*/*.csv")):
        warnings.warn(f"{directory}: only ncclsee profiler-mode csv (per-size totals) found; "
                      "use NCCLSEE_MODE=tracer (or full) for per-op times. Returning per-size means.")
        return load_ncclsee_csv(directory), "mean_runtime"
    return pd.DataFrame(rows), metric


def load_ncclsee_csv(directory):
    """ncclsee profiler mode: one row per (op, count) with the mean runtime per call; no rank info"""
    frames = []
    for path in sorted(Path(directory).glob("comm_*/*.csv")):
        df = pd.read_csv(path)
        frames.append(pd.DataFrame({
            "operation": df["Operation"].map(_op_name),
            "size_bytes": df["Count"] * df["Datatype"].map(_NCCL_DTYPE_BYTES),
            "rank": np.nan,
            "seq": np.nan,
            "start": np.nan,
            "time_s": df["Seconds"] / df["#Calls"],
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


PROFILER_LOADERS = {"comma": load_comma, "inspector": load_inspector, "ncclsee": load_ncclsee}


# ---------------------------------------------------------------------------------------------
# benchmark outputs
# ---------------------------------------------------------------------------------------------

def load_nccl_tests(path):
    """per-size average times (out-of-place and in-place) of an nccl-tests run"""
    rows = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) != 13:
                continue
            size = int(fields[0])
            rows.append({"size_bytes": size, "metric": "time_oop", "time_s": float(fields[5]) * 1e-6})
            rows.append({"size_bytes": size, "metric": "time_ip", "time_s": float(fields[9]) * 1e-6})
    return pd.DataFrame(rows)


def load_ncclbench(path):
    """per-iteration times of a lib-ncclbench --blocking --csv run (max over ranks per iteration)"""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "operation": df["Operation"].map(_op_name),
        "size_bytes": df["Msg_Size_B"],
        "time_s": df["Time_us"] * 1e-6,
    })
    out["iteration"] = out.groupby(["operation", "size_bytes"]).cumcount()
    out["metric"] = "time_iter"
    return out


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------

def extract_profiler_outputs(run_dir):
    """extracts <run>/profiler-out.tar.gz (once) and returns the extraction directory"""
    run_dir = Path(run_dir)
    target = run_dir / "profiler-out-extracted"
    if not target.exists():
        with tarfile.open(run_dir / "profiler-out.tar.gz") as tar:
            tar.extractall(target, filter="data")
    return target


def _profiler_dirs(root, profiler, test):
    """(host, directory) of all nodes for one profiler/test"""
    return [(d.name, d) for d in sorted((root / "profiler-out" / profiler / test).glob("*")) if d.is_dir()]


def _assign_iterations(df, warmups, iterations, context):
    """match profiler records to benchmark iterations by order, per (operation, size, rank)"""
    if df.empty or df["rank"].isna().all():
        return df.assign(iteration=np.nan)
    out = []
    for key, g in df.groupby(["operation", "size_bytes", "rank"]):
        g = g.sort_values(["seq", "start"], na_position="last")
        n = len(g)
        if n != warmups + iterations:
            warnings.warn(f"{context}: {key}: {n} profiler records, expected {warmups}+{iterations}; "
                          f"keeping the last {min(n, iterations)}")
        # the last `iterations` records are the measured ones; if records are missing, align them to the end
        g = g.iloc[max(0, n - iterations):]
        out.append(g.assign(iteration=np.arange(len(g)) + (iterations - len(g))))
    return pd.concat(out)


def load_run(run_dir, profilers=PROFILERS, nccl_tests=NCCL_TESTS, nccl_bench_tests=NCCL_BENCH_TESTS,
             warmups=5, iterations=50, metrics=None):
    """
    long-format DataFrame (see COLUMNS) with benchmark and profiler times of one run folder

    warmups, iterations: ncclbench settings of the run, used to match profiler records to iterations
    metrics: optional {profiler: metric} overriding PROFILER_METRICS
    """
    run_dir = Path(run_dir)
    metrics = {**PROFILER_METRICS, **(metrics or {})}
    root = extract_profiler_outputs(run_dir)
    frames = []

    for profiler in profilers:
        for test in list(nccl_tests) + list(nccl_bench_tests):
            is_bench = test in nccl_bench_tests
            meta = {"run": run_dir.name, "test": test, "profiler": profiler,
                    "benchmark": "ncclbench" if is_bench else "nccl-tests"}

            # benchmark side
            out_file = run_dir / f"{test}_{profiler}.out"
            if out_file.exists():
                if is_bench:
                    b = load_ncclbench(out_file)
                else:
                    b = load_nccl_tests(out_file).assign(operation=_test_operation(test), iteration=np.nan)
                frames.append(b.assign(source="benchmark", rank=np.nan, host=None, **meta))
            else:
                warnings.warn(f"missing benchmark output {out_file}")

            # profiler side, all nodes
            for host, directory in _profiler_dirs(root, profiler, test):
                p, metric = PROFILER_LOADERS[profiler](directory, metric=metrics[profiler])
                if p.empty:
                    continue
                # only the benchmarked operation (e.g. drops anything else NCCL might report)
                p = p[p["operation"] == _test_operation(test)]
                if is_bench:
                    p = _assign_iterations(p, warmups, iterations, f"{run_dir.name}/{profiler}/{test}")
                else:
                    p = p.assign(iteration=np.nan)
                frames.append(p.assign(source="profiler", metric=metric, host=host, **meta))

    df = pd.concat(frames, ignore_index=True)
    return df[COLUMNS]


def load_runs(run_dirs, **kwargs):
    return pd.concat([load_run(d, **kwargs) for d in run_dirs], ignore_index=True)


def max_over_ranks(df):
    """profiler times reduced to the max over ranks per iteration (same reduction as ncclbench)"""
    prof = df[(df["source"] == "profiler") & df["iteration"].notna()]
    keys = ["run", "test", "benchmark", "profiler", "source", "metric", "operation", "size_bytes", "iteration"]
    reduced = prof.groupby(keys, as_index=False)["time_s"].max().assign(rank=np.nan, host=None)
    return pd.concat([df[df["source"] == "benchmark"], reduced], ignore_index=True)[COLUMNS]


def _nccl_tests_calls_per_size_and_rank(path):
    """
    (NCCL calls per size and rank, number of ranks) of an nccl-tests run, from its output header

    nccl-tests (src/common.cu) issues per size and rank: `warmup iters` calls in the warm-up loop over all sizes,
    then two BenchTime passes (out-of-place, in-place), each with 1 sync call + `iters` * `agg iters` timed calls
    + `validation` check calls; with the defaults 1 + 2 * (1 + 20 + 1) = 45
    """
    text = Path(path).read_text()
    m = re.search(r"warmup iters: (\d+) iters: (\d+) agg iters: (\d+) validation: (\d+)", text)
    if m is None:
        raise ValueError(f"{path}: no nccl-tests parameter line in the header")
    warmup, iters, agg, validation = map(int, m.groups())
    n_ranks = len(re.findall(r"^#\s+Rank\s+\d+\s+Group", text, re.M))
    return warmup + 2 * (1 + iters * agg + validation), n_ranks


def record_counts(run_dir, profilers=PROFILERS, nccl_tests=NCCL_TESTS, nccl_bench_tests=NCCL_BENCH_TESTS, warmups=5):
    """
    number of NCCL calls each benchmark issued (all ranks, incl. warm-ups) vs. the number of records each profiler
    has for them, per (test, profiler); one row each with columns expected, recorded, recorded_pct

    recorded = profiler records with a usable time for the benchmarked operation and sizes (e.g. CoMMA ops that were
    reclaimed after its op timeout have no time and count as missed)
    expected: nccl-tests from its output header (see _nccl_tests_calls_per_size_and_rank); ncclbench (--blocking)
    issues `warmups` + one call per csv row and size, on every rank. The number of ranks is taken from the nccl-tests
    output of the same profiler run (all tests run on the same hosts).
    """
    run_dir = Path(run_dir)
    root = extract_profiler_outputs(run_dir)
    rows = []
    for profiler in profilers:
        n_ranks = None
        for test in nccl_tests:
            out_file = run_dir / f"{test}_{profiler}.out"
            if out_file.exists():
                n_ranks = _nccl_tests_calls_per_size_and_rank(out_file)[1]
                break
        for test in list(nccl_tests) + list(nccl_bench_tests):
            is_bench = test in nccl_bench_tests
            operation = _test_operation(test)
            out_file = run_dir / f"{test}_{profiler}.out"
            if not out_file.exists():
                warnings.warn(f"missing benchmark output {out_file}")
                continue
            if is_bench:
                bench = load_ncclbench(out_file)
                sizes = set(bench["size_bytes"])
                expected_per_rank = warmups * len(sizes) + len(bench)
                ranks = n_ranks
            else:
                sizes = set(load_nccl_tests(out_file)["size_bytes"])
                calls, ranks = _nccl_tests_calls_per_size_and_rank(out_file)
                expected_per_rank = calls * len(sizes)
            if ranks is None:
                warnings.warn(f"{run_dir}/{profiler}: unknown number of ranks (no nccl-tests output), skipping {test}")
                continue

            recorded = 0
            for _, directory in _profiler_dirs(root, profiler, test):
                p, _ = PROFILER_LOADERS[profiler](directory)
                if not p.empty:
                    recorded += int(((p["operation"] == operation) & p["size_bytes"].isin(sizes)).sum())
            expected = expected_per_rank * ranks
            rows.append({"run": run_dir.name, "test": test, "benchmark": "ncclbench" if is_bench else "nccl-tests",
                         "profiler": profiler, "operation": operation, "ranks": ranks,
                         "expected": expected, "recorded": recorded, "recorded_pct": 100 * recorded / expected})
    return pd.DataFrame(rows)


def size_label(size_bytes):
    """8 -> "8 B", 1024 -> "1 KiB", 2**25 -> "32 MiB" (sizes that are not a whole multiple keep the smaller unit)"""
    for unit, factor in [("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]:
        if size_bytes >= factor and size_bytes % factor == 0:
            return f"{size_bytes // factor} {unit}"
    return f"{size_bytes} B"


def reported_time_tables(df, profilers=PROFILERS):
    """
    {(benchmark, operation): table} with the mean times per message size, in microseconds; rows: sizes the
    benchmark ran; columns: per profiler a pair (profiler, benchmark): the time the profiler reported and the time
    the benchmark itself reported in the same run (each profiler has its own benchmark run); NaN where there are no
    records for a size

    The mean is the statistic nccl-tests reports (src/common.cu: elapsed time / iterations per rank, then the
    average over all ranks), so all of a profiler's records for a size are averaged, over all ranks and all runs in
    `df`. For nccl-tests these include its warm-up, sync and validation calls (the profilers cannot tell them apart);
    for ncclbench the warm-ups are already dropped by load_run. Benchmark side: nccl-tests = mean of its
    out-of-place and in-place averages (the profiler records contain both kinds of calls); ncclbench = mean over its
    per-iteration times (each the max over ranks).
    """
    tables = {}
    for (benchmark, operation), sel in df.groupby(["benchmark", "operation"], sort=False):
        sizes = sorted(sel.loc[sel["source"] == "benchmark", "size_bytes"].unique())
        means = (sel.groupby(["size_bytes", "profiler", "source"])["time_s"].mean().mul(1e6)
                    .unstack(["profiler", "source"]))
        columns = pd.MultiIndex.from_product([profilers, ["profiler", "benchmark"]])
        table = means.reindex(index=sizes, columns=columns)
        table.index = pd.Index([size_label(int(s)) for s in table.index], name="size")
        table.columns = pd.MultiIndex.from_tuples([(PROFILER_LABELS.get(p, p), s) for p, s in table.columns],
                                                  names=["mean time [us]", "reported by"])
        tables[(benchmark, operation)] = table
    order = {op: i for i, op in enumerate(OPERATION_LABELS)}
    return dict(sorted(tables.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))))


def summary(df, rank=None):
    """median and mean per (test, profiler, source, metric, size); rank=None uses all ranks"""
    if rank is not None:
        df = df[(df["source"] == "benchmark") | (df["rank"] == rank)]
    return (df.groupby(["test", "profiler", "source", "metric", "size_bytes"])["time_s"]
              .agg(["count", "median", "mean", "min", "max"]))


# ---------------------------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------------------------

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
}
COLORS = {"benchmark": "#6baed6", "profiler": "#fdae6b"}
PROFILER_COLORS = {"comma": "#fdae6b", "inspector": "#a1d99b", "ncclsee": "#9ecae1"}
OPERATION_LABELS = {"all_reduce": "AllReduce", "broadcast": "Broadcast", "all_gather": "AllGather",
                    "reduce_scatter": "ReduceScatter", "reduce": "Reduce"}
EDGE_COLOR = "#606060"


def _half_violin(ax, data, position, side, color, width):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return
    sign = -1 if side == "left" else 1
    if len(data) == 1 or np.ptp(data) == 0:
        # no distribution to show, only the median line below
        pass
    else:
        body = ax.violinplot([data], positions=[position], widths=width, showextrema=False)["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], position, np.inf) if side == "right" else np.clip(verts[:, 0], -np.inf, position)
        body.set_facecolor(color)
        body.set_edgecolor(EDGE_COLOR)
        body.set_linewidth(1)
        body.set_alpha(1)
    median = np.median(data)
    ax.plot([position, position + sign * width / 4], [median, median], color="black", linewidth=2, zorder=3)


def plot_split_violins(df, test, size_bytes, profilers=PROFILERS, profiler_ranks="max",
                       ax=None, width=0.8, legend_loc="upper right"):
    """
    split violin per profiler: left half benchmark times, right half profiler times (like figure_example.png)

    test: e.g. "all_reduce_bench" (per-iteration benchmark times) or "all_reduce" (nccl-tests: per-size
          averages, so the benchmark half is a distribution over runs)
    profiler_ranks: "max" (max over ranks per iteration, same reduction as ncclbench; ncclbench only),
                    "all" (every rank's record), or a rank number
    """
    sel = df[(df["test"] == test) & (df["size_bytes"] == size_bytes)]
    if profiler_ranks == "max":
        if sel["benchmark"].eq("ncclbench").all():
            sel = max_over_ranks(sel)
        else:
            warnings.warn("profiler_ranks='max' needs per-iteration data (ncclbench); using all ranks")
    elif profiler_ranks != "all":
        sel = sel[(sel["source"] == "benchmark") | (sel["rank"] == profiler_ranks)]

    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        positions = np.arange(len(profilers))
        for pos, profiler in zip(positions, profilers):
            s = sel[sel["profiler"] == profiler]
            _half_violin(ax, s.loc[s["source"] == "benchmark", "time_s"], pos, "left", COLORS["benchmark"], width)
            _half_violin(ax, s.loc[s["source"] == "profiler", "time_s"], pos, "right", COLORS["profiler"], width)

        ax.set_xticks(positions, [PROFILER_LABELS.get(p, p) for p in profilers])
        for label in ax.get_xticklabels():
            if label.get_text() == "ncclsee":
                label.set_fontfamily("monospace")
        ax.set_ylabel("Time [s]")
        ax.legend(handles=[Patch(facecolor=COLORS["benchmark"], edgecolor=EDGE_COLOR, label="Benchmark"),
                           Patch(facecolor=COLORS["profiler"], edgecolor=EDGE_COLOR, label="Profiler")],
                  loc=legend_loc)
    return ax


def _percent_label(value):
    # one decimal at most (integers without); anything below 100 must not be rounded up to "100",
    # which would hide that a profiler missed calls
    label = f"{value:.1f}"
    if value < 100 and float(label) >= 100:
        label = "99.9"
    return label.rstrip("0").rstrip(".")


def plot_recorded_operations(counts, profilers=PROFILERS, ax=None, width=0.25, legend=True):
    """
    grouped bar plot: share of the NCCL calls each benchmark issued that each profiler recorded, per operation and
    benchmark (like bar_plot_example.png); `counts` from record_counts, rows of several runs/iterations are summed
    """
    agg = counts.groupby(["benchmark", "operation", "profiler"])[["expected", "recorded"]].sum()
    groups = list(dict.fromkeys((b, o) for b, o, _ in agg.index))  # (benchmark, operation) in order of appearance
    groups.sort(key=lambda g: (list(OPERATION_LABELS).index(g[1]) if g[1] in OPERATION_LABELS else 99, g[0]))
    x = np.arange(len(groups))

    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(1.6 * len(groups) + 1.5, 3.2))
        for k, profiler in enumerate(profilers):
            pct = [100 * agg.loc[(b, o, profiler), "recorded"] / agg.loc[(b, o, profiler), "expected"]
                   if (b, o, profiler) in agg.index else np.nan for b, o in groups]
            bars = ax.bar(x + (k - (len(profilers) - 1) / 2) * width, pct, width,
                          color=PROFILER_COLORS.get(profiler), edgecolor=EDGE_COLOR, linewidth=1,
                          label=PROFILER_LABELS.get(profiler, profiler), zorder=2)
            ax.bar_label(bars, labels=["" if np.isnan(v) else _percent_label(v) for v in pct], padding=1)

        ax.axhline(100, color="gray", linestyle="-.", linewidth=1, zorder=1)
        ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x, [f"{OPERATION_LABELS.get(o, o)}\n({b})" for b, o in groups])
        ax.set_ylabel("Recorded Operations [%]")
        if legend:
            leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(profilers))
            for text in leg.get_texts():
                if text.get_text() == "ncclsee":
                    text.set_fontfamily("monospace")
    return ax
