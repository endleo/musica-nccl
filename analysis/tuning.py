"""Evaluation of a tuning run (experiments/tuning.sbatch): NCCL's own choice vs. pinned algorithm/protocol.

    uv run analysis/analyze.py results/tuning/<job>      analyze.py hands runs with variants to this module
    uv run analysis/tuning.py [results/tuning/<job>]     the same; default: newest tuning run

Layout of a run: <run>/<round>/<variant>/ with the ncclbench outputs and profiler-out.tar.gz (ncclsee) of one variant,
<run>/job-info.txt for the whole job. Every variant runs AllReduce and Broadcast; the tree_* variants only pin
AllReduce, so their Broadcast runs with NCCL's choice.

Writes into the run folder:
    choices.txt               algorithm/protocol/channels/warps ncclsee recorded per variant and size (all ranks,
                              all calls, all rounds), checked against what was pinned; NCCL's choice compared with
                              the logging run and with the cost model
    model_vs_measured.txt     per operation, size and combination: NCCL's estimate and the measured median/mean,
                              and where the estimate lies in the measured distribution (percentile)
    model_vs_measured.pdf     overview over all sizes: measured distribution (box) and NCCL's estimate (point),
                              both relative to the measured median
    model_violins.pdf         selected sizes, absolute times: violins of the measured times and the estimate
    round_consistency.pdf     per variant, how far each round's median is from the median over the rounds
    pinned_vs_default.pdf     median latency of every variant, and pinned / default per size
    best_vs_default.txt       per size: NCCL's choice, the fastest measured combination and the difference

Statistics as in analyze.py: ncclbench's per-call times (max over ranks), warm-ups dropped; medians and quartiles
over the 50 iterations of each of the 3 rounds together (150 calls per size and variant) unless stated otherwise.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only files, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import profiler_results as pr

RESULTS = (Path(__file__).resolve().parent.parent / "results").resolve()
# the logging run whose TUNING table (latency/bandwidth per combination) and choices the tuning run is compared with
LOGGING_RUN = RESULTS / "logging" / "1889653" / "01"

TESTS = {"all_reduce_bench": "all_reduce", "broadcast_bench": "broadcast"}
ALGOS = ["tree", "ring"]
PROTOS = ["ll", "ll128", "simple"]
PROTO_LABELS = {"ll": "LL", "ll128": "LL128", "simple": "Simple"}
ALGO_LABELS = {"tree": "Tree", "ring": "Ring"}
# combinations available per operation on MUSICA (TUNING table of the logging run, everything else 0)
COMBINATIONS = {"all_reduce": [(a, p) for a in ALGOS for p in PROTOS], "broadcast": [("ring", p) for p in PROTOS]}

# protocol = color (checked with the dataviz validator: CVD and normal-vision separation, contrast on white),
# algorithm = line style and marker; NCCL's choice ("default") in black
PROTO_COLORS = {"ll": "#1b9e77", "ll128": "#d95f02", "simple": "#7570b3"}
ALGO_LINESTYLES = {"tree": "-", "ring": "--"}
ALGO_MARKERS = {"tree": "o", "ring": "s"}
DEFAULT_COLOR = "black"
ROUND_COLORS = pr.ITERATION_COLORS
ROUND_MARKERS = pr.ITERATION_MARKERS

# sizes of model_violins.pdf: small (latency-bound), the switch points of the model, large (bandwidth-bound)
VIOLIN_SIZES = {"all_reduce": [8, 2**16, 2**20, 2**21, 2**26, 2**30], "broadcast": [8, 2**16, 2**19, 2**20, 2**26, 2**30]}


def label(path):
    """short name of a run folder for figure titles: results/tuning/1893400 -> tuning/1893400"""
    path = Path(path).resolve()
    return str(path.relative_to(RESULTS)) if path.is_relative_to(RESULTS) else str(path)


def variant_name(algo, proto):
    return f"{algo}_{proto}"


def combination_label(algo, proto):
    return f"{ALGO_LABELS[algo]}/{PROTO_LABELS[proto]}"


def pinned_combination(variant, operation):
    """(algo, proto) a variant pins for an operation, None if the operation runs with NCCL's choice"""
    if variant == "default":
        return None
    algo, proto = variant.split("_")
    if operation == "broadcast" and algo == "tree":
        return None  # the tree_* variants only pin AllReduce
    return algo, proto


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

def variant_dirs(run):
    """[(round, variant, directory)] of a tuning run, in round order"""
    out = []
    for round_dir in sorted(p for p in Path(run).iterdir() if p.is_dir() and p.name.isdigit()):
        for d in sorted(p for p in round_dir.iterdir() if p.is_dir()):
            if (d / "profiler-out.tar.gz").exists():
                out.append((round_dir.name, d.name, d))
            else:
                print(f"{d}: no profiler output (failed run?), skipped")
    return out


def ncclbench_settings(run):
    """warmups, iterations from the job info"""
    info = (Path(run) / "job-info.txt").read_text()
    return int(re.search(r"--warmups (\d+)", info).group(1)), int(re.search(r"--iterations (\d+)", info).group(1))


def load_times(run):
    """
    benchmark and ncclsee times of all variants and rounds (profiler_results.load_run per variant folder), with
    columns round and variant; run = round, so that the per-run helpers of profiler_results treat rounds as runs
    """
    warmups, iterations = ncclbench_settings(run)
    frames = []
    for rnd, variant, d in variant_dirs(run):
        df = pr.load_run(d, profilers=["ncclsee"], nccl_tests=[], nccl_bench_tests=list(TESTS),
                         warmups=warmups, iterations=iterations)
        frames.append(df.assign(run=rnd, round=rnd, variant=variant))
    return pd.concat(frames, ignore_index=True)


def benchmark_times(df):
    """ncclbench's per-call times [us] (already the max over ranks): round, variant, operation, size_bytes, time_us"""
    b = df[df["source"] == "benchmark"]
    return b[["round", "variant", "operation", "size_bytes", "iteration"]].assign(time_us=b["time_s"] * 1e6)


def ncclsee_calls(directory):
    """algorithm, protocol, channels and warps of every COLL event ncclsee recorded (all ranks) below `directory`"""
    rows = []
    for path in sorted(Path(directory).glob("*/comm_*/*.json")):
        for ev in pr._ncclsee_trace_events(path):
            if ev.get("ph") != "b" or ev.get("cat") != "COLL":
                continue
            a = ev["args"]
            rows.append({"operation": pr._op_name(ev["name"]),
                         "size_bytes": a["count"] * pr._NCCL_DTYPE_BYTES[a["datatype"]], "rank": a["rank"],
                         "algo": a["algo"].lower(), "proto": a["protocol"].lower(), "channels": a["nChannels"],
                         "warps": a["nWarps"]})
    return pd.DataFrame(rows)


def load_choices(run):
    """ncclsee's records of what NCCL did, per round, variant and test"""
    frames = []
    for rnd, variant, d in variant_dirs(run):
        root = pr.extract_profiler_outputs(d) / "profiler-out" / "ncclsee"
        for test in TESTS:
            frames.append(ncclsee_calls(root / test).assign(round=rnd, variant=variant, test=test))
    return pd.concat(frames, ignore_index=True)


def logging_choices(logging_run=LOGGING_RUN):
    root = pr.extract_profiler_outputs(logging_run) / "profiler-out" / "ncclsee"
    return pd.concat([ncclsee_calls(root / test).assign(test=test) for test in TESTS], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# NCCL's cost model (NCCL 2.27.7, src/graph/tuning.cc, ncclTopoGetAlgoTime; see report/logging/summary.typ)
# ---------------------------------------------------------------------------------------------

# treeCorrectionFactor[protocol][log2(bytes / 64)], protocols LL, LL128, Simple
TREE_CORRECTION = {
    "ll":     [1.0, 1.0, 1.0, 1.0, .9, .8, .7, .7, .7, .7, .6, .5, .4, .4, .5, .6, .7, .8, .9, 1.0, 1.0, 1.0, 1.0],
    "ll128":  [1.0, 1.0, 1.0, 1.0, 1.0, .9, .8, .8, .8, .7, .6, .6, .6, .6, .6, .6, .8, .9, .9, .9, .9, 1.0, 1.0],
    "simple": [.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5, .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9],
}
# communicator of the runs (logging run: 8 channels, 8 ranks on 2 nodes, H100 = compute capability 90)
COMM_CHANNELS, COMM_RANKS, COMM_NODES = 8, 8, 2


def model_table(logging_run=LOGGING_RUN):
    """
    {(operation, algo, proto): (latency [us], bandwidth [GB/s])} from the TUNING table in rank 0's log of the
    logging run (first block: Tree, Ring, CollNetDirect, each LL/LL128/Simple; printed with one decimal)
    """
    log = next((pr.extract_profiler_outputs(logging_run) / "profiler-out" / "ncclsee" / "all_reduce_bench")
               .glob("*/nccl-rank0.log"))
    table = {}
    for line in log.read_text().splitlines():
        m = re.search(r"NCCL INFO\s+(AllReduce|Broadcast) \|(.*)", line)
        if not m or pr._op_name(m.group(1)) in {op for op, _, _ in table}:
            continue  # only the first block (Tree, Ring)
        cells = [c.split("/") for c in m.group(2).split("|") if c.strip()]
        for i, (algo, proto) in enumerate((a, p) for a in ALGOS for p in PROTOS):
            lat, bw = float(cells[i][0]), float(cells[i][1])
            if bw > 0:
                table[(pr._op_name(m.group(1)), algo, proto)] = (lat, bw)
    return table


def model_time(table, operation, algo, proto, nbytes):
    """NCCL's estimated time [us] of one call (numPipeOps = 1, as in ncclbench --blocking); NaN if not available"""
    if (operation, algo, proto) not in table:
        return np.nan
    lat, bw = table[(operation, algo, proto)]
    log_size = int(np.floor(np.log2(nbytes >> 6))) if nbytes >> 6 > 0 else -1
    if algo == "tree" and operation == "all_reduce" and 0 <= log_size < 23:
        bw *= TREE_CORRECTION[proto][log_size]
    if (algo == "ring" and proto == "simple" and COMM_NODES > 1 and operation == "all_reduce"
            and nbytes / (COMM_CHANNELS * COMM_RANKS) >= 64):
        lat *= 1.4  # "plateau effect of ring" (1.9 below compute capability 8.0)
    return lat + nbytes / (1000 * bw)


def model_estimates(table, sizes):
    """estimates of every available combination: operation, size_bytes, algo, proto, estimate_us, model_choice"""
    rows = [{"operation": op, "size_bytes": s, "algo": a, "proto": p, "estimate_us": model_time(table, op, a, p, s)}
            for op, combos in COMBINATIONS.items() for s in sizes for a, p in combos]
    est = pd.DataFrame(rows)
    best = est.loc[est.groupby(["operation", "size_bytes"])["estimate_us"].idxmin()]
    est["model_choice"] = est.merge(best[["operation", "size_bytes", "algo", "proto"]], how="left",
                                    on=["operation", "size_bytes", "algo", "proto"], indicator=True)["_merge"].eq("both")
    return est


# ---------------------------------------------------------------------------------------------
# check of the choices
# ---------------------------------------------------------------------------------------------

def _size_ranges(table, keys):
    """collapse consecutive sizes with the same values of `keys` into ranges ("8 B - 4 KiB")"""
    rows, start, prev, prev_vals = [], None, None, None
    for size, r in table.iterrows():
        vals = tuple(r[k] for k in keys)
        if vals != prev_vals and start is not None:
            rows.append((start, prev, prev_vals))
            start = None
        if start is None:
            start = size
        prev, prev_vals = size, vals
    rows.append((start, prev, prev_vals))
    return [(pr.size_label(int(a)) if a == b else f"{pr.size_label(int(a))} - {pr.size_label(int(b))}", *v)
            for a, b, v in rows]


def _choice_per_size(calls):
    """per size: the single (algo, proto, channels, warps) of all calls, or every combination seen joined by '|'"""
    def one(col):
        return calls.groupby("size_bytes")[col].agg(lambda v: "|".join(map(str, sorted(set(v)))))
    return pd.DataFrame({c: one(c) for c in ["algo", "proto", "channels", "warps"]}).assign(
        calls=calls.groupby("size_bytes").size())


def check_choices(run, choices, estimates):
    """text report: every variant used what it pinned; NCCL's choice vs. logging run and model"""
    warmups, iterations = ncclbench_settings(run)
    text = []
    problems = 0

    text.append("Calls recorded by ncclsee per (round, variant, test, size), all ranks; expected "
                f"8 ranks x ({warmups} + {iterations}) = {8 * (warmups + iterations)}")
    counts = choices.groupby(["round", "variant", "test", "size_bytes"]).size()
    text.append(f"  min {counts.min()}, max {counts.max()} over {len(counts)} groups"
                + ("" if counts.min() == counts.max() == 8 * (warmups + iterations) else "  <-- MISMATCH"))
    problems += counts.min() != 8 * (warmups + iterations) or counts.max() != counts.min()
    rounds = choices.groupby(["variant", "test", "size_bytes", "algo", "proto", "channels", "warps"])["round"].nunique()
    text.append(f"  rounds per (variant, test, size, algo, proto, channels, warps): {sorted(set(rounds))} "
                "(3 = the same choice in every round)")
    text.append("")

    for test, operation in TESTS.items():
        for variant in sorted(choices["variant"].unique(), key=lambda v: (v != "default", v)):
            calls = choices[(choices["test"] == test) & (choices["variant"] == variant)]
            per_size = _choice_per_size(calls)
            pinned = pinned_combination(variant, operation)
            status = ""
            if pinned is not None:
                ok = (per_size["algo"] == pinned[0]) & (per_size["proto"] == pinned[1])
                status = "OK, every call used the pinned combination" if ok.all() else \
                    f"NOT PINNED at {[pr.size_label(int(s)) for s in per_size.index[~ok]]}"
                problems += not ok.all()
            what = combination_label(*pinned) + " pinned" if pinned else "NCCL's choice"
            text.append(f"{pr.OPERATION_LABELS[operation]}, {variant} ({what}){': ' + status if status else ''}")
            text.append(f"  {'size':<22}{'algo':<8}{'proto':<9}{'channels':<10}warps")
            for rng, algo, proto, ch, warps in _size_ranges(per_size, ["algo", "proto", "channels", "warps"]):
                text.append(f"  {rng:<22}{algo:<8}{proto:<9}{ch:<10}{warps}")
            text.append("")

    # NCCL's choice: default variant vs. logging run vs. model
    text.append("NCCL's choice (default variant) vs. logging run 1889653 (ncclsee records) and the cost model "
                "(minimum estimate)")
    log = logging_choices()
    for test, operation in TESTS.items():
        tun = _choice_per_size(choices[(choices["test"] == test) & (choices["variant"] == "default")])
        lg = _choice_per_size(log[log["test"] == test])
        model = estimates[(estimates["operation"] == operation) & estimates["model_choice"]].set_index("size_bytes")
        text.append(f"  {pr.OPERATION_LABELS[operation]}:")
        for size, r in tun.iterrows():
            t = f"{r['algo']}/{r['proto']}/{r['channels']}ch/{r['warps']}w"
            l = (f"{lg.loc[size, 'algo']}/{lg.loc[size, 'proto']}/{lg.loc[size, 'channels']}ch/{lg.loc[size, 'warps']}w"
                 if size in lg.index else "(not in logging run)")
            m = f"{model.loc[size, 'algo']}/{model.loc[size, 'proto']}"
            same_log = size not in lg.index or l == t
            same_model = m == f"{r['algo']}/{r['proto']}"
            problems += not (same_log and same_model)
            text.append(f"    {pr.size_label(int(size)):<10} tuning {t:<26} logging {l:<26} model {m:<14}"
                        f"{'' if same_log and same_model else '<-- DIFFERS'}")
    text.append("")
    text.append(f"{problems} problem(s)")
    return "\n".join(text), problems


# ---------------------------------------------------------------------------------------------
# measured vs. estimated
# ---------------------------------------------------------------------------------------------

def measured_combinations(bench):
    """
    benchmark times per (operation, combination): the pinned variants, and NCCL's choice as its own entry
    (algo = proto = "default"); the tree_* variants' Broadcast runs (NCCL's choice as well) are left out, so every
    entry has the same number of calls
    """
    rows = []
    for variant, g in bench.groupby("variant"):
        for operation, h in g.groupby("operation"):
            pinned = pinned_combination(variant, operation)
            if pinned is None and variant != "default":
                continue
            algo, proto = pinned or ("default", "default")
            rows.append(h.assign(algo=algo, proto=proto))
    return pd.concat(rows, ignore_index=True)


def model_comparison(measured, estimates, choices):
    """
    per operation, size, combination: estimate, measured median/mean/quartiles, percentile of the estimate in the
    measured distribution; NCCL's choice ("default") gets the estimate of the combination it actually used
    """
    g = measured.groupby(["operation", "size_bytes", "algo", "proto"])["time_us"]
    stats = pd.DataFrame({"median_us": g.median(), "mean_us": g.mean(), "q25_us": g.quantile(0.25),
                          "q75_us": g.quantile(0.75), "calls": g.size()}).reset_index()
    used = (choices[choices["variant"] == "default"].assign(operation=lambda d: d["test"].map(TESTS))
            .groupby(["operation", "size_bytes"])[["algo", "proto"]].first().reset_index())
    default_est = used.merge(estimates, on=["operation", "size_bytes", "algo", "proto"])
    default_est = default_est.rename(columns={"algo": "used_algo", "proto": "used_proto"}).assign(
        algo="default", proto="default")
    est = pd.concat([estimates.assign(used_algo=estimates["algo"], used_proto=estimates["proto"]), default_est])
    out = stats.merge(est, on=["operation", "size_bytes", "algo", "proto"], how="left")
    pct = []
    for _, r in out.iterrows():
        t = measured.loc[(measured["operation"] == r["operation"]) & (measured["size_bytes"] == r["size_bytes"])
                         & (measured["algo"] == r["algo"]) & (measured["proto"] == r["proto"]), "time_us"]
        pct.append(100 * (t < r["estimate_us"]).mean())
    out["estimate_percentile"] = pct
    out["estimate_over_median"] = out["estimate_us"] / out["median_us"]
    return out


def comparison_text(comp, run):
    text = [f"NCCL's estimate vs. measured ncclbench time per call [us], {label(run)}, all rounds together",
            "estimate: NCCL 2.27.7's cost model with the TUNING table of logging run 1889653; pct: share of the "
            "measured calls faster than the estimate (0 = the estimate is below every call); est/med: estimate / "
            "measured median; default: NCCL's choice, with the estimate of the combination it used", ""]
    for operation in TESTS.values():
        sel = comp[comp["operation"] == operation]
        combos = [("default", "default")] + COMBINATIONS[operation]
        header = f"{'size':<9}" + "".join(
            f"{'NCCL choice' if a == 'default' else combination_label(a, p):>34}" for a, p in combos)
        sub = f"{'':<9}" + "".join(f"{'estimate':>10}{'median':>10}{'pct':>6}{'est/med':>8}" for _ in combos)
        text += [pr.OPERATION_LABELS[operation], header, sub]
        for size, g in sel.groupby("size_bytes"):
            g = g.set_index(["algo", "proto"])
            line = f"{pr.size_label(int(size)):<9}"
            for c in combos:
                r = g.loc[c]
                line += f"{r['estimate_us']:>10.1f}{r['median_us']:>10.1f}{r['estimate_percentile']:>6.0f}" \
                        f"{r['estimate_over_median']:>8.2f}"
            text.append(line)
        text.append("")
    return "\n".join(text)


def _size_axis(ax, sizes, label=True):
    """profiler_results._size_axis with ticks only at powers of two (not at the extra sizes 1.5 and 3 MiB)"""
    pr._size_axis(ax, [s for s in sizes if int(s) & (int(s) - 1) == 0], label=label)


def _combo_style(algo, proto):
    if algo == "default":
        return {"color": DEFAULT_COLOR, "linestyle": "-", "marker": "D"}
    return {"color": PROTO_COLORS[proto], "linestyle": ALGO_LINESTYLES[algo], "marker": ALGO_MARKERS[algo]}


def plot_model_overview(measured, comp):
    """
    one panel per (operation, combination), over all sizes: box = measured calls / measured median (quartiles,
    whiskers 5 - 95 %), point = NCCL's estimate / measured median; log y
    """
    panels = [(op, a, p) for op in TESTS.values() for a, p in COMBINATIONS[op]]
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True, sharey=True, layout="constrained")
        for ax, (op, algo, proto) in zip(axs.flat, panels):
            sel = measured[(measured["operation"] == op) & (measured["algo"] == algo) & (measured["proto"] == proto)]
            c = comp[(comp["operation"] == op) & (comp["algo"] == algo) & (comp["proto"] == proto)].set_index("size_bytes")
            sizes = sorted(sel["size_bytes"].unique())
            pos = np.log2(sizes)
            data = [sel.loc[sel["size_bytes"] == s, "time_us"] / c.loc[s, "median_us"] for s in sizes]
            color = PROTO_COLORS[proto]
            ax.boxplot(data, positions=pos, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                       boxprops={"facecolor": color, "alpha": 0.6, "edgecolor": pr.EDGE_COLOR, "linewidth": 0.8},
                       medianprops={"color": "black", "linewidth": 1}, whiskerprops={"color": pr.EDGE_COLOR},
                       capprops={"color": pr.EDGE_COLOR}, manage_ticks=False, zorder=2)
            ax.scatter(pos, c.loc[sizes, "estimate_over_median"], marker=ALGO_MARKERS[algo], s=22, color=color,
                       edgecolor="black", linewidth=0.6, zorder=4)
            # sizes where NCCL itself chose this combination
            chosen = c.loc[sizes, "model_choice"].astype(bool).to_numpy()
            ax.scatter(pos[chosen], np.full(chosen.sum(), 0.28), marker="^", s=14, color="black", zorder=4)
            ax.axhline(1, color="gray", linestyle="-.", linewidth=0.8, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(0.25, 4.2)
            ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
            ax.minorticks_off()
            ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
            ticks = [t for t in pos if t == int(t)][::3]
            ax.set_xticks(ticks, [pr.size_label(int(2**t)) for t in ticks], rotation=45, ha="right")
            ax.set_title(f"{pr.OPERATION_LABELS[op]}, {combination_label(algo, proto)}")
        for ax in axs[-1]:
            ax.set_xlabel("Message size")
        for ax in axs[:, 0]:
            ax.set_ylabel("Time / measured median")
        handles = [Patch(facecolor="gray", alpha=0.6, edgecolor=pr.EDGE_COLOR,
                         label="measured (box: quartiles, whiskers: 5 - 95 %)"),
                   Line2D([], [], color="gray", marker="o", markeredgecolor="black", linestyle="none",
                          label="NCCL's estimate"),
                   Line2D([], [], color="black", marker="^", linestyle="none", label="NCCL chooses this combination")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return fig


def plot_model_violins(measured, comp):
    """selected sizes, absolute times: violin of the measured calls per combination, NCCL's estimate as a point"""
    ops = list(TESTS.values())
    ncols = max(len(VIOLIN_SIZES[op]) for op in ops)
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(len(ops), ncols, figsize=(3.1 * ncols, 3.3 * len(ops)), layout="constrained",
                                squeeze=False)
        for row, op in zip(axs, ops):
            combos = COMBINATIONS[op]
            for ax, size in zip(row, VIOLIN_SIZES[op]):
                c = comp[(comp["operation"] == op) & (comp["size_bytes"] == size)].set_index(["algo", "proto"])
                for x, (algo, proto) in enumerate(combos):
                    t = measured.loc[(measured["operation"] == op) & (measured["size_bytes"] == size)
                                     & (measured["algo"] == algo) & (measured["proto"] == proto), "time_us"]
                    body = ax.violinplot([t], positions=[x], widths=0.8, showextrema=False)["bodies"][0]
                    body.set_facecolor(PROTO_COLORS[proto])
                    body.set_edgecolor(pr.EDGE_COLOR)
                    body.set_alpha(0.55)
                    ax.plot([x - 0.25, x + 0.25], [t.median()] * 2, color="black", linewidth=1.5, zorder=3)
                    ax.scatter([x], [c.loc[(algo, proto), "estimate_us"]], marker=ALGO_MARKERS[algo], s=30,
                               color=PROTO_COLORS[proto], edgecolor="black", linewidth=0.8, zorder=4)
                    if c.loc[(algo, proto), "model_choice"]:
                        ax.annotate("*", (x, 1), xycoords=("data", "axes fraction"), ha="center", va="top",
                                    fontsize=14)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                    lambda v, _: f"{v:g}" if f"{v:.1g}"[0] in "25" else ""))
                ax.tick_params(axis="y", which="minor", labelsize=8)
                ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
                ax.set_xticks(range(len(combos)), [combination_label(a, p).replace("/", "\n") for a, p in combos],
                              fontsize=8)
                ax.set_title(f"{pr.OPERATION_LABELS[op]}, {pr.size_label(int(size))}")
            row[0].set_ylabel("Time [µs]")
            for ax in row[len(VIOLIN_SIZES[op]):]:
                ax.axis("off")
        handles = [Patch(facecolor="gray", alpha=0.55, edgecolor=pr.EDGE_COLOR, label="measured calls (3 rounds)"),
                   Line2D([], [], color="black", linewidth=1.5, label="measured median"),
                   Line2D([], [], color="gray", marker="o", markeredgecolor="black", linestyle="none",
                          label="NCCL's estimate"),
                   Line2D([], [], color="black", marker="$*$", markersize=9, linestyle="none",
                          label="NCCL's choice (minimum estimate)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=4)
    return fig


# ---------------------------------------------------------------------------------------------
# rounds, pinned vs. default
# ---------------------------------------------------------------------------------------------

def variant_order(variants):
    order = ["default"] + [variant_name(a, p) for a in ALGOS for p in PROTOS]
    return [v for v in order if v in set(variants)] + sorted(set(variants) - set(order))


def round_deviation(bench):
    """per (variant, operation, size) and round: median as deviation from the median over the rounds [%], and IQR [%]"""
    df = bench.rename(columns={"time_us": "time_s", "variant": "profiler", "round": "run"}).assign(
        source="benchmark", benchmark="ncclbench", test="", metric="time_iter", rank=np.nan, host=None)
    return pr.iteration_deviation(df).rename(columns={"profiler": "variant", "run": "round"})


def plot_round_consistency(dev):
    """as profiler_results.plot_iteration_consistency, with rounds instead of iterations and one row per variant"""
    variants = variant_order(dev["variant"].unique())
    ops = list(TESTS.values())
    rounds = sorted(dev["round"].unique())
    band = dev.groupby(["operation", "variant", "size_bytes"])["iqr_pct"].mean().reset_index()
    limit = max(dev["deviation_pct"].abs().max(), (band["iqr_pct"] / 2).max()) * 1.1
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(len(variants), len(ops), figsize=(4.5 * len(ops), 1.45 * len(variants) + 0.8),
                                sharex="col", sharey=True, layout="constrained", squeeze=False)
        for r, variant in enumerate(variants):
            for c, op in enumerate(ops):
                ax = axs[r][c]
                b = band[(band["operation"] == op) & (band["variant"] == variant)].sort_values("size_bytes")
                ax.fill_between(b["size_bytes"], -b["iqr_pct"] / 2, b["iqr_pct"] / 2, color="#d9d9d9", linewidth=0,
                                zorder=1)
                ax.axhline(0, color="gray", linewidth=0.8, zorder=2)
                for i, rnd in enumerate(rounds):
                    sel = dev[(dev["operation"] == op) & (dev["variant"] == variant)
                              & (dev["round"] == rnd)].sort_values("size_bytes")
                    ax.plot(sel["size_bytes"], sel["deviation_pct"], color=ROUND_COLORS[i % len(ROUND_COLORS)],
                            marker=ROUND_MARKERS[i % len(ROUND_MARKERS)], markersize=3, markeredgecolor="white",
                            markeredgewidth=0.5, linewidth=1.1, zorder=3)
                ax.set_ylim(-limit, limit)
                ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
                _size_axis(ax, b["size_bytes"], label=(r == len(variants) - 1))
                if r == 0:
                    ax.set_title(pr.OPERATION_LABELS[op])
                pinned = pinned_combination(variant, op)
                name = "default" if variant == "default" else (
                    combination_label(*pinned) if pinned else f"NCCL's choice (run of {variant})")
                ax.text(0.01, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
        handles = [Line2D([], [], color=ROUND_COLORS[i % len(ROUND_COLORS)], marker=ROUND_MARKERS[i % len(ROUND_MARKERS)],
                          markeredgecolor="white", linewidth=1.2, label=f"round {rnd}") for i, rnd in enumerate(rounds)]
        handles.append(Patch(facecolor="#d9d9d9", label="spread within a round (IQR)"))
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        fig.supylabel("Deviation of the round median from the median over rounds [%]", fontsize=10)
    return fig


def pinned_vs_default(measured, comp):
    """
    per operation, size and pinned combination: median over all rounds, ratio to NCCL's choice (default), the range
    of the per-round ratio (round median pinned / round median default of the same round), the number of rounds in
    which it was faster (rounds_faster), and whether it is the combination NCCL chose itself at that size (same)
    """
    med = measured.groupby(["operation", "size_bytes", "algo", "proto"])["time_us"].median().rename("median_us")
    per_round = measured.groupby(["operation", "size_bytes", "algo", "proto", "round"])["time_us"].median()
    default = med.xs(("default", "default"), level=["algo", "proto"])
    default_round = per_round.xs(("default", "default"), level=["algo", "proto"])
    out = med.reset_index()
    out["ratio"] = out["median_us"] / out.set_index(["operation", "size_bytes"]).index.map(default)
    rr = (per_round / default_round.reindex(per_round.index.droplevel(["algo", "proto"])).to_numpy()).rename("r")
    rr = rr.groupby(["operation", "size_bytes", "algo", "proto"]).agg(
        ratio_min="min", ratio_max="max", rounds_faster=lambda r: int((r < 1).sum()))
    out = out.merge(rr.reset_index(), on=["operation", "size_bytes", "algo", "proto"])
    used = comp.loc[comp["algo"] == "default", ["operation", "size_bytes", "used_algo", "used_proto"]]
    out = out.merge(used, on=["operation", "size_bytes"])
    out["same"] = (out["algo"] == out["used_algo"]) & (out["proto"] == out["used_proto"])
    return out.drop(columns=["used_algo", "used_proto"])


def noise_floor(bench, pvd):
    """
    per (operation, size_bytes): largest deviation from 1 of the per-round ratio between two runs of the same
    configuration (separate processes, same round): the pinned combination NCCL chose itself vs. default, and for
    Broadcast also the tree_* variants (NCCL's choice as well) vs. default. Differences below it are not
    attributable to the combination
    """
    same = pvd[pvd["same"]].set_index(["operation", "size_bytes"])
    noise = np.maximum(same["ratio_max"] - 1, 1 - same["ratio_min"])
    medians = bench.groupby(["operation", "variant", "round", "size_bytes"])["time_us"].median()
    for operation in TESTS.values():
        default = medians.loc[(operation, "default")]
        for variant in bench["variant"].unique():
            if variant == "default" or pinned_combination(variant, operation) is not None:
                continue
            dev = (medians.loc[(operation, variant)] / default - 1).abs().groupby("size_bytes").max()
            idx = pd.MultiIndex.from_product([[operation], dev.index], names=["operation", "size_bytes"])
            noise = np.maximum(noise, pd.Series(dev.to_numpy(), index=idx).reindex(noise.index).fillna(0))
    return noise.rename("noise")


def clearly_faster(pvd, noise):
    """pinned combinations faster than NCCL's choice in every round, and by more than the noise floor of the size"""
    n = pvd.set_index(["operation", "size_bytes"]).index.map(noise).to_numpy()
    return (pvd["algo"] != "default") & ~pvd["same"] & (pvd["ratio_max"] < 1) & (pvd["ratio"] < 1 - n)


def plot_pinned_vs_default(pvd, noise):
    """
    like latency_ratio.pdf: top median latency per combination (NCCL's choice in black), bottom median pinned /
    median default with the range of the per-round ratios as band; below 1 = the pinned combination is faster.
    Circled: faster in every round and by more than the noise floor (see noise_floor); gray band: the noise floor
    """
    ops = list(TESTS.values())
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(2, len(ops), figsize=(5.5 * len(ops), 6.4), sharex="col", layout="constrained",
                                squeeze=False, height_ratios=[1, 1.15])
        for col, op in enumerate(ops):
            top, bottom = axs[0][col], axs[1][col]
            for algo, proto in [("default", "default")] + COMBINATIONS[op]:
                sel = pvd[(pvd["operation"] == op) & (pvd["algo"] == algo) & (pvd["proto"] == proto)].sort_values(
                    "size_bytes")
                st = _combo_style(algo, proto)
                is_default = algo == "default"
                kw = {"color": st["color"], "linestyle": st["linestyle"], "linewidth": 2.2 if is_default else 1.4,
                      "zorder": 5 if is_default else 3}
                mk = {"marker": st["marker"], "markersize": 3.5, "markeredgecolor": "white", "markeredgewidth": 0.5}
                top.plot(sel["size_bytes"], sel["median_us"], **kw, **mk)
                if is_default:
                    bottom.axhline(1, **kw)
                    continue
                bottom.fill_between(sel["size_bytes"], sel["ratio_min"], sel["ratio_max"], color=st["color"],
                                    alpha=0.15, linewidth=0, zorder=2)
                bottom.plot(sel["size_bytes"], sel["ratio"], **kw, **mk)
                faster = sel[clearly_faster(sel, noise)]
                bottom.scatter(faster["size_bytes"], faster["ratio"], s=60, facecolor="none",
                               edgecolor=st["color"], linewidth=1.3, zorder=6)
            sizes = pvd.loc[pvd["operation"] == op, "size_bytes"].unique()
            top.set_yscale("log")
            top.set_title(pr.OPERATION_LABELS[op])
            _size_axis(top, sizes, label=False)
            _size_axis(bottom, sizes)
            # clipped: the interesting part is around 1 (the top panel shows how much slower the others are)
            bottom.set_yscale("log")
            bottom.set_ylim(0.8, 1.5)
            bottom.set_yticks([0.8, 0.9, 1, 1.1, 1.25, 1.5], ["0.8", "0.9", "1", "1.1", "1.25", "1.5"])
            bottom.minorticks_off()
            n = noise.loc[op].sort_index()
            bottom.fill_between(n.index, (1 - n).clip(lower=0.8), 1 + n, color="#e0e0e0", linewidth=0, zorder=1)
            for ax in (top, bottom):
                ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
        axs[0][0].set_ylabel("Median latency [µs]")
        axs[1][0].set_ylabel("Pinned / NCCL's choice (clipped)")
        handles = [Line2D([], [], color=DEFAULT_COLOR, linewidth=2.2, marker="D", markersize=4,
                          label="NCCL's choice (default)")]
        handles += [Line2D([], [], color=PROTO_COLORS[p], linestyle=ALGO_LINESTYLES[a], marker=ALGO_MARKERS[a],
                           markersize=4, markeredgecolor="white", linewidth=1.4, label=combination_label(a, p))
                    for a in ALGOS for p in PROTOS]
        handles += [Patch(facecolor="gray", alpha=0.25, label="range of the per-round ratios"),
                    Patch(facecolor="#e0e0e0", label="noise floor (same configuration, separate runs)"),
                    Line2D([], [], color="gray", marker="o", markerfacecolor="none", markersize=8, linestyle="none",
                           label="faster in every round, beyond the noise")]
        fig.legend(handles=handles, loc="outside lower center", ncol=4)
    return fig


def best_text(pvd, comp, noise, run):
    """per size: NCCL's choice and the fastest measured combination (median over all rounds)"""
    text = [f"NCCL's choice vs. the fastest combination (median over all rounds, us), {label(run)}",
            "ratio: fastest / NCCL's choice; rounds: in how many rounds it was faster (round medians); model rank: "
            "rank of the fastest by NCCL's estimate (1 = NCCL's choice); = : the fastest is NCCL's choice itself",
            "noise: largest deviation between two runs of the same configuration in the same round (NCCL's own "
            "combination pinned vs. default; Broadcast also tree_* vs. default); * = faster in all rounds and by more "
            "than the noise", ""]
    fast = pvd[clearly_faster(pvd, noise)]
    for op in TESTS.values():
        text.append(pr.OPERATION_LABELS[op])
        text.append(f"  {'size':<10}{'NCCL choice':<13}{'median':>9}  {'fastest':<14}{'median':>9}{'ratio':>7}"
                    f"{'rounds':>8}{'noise':>8}{'model rank':>12}")
        for size, g in pvd[(pvd["operation"] == op) & (pvd["algo"] != "default")].groupby("size_bytes"):
            c = comp[(comp["operation"] == op) & (comp["size_bytes"] == size)]
            d = c[c["algo"] == "default"].iloc[0]
            best = g.loc[g["median_us"].idxmin()]
            ranked = c[c["algo"] != "default"].sort_values("estimate_us")
            rank = [(a, p) for a, p in zip(ranked["algo"], ranked["proto"])].index((best["algo"], best["proto"])) + 1
            mark = "=" if best["same"] else ("*" if best.name in fast.index else "")
            text.append(f"  {pr.size_label(int(size)):<10}{combination_label(d['used_algo'], d['used_proto']):<13}"
                        f"{d['median_us']:>9.1f}  {combination_label(best['algo'], best['proto']) + ' ' + mark:<14}"
                        f"{best['median_us']:>9.1f}{best['ratio']:>7.3f}{best['rounds_faster']:>6}/3"
                        f"{100 * noise.loc[(op, size)]:>7.1f}%{rank:>12}")
        text.append("")
    return "\n".join(text)


# ---------------------------------------------------------------------------------------------

def analyze(run):
    run = Path(run)
    print(f"{run}: tuning run, {len(variant_dirs(run))} variant runs (rounds x variants)")
    df = load_times(run)
    bench = benchmark_times(df)
    choices = load_choices(run)
    estimates = model_estimates(model_table(), sorted(bench["size_bytes"].unique()))

    text, problems = check_choices(run, choices, estimates)
    (run / "choices.txt").write_text(text + "\n")
    print(text)

    measured = measured_combinations(bench)
    comp = model_comparison(measured, estimates, choices)
    text = comparison_text(comp, run)
    (run / "model_vs_measured.txt").write_text(text + "\n")
    print(text)

    for name, fig, title in [
            ("model_vs_measured.pdf", plot_model_overview(measured, comp), "NCCL's estimate vs. measured time"),
            ("model_violins.pdf", plot_model_violins(measured, comp), "NCCL's estimate vs. measured time"),
            ("round_consistency.pdf", plot_round_consistency(round_deviation(bench)), "consistency of the rounds"),
    ]:
        with plt.rc_context(pr.PLOT_STYLE):
            fig.suptitle(f"{label(run)}: {title} (ncclbench)")
        fig.savefig(run / name, bbox_inches="tight")
        plt.close(fig)

    pvd = pinned_vs_default(measured, comp)
    noise = noise_floor(bench, pvd)
    fig = plot_pinned_vs_default(pvd, noise)
    with plt.rc_context(pr.PLOT_STYLE):
        fig.suptitle(f"{label(run)}: pinned algorithm/protocol vs. NCCL's choice (ncclbench, 3 rounds)")
    fig.savefig(run / "pinned_vs_default.pdf", bbox_inches="tight")
    plt.close(fig)
    text = best_text(pvd, comp, noise, run)
    (run / "best_vs_default.txt").write_text(text + "\n")
    print(text)


def is_tuning_run(run):
    """a run with variants: <run>/<round>/<variant>/profiler-out.tar.gz"""
    return any(Path(run).glob("[0-9]*/*/profiler-out.tar.gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="run folder (default: newest in results/tuning/)")
    args = parser.parse_args()
    run = args.run or max((p for p in (RESULTS / "tuning").glob("*") if p.name.isdigit()), key=lambda p: int(p.name))
    analyze(run)


if __name__ == "__main__":
    main()
