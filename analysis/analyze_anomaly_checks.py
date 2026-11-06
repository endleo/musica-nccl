"""Anomaly checks (experiments/ll128_chunks.sbatch, experiments/ll_threads.sbatch): experiments that test an explanation
of an anomaly of the tuning runs, Broadcast with a few variants vs. NCCL's choice.

    uv run analysis/analyze_anomaly_checks.py [results/<ll128_chunks|ll_threads>/<job>]

Default: the newest run of either experiment. Layout as in the tuning experiment (<run>/<round>/<variant>/, only
broadcast_bench), see analyze_tuning.py. Unlike there, the variants are not one algorithm/protocol each but settings
of the experiment (buffer size, channels, threads), listed in VARIANTS per experiment.

Writes into the run folder:
    pinned_vs_default.pdf     as in analyze_tuning.py: median latency of every variant, and variant / default per size
and prints what ncclsee recorded per variant and size (algorithm, protocol, channels, warps), to check the pinning.
Variants in HIDDEN are not drawn (but used for the noise floor). Where NCCL's choice uses another channel count than
the variants pin (PINNED_CHANNELS), the plot is shaded: there the ratio also compares channel counts.

Unlike analyze_tuning.py, the plot shows neither the range of the per-round ratios nor the noise floor; the noise
floor only decides the circles. Noise floor: as in analyze_tuning.py the largest per-round deviation between two runs
of the same configuration.
Here those are the variants that only pin what NCCL could choose itself (PIN_ONLY), at the sizes where ncclsee
recorded the same algorithm, protocol, channels and warps as for default. Sizes without such a pair have no noise
floor (0); the circles then only mean "faster in every round".

Statistics as in analyze_tuning.py: ncclbench's per-call times (max over ranks), warm-ups dropped; medians over the
iterations of all rounds together.
"""

from pathlib import Path

from common import choose_run, label, run_parser  # before pyplot: sets matplotlib's backend

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyze_tuning as at
import profiler_results as pr

EXPERIMENTS = ["ll128_chunks", "ll_threads"]
TEST, OPERATION = "broadcast_bench", "broadcast"
# per experiment: variant -> (legend label, style); styles as in analyze_tuning.py (protocol = color, Ring = dashed);
# variants with the same algorithm and protocol get further colors of their own (Dark2, as PROTO_COLORS)
VARIANTS = {
    "ll128_chunks": {
        "ring_ll128_halfbuf": ("Ring/LL128, 1/2 LL128 buffer (NCCL_LL128_BUFFSIZE=2457600)",
                               {"color": at.PROTO_COLORS["ll128"], "linestyle": "--", "marker": "s"}),
        "ring_ll128_quarterbuf": ("Ring/LL128, 1/4 LL128 buffer (NCCL_LL128_BUFFSIZE=1228800)",
                                  {"color": "#e6ab02", "linestyle": "--", "marker": "^"}),
        "ring_ll": ("Ring/LL", {"color": at.PROTO_COLORS["ll"], "linestyle": "--", "marker": "s"}),
    },
    "ll_threads": {
        "ll_1ch": ("Ring/LL, 1 channel, NCCL's thread count",
                   {"color": at.PROTO_COLORS["ll"], "linestyle": "--", "marker": "s"}),
        "ll_1ch_512t": ("Ring/LL, always 512 threads (1 channel)",
                        {"color": "#e7298a", "linestyle": "--", "marker": "^"}),
        "ll_1ch_128t": ("Ring/LL, always 128 threads (1 channel)",
                        {"color": "#a6761d", "linestyle": "--", "marker": "v"}),
    },
}
# variants that only pin choices NCCL makes itself at some sizes (no other setting): the noise floor's reference
PIN_ONLY = {"ll_1ch", "ring_ll"}
# variants not drawn: ll_1ch only checks that the channel pin changes nothing else (it matches default where default
# has 1 channel, and gives the noise floor there)
HIDDEN = {"ll_1ch"}
# per experiment: the channel count its variants pin. Where default uses a different one, the ratio compares channel
# counts as well, not only the setting under test: shaded in the plot (the thread threshold also sets the channels,
# src/enqueue.cc, so ll_threads has to pin them)
PINNED_CHANNELS = {"ll_threads": 1}
DEFAULT_STYLE = {"color": at.DEFAULT_COLOR, "linestyle": "-", "marker": "D"}
CHOICE_KEYS = ["algo", "proto", "channels", "warps"]


def check_variants(run, experiment, info):
    """abort if the run's variants are not default + the experiment's VARIANTS"""
    expected = {"default", *VARIANTS[experiment]}
    if set(info["variants"]) != expected:
        raise SystemExit(f"{run}: variants {sorted(info['variants'])}, expected {sorted(expected)} "
                         f"(experiments/{experiment}.sbatch; add new variants to VARIANTS in this script)")


def load_bench(run):
    """ncclbench's per-call times of all rounds and variants: round, variant, size_bytes, time_us"""
    warmups, iterations = at.ncclbench_settings(run)
    frames = []
    for rnd, variant, d in at.variant_dirs(run):
        df = pr.load_run(d, profilers=["ncclsee"], nccl_tests=[], nccl_bench_tests=[TEST],
                         warmups=warmups, iterations=iterations)
        frames.append(df.assign(run=rnd, round=rnd, variant=variant))
    bench = at.benchmark_times(pd.concat(frames, ignore_index=True))
    return bench[bench["operation"] == OPERATION]


def load_choices(run):
    """ncclsee's records per round and variant (all ranks, all calls)"""
    frames = []
    for rnd, variant, d in at.variant_dirs(run):
        root = pr.extract_profiler_outputs(d) / "profiler-out" / "ncclsee"
        frames.append(at.ncclsee_calls(root / TEST).assign(round=rnd, variant=variant))
    return pd.concat(frames, ignore_index=True)


def choices_text(choices, variants):
    """per variant: algorithm/protocol/channels/warps per size range (all rounds and ranks together)"""
    text = []
    for variant in variants:
        per_size = at._choice_per_size(choices[choices["variant"] == variant])
        text += [f"{variant}", f"  {'size':<22}{'algo':<8}{'proto':<9}{'channels':<10}warps"]
        text += [f"  {r:<22}{a:<8}{p:<9}{c:<10}{w}" for r, a, p, c, w in at._size_ranges(per_size, CHOICE_KEYS)]
        text.append("")
    return "\n".join(text)


def same_as_default(choices):
    """per (variant, size_bytes): whether ncclsee recorded the same single algo/proto/channels/warps as for default"""
    per = {v: at._choice_per_size(g)[CHOICE_KEYS] for v, g in choices.groupby("variant")}
    default = per["default"]
    rows = []
    for variant, p in per.items():
        if variant == "default":
            continue
        common = p.index.intersection(default.index)
        same = (p.loc[common] == default.loc[common]).all(axis=1) & ~p.loc[common].apply(
            lambda r: any("|" in str(v) for v in r), axis=1)
        rows.append(pd.DataFrame({"variant": variant, "size_bytes": common, "same": same.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def pinned_vs_default(bench, same):
    """
    per variant and size: median over all rounds, ratio to default, range of the per-round ratios (round median
    variant / round median default of the same round), rounds in which it was faster, and same (see same_as_default)
    """
    med = bench.groupby(["variant", "size_bytes"])["time_us"].median().rename("median_us")
    per_round = bench.groupby(["variant", "size_bytes", "round"])["time_us"].median()
    default, default_round = med.loc["default"], per_round.loc["default"]
    out = med.reset_index()
    out["ratio"] = out["median_us"] / out["size_bytes"].map(default).to_numpy()
    rr = (per_round / default_round.reindex(per_round.index.droplevel("variant")).to_numpy()).rename("r")
    rr = rr.groupby(["variant", "size_bytes"]).agg(ratio_min="min", ratio_max="max",
                                                   rounds_faster=lambda r: int((r < 1).sum()))
    out = out.merge(rr.reset_index(), on=["variant", "size_bytes"])
    out = out.merge(same, on=["variant", "size_bytes"], how="left")
    # only a PIN_ONLY variant can be the same configuration: the others change settings ncclsee does not record
    out["same"] = out["same"].fillna(False).astype(bool) & out["variant"].isin(PIN_ONLY)
    return out


def noise_floor(pvd):
    """per size: largest per-round deviation from 1 of a PIN_ONLY variant with the same configuration as default"""
    sel = pvd[pvd["same"] & pvd["variant"].isin(PIN_ONLY)]
    dev = np.maximum(sel["ratio_max"] - 1, 1 - sel["ratio_min"]).groupby(sel["size_bytes"]).max()
    return dev.reindex(sorted(pvd["size_bytes"].unique())).rename("noise")


def clearly_faster(pvd, noise):
    """faster than default in every round, and by more than the noise floor of the size (0 where there is none)"""
    n = pvd["size_bytes"].map(noise.fillna(0)).to_numpy()
    return (pvd["variant"] != "default") & ~pvd["same"] & (pvd["ratio_max"] < 1) & (pvd["ratio"] < 1 - n)


def _ratio_axis(ax, pvd):
    """log axis as in analyze_tuning.py (0.8 - 1.5), widened to the data: here the effects are larger than that"""
    sel = pvd[pvd["variant"] != "default"]
    lo, hi = min(0.8, sel["ratio_min"].min() / 1.05), max(1.5, sel["ratio_max"].max() * 1.05)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in [0.25, 0.33, 0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8] if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    return lo


def _latency_ticks(ax, times):
    """labelled ticks on the log latency axis also when it spans less than a decade (1, 2, 5 x 10^n)"""
    lo, hi = times.min(), times.max()
    ticks = [m * 10**e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1) for m in (1, 2, 5)]
    inner = [t for t in ticks if lo <= t <= hi]
    if len(inner) >= 3:
        return  # matplotlib's decade ticks are enough
    ax.set_yticks([t for t in ticks if lo / 1.5 <= t <= hi * 1.5], minor=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(lo / 1.1, hi * 1.1)


def _size_axis(ax, sizes, label=True):
    """log2 size axis with a tick at every power of two (analyze_tuning: every third), so that the in-between sizes
    (1.5x, 256 KiB steps) are easier to place"""
    ax.set_xscale("log", base=2)
    ticks = sorted(int(t) for t in sizes if int(t) & (int(t) - 1) == 0)
    ax.set_xticks(ticks, [pr.size_label(t) for t in ticks] if label else [], rotation=45, ha="right")
    ax.minorticks_off()
    if label:
        ax.set_xlabel("Message size")


def other_channels(choices, channels):
    """sizes at which default used another channel count than `channels` (None: no pin, no sizes)"""
    if channels is None:
        return []
    per_size = at._choice_per_size(choices[choices["variant"] == "default"])["channels"]
    return sorted(s for s, c in per_size.items() if c != str(channels))


def plot_pinned_vs_default(pvd, noise, variants, shaded=(), pinned_channels=None):
    """
    as analyze_tuning.plot_pinned_vs_default, for Broadcast and the experiment's variants (without HIDDEN); `shaded`:
    sizes at which default uses another channel count than the variants (`pinned_channels`), grayed out
    """
    variants = {v: x for v, x in variants.items() if v not in HIDDEN}
    with plt.rc_context(pr.PLOT_STYLE):
        fig, axs = plt.subplots(2, 1, figsize=(5.5, 6.4), sharex="col", layout="constrained", squeeze=False,
                                height_ratios=[1, 1.15])
        top, bottom = axs[0][0], axs[1][0]
        _ratio_axis(bottom, pvd)
        for variant in ["default", *variants]:
            sel = pvd[pvd["variant"] == variant].sort_values("size_bytes")
            st = DEFAULT_STYLE if variant == "default" else variants[variant][1]
            is_default = variant == "default"
            kw = {"color": st["color"], "linestyle": st["linestyle"], "linewidth": 2.2 if is_default else 1.4,
                  "zorder": 5 if is_default else 3}
            mk = {"marker": st["marker"], "markersize": 3.5, "markeredgecolor": "white", "markeredgewidth": 0.5}
            top.plot(sel["size_bytes"], sel["median_us"], **kw, **mk)
            if is_default:
                bottom.axhline(1, **kw)
                continue
            bottom.plot(sel["size_bytes"], sel["ratio"], **kw, **mk)
            faster = sel[clearly_faster(sel, noise)]
            bottom.scatter(faster["size_bytes"], faster["ratio"], s=60, facecolor="none", edgecolor=st["color"],
                           linewidth=1.3, zorder=6)
        sizes = pvd["size_bytes"].unique()
        top.set_yscale("log")
        _latency_ticks(top, pvd["median_us"])
        top.set_title(pr.OPERATION_LABELS[OPERATION])
        _size_axis(top, sizes, label=False)
        _size_axis(bottom, sizes)
        if len(shaded):
            # from halfway (log scale) between the last size with the same channel count and the first without
            # from the first size with a different channel count: NCCL switches exactly there (nBytes >=
            # nc x nThreads x threshold, src/enqueue.cc), not somewhere between the measured sizes
            for ax in (top, bottom):
                ax.axvspan(shaded[0], max(sizes) * 1.3, color="#f0f0f0", zorder=0, linewidth=0)
                ax.set_xlim(right=max(sizes) * 1.3)
        n = noise.dropna()  # not drawn (too many gray areas), only used for the circles
        for ax in (top, bottom):
            ax.yaxis.grid(True, linestyle=":", color="lightgray", zorder=0)
        top.set_ylabel("Median latency [µs]", fontsize=9)
        bottom.set_ylabel("Variant / default", fontsize=9)
        handles = [Line2D([], [], color=DEFAULT_STYLE["color"], linewidth=2.2, marker="D", markersize=4,
                          label="NCCL's choice (default)")]
        handles += [Line2D([], [], color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=4,
                           markeredgecolor="white", linewidth=1.4, label=name) for name, st in variants.values()]
        if len(shaded):
            handles.append(Patch(facecolor="#f0f0f0", edgecolor="#c0c0c0", linewidth=0.5,
                                 label=f"default uses more than {pinned_channels} channel"
                                       f"{'s' if pinned_channels != 1 else ''} (channel counts differ)"))
        handles.append(Line2D([], [], color="gray", marker="o", markerfacecolor="none", markersize=8,
                              linestyle="none", label="faster in every round" + (", beyond the noise" if len(n) else "")))
        fig.legend(handles=handles, loc="outside lower center", ncol=1)
    return fig


def analyze(run, experiment):
    run = Path(run)
    info = at.job_info(run / "job-info.txt")
    check_variants(run, experiment, info)
    variants = VARIANTS[experiment]
    print(f"{run}: {experiment} run, {len(at.variant_dirs(run))} variant runs (rounds x variants)")
    choices = load_choices(run)
    print(choices_text(choices, ["default", *variants]))
    bench = load_bench(run)
    rounds = bench["round"].nunique()
    pvd = pinned_vs_default(bench, same_as_default(choices))
    noise = noise_floor(pvd)
    pinned_channels = PINNED_CHANNELS.get(experiment)
    fig = plot_pinned_vs_default(pvd, noise, variants, other_channels(choices, pinned_channels), pinned_channels)
    with plt.rc_context(pr.PLOT_STYLE):
        fig.suptitle(f"{label(run)}: variants vs. NCCL's choice\n(ncclbench, {rounds} rounds, "
                     f"{len(info['gpus'])} nodes)")
    fig.savefig(run / "pinned_vs_default.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {run / 'pinned_vs_default.pdf'}")


def main():
    args = run_parser(__doc__, "ll128_chunks or ll_threads").parse_args()
    if args.run is None:  # newest run of either experiment
        runs = [r for e in EXPERIMENTS for r in (at.RESULTS / e).glob("*") if r.is_dir()]
        if not runs:
            raise SystemExit(f"no runs in {', '.join(str(at.RESULTS / e) for e in EXPERIMENTS)}")
        args.run = pr.newest_run(runs)
    run = choose_run(args.run, EXPERIMENTS[0], accepted=EXPERIMENTS)
    from common import run_experiment
    analyze(run, run_experiment(run))


if __name__ == "__main__":
    main()
