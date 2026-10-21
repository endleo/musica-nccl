// Short summary of a run, to send around. Compile from the repository root (figures are read from results/):
//   typst compile --root . report/summary.typ
//   typst compile --root . --input run=results/full/<job> report/summary.typ     other run
//   typst watch --root . report/summary.typ                                      recompile on changes
// The figures come from analysis/analyze.py; the text is written for run 1863257.

#let run = sys.inputs.at("run", default: "results/full/1863257")
#let fig(name) = "/" + run + "/" + name

#set document(title: "NCCL profilers on MUSICA: first results")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#show raw: set text(size: 0.95em)

= NCCL profilers on MUSICA: first results

#text(fill: luma(90))[Run #raw(run), #datetime.today().display()]

== Setup

The Chameleon experiment, repeated on the MUSICA cluster: every benchmark is run once per NCCL profiler plugin, and
the latencies the profilers report are compared with the times the benchmarks measure.

- 2 nodes with 4 NVIDIA H100 each (partition `zen4_0768_h100x4`), one rank per GPU, 8 ranks in total;
  3 iterations, which all ran on the same two nodes
- NCCL 2.27.7, CUDA 12.9.1 and OpenMPI from the cluster's (EESSI) modules
- benchmarks: nccl-tests v2.19.6 and lib-ncclbench (`3b9d2c4`, blocking, 5 warm-ups + 50 iterations), AllReduce and
  Broadcast, 8 B -- 1 GiB
- profilers: CoMMA (`1d5b967`) and `ncclsee` (`0e8433a`, tracer mode). No NCCL Inspector: it implements only profiler
  API v5, and NCCL 2.27 loads plugins up to v4.

All runs completed, nccl-tests validated its results, and both profilers produced output on both nodes.

== Recorded operations

`ncclsee` records a time for practically every NCCL call (99.99 -- 100 %). CoMMA records every call as well, but only
60 -- 69 % of them have a time; the rest are dropped as "reclaimed" after CoMMA's timeout.

#figure(
  image(fig("recorded_by_size.pdf"), width: 100%),
  caption: [Share of the NCCL calls with a recorded time, per message size (all ranks, incl. warm-ups, 3
    iterations). Solid: nccl-tests, dashed: ncclbench.],
)

The missed calls are small and medium messages. CoMMA derives the time of an operation from its child events, by
default only from proxy operations (network transfers); kernel-channel events are off by default
(`NCCL_PROFILER_TRACK_KERNEL_CH`). Ranks that only exchange data with GPUs on their own node have no network transfer
in an operation, never get a time, and their records are reclaimed. With small messages NCCL uses a single channel,
so only the two GPUs where it crosses between the nodes use the network (25 % = 2 of 8 ranks). Larger messages use
more channels, which cross the nodes at different GPUs, until every GPU has a network transfer: from 256 KiB
(AllReduce) or 1 MiB (Broadcast) on, CoMMA records 100 %. On Chameleon (one GPU per node, every rank on the network)
this did not show up.

#figure(
  image(fig("recorded_operations.pdf"), width: 75%),
  caption: [Share of the NCCL calls with a recorded time, all sizes together.],
)

== Latencies

Both profilers report lower times than the benchmarks for small messages: the benchmarks measure the wall-clock time
of a call including the launch overhead, the profilers the operation itself. Per call, the profiler time is about
0.45 -- 0.75 of the benchmark time for small messages (CoMMA Broadcast about 0.4) and converges to about 1.0 for large
ones (at 1 GiB within about 2 %; AllReduce of 1 GiB: about 12.5 ms).

#figure(
  image(fig("latency_ratio.pdf"), width: 100%),
  caption: [ncclbench latencies over all message sizes, all 3 iterations. Top: median latency, profiler (solid) and
    benchmark (dashed) of each profiler's run. Bottom: profiler time / benchmark time of the same call, median and
    interquartile range. Profiler times are the maximum over the ranks (the same reduction as ncclbench). Note: up to
    128 KiB (AllReduce) and 32 KiB (Broadcast), CoMMA has times only for some of the ranks (see above), so its
    maximum covers only those ranks.],
)

#figure(
  image(fig("01/violins.pdf"), width: 100%),
  caption: [Distribution of the latencies for selected sizes: benchmark (left halves) vs. profiler (right halves)
    per ncclbench iteration, first iteration of the run; profiler times are the maximum over the ranks.],
)

== Consistency of the iterations

From 4 MiB on, all iterations agree within 5 % (@consistency). Below that:

- CoMMA run: the benchmark times of the iterations agree within 5 %; CoMMA's own times as well, except iteration 03
  (5 -- 12.5 % lower AllReduce times up to 32 KiB) and single Broadcast sizes (up to ±11 %).
- `ncclsee` run: larger differences, in the benchmark and in the profiler times alike, so they come from the run,
  not from the profiler. Iteration 01: Broadcast 4 -- 16 KiB up to 28 % (benchmark) / 39 % (profiler) higher;
  iteration 02: Broadcast up to 4 KiB up to 27 % / 36 % lower, alternating between sizes; iteration 03: AllReduce up
  to 2 MiB up to 15 % / 26 % higher.

All iterations ran on the same two nodes.

#figure(
  image(fig("iteration_consistency.pdf"), width: 100%),
  placement: auto,
  caption: [Lines: median latency of each iteration (ncclbench), as deviation from the median over the 3
    iterations. Band: spread of the calls within an iteration, the interquartile range in % of the iteration's
    median, averaged over the 3 iterations and centered on 0. Lines inside the band: the iterations differ less than
    the calls within one iteration.],
) <consistency>

== Open questions

- CoMMA on several GPUs per node: enable kernel-channel tracking (every rank gets a time, but CoMMA then measures
  something else than on Chameleon), run with one rank per node (as on Chameleon), or only compare CoMMA on the ranks
  it has times for
- NCCL from the module (possibly tuned for the machine, but no Inspector) vs. an own build of a newer NCCL (all three
  profilers, same version as on Chameleon)
- the outlier iterations of the `ncclsee` run (above), and iterations on different nodes, for node-to-node variation
