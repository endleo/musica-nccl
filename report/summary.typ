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
of a call including the launch overhead, the profilers only parts of the operation (what exactly: @measured). Per
call, the profiler time is about 0.45 -- 0.75 of the benchmark time for small messages and converges to about 1.0 for
large ones (at 1 GiB within about 2 %; AllReduce of 1 GiB: about 12.5 ms). CoMMA's Broadcast times are particularly
low for small messages (about 0.4): there it only measures the network transfer between the nodes, see the example in
@bcast-example.

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

#pagebreak()
// appendix: A, A.1, ... (the main part has no numbered headings); references read "Appendix A.1"
#set heading(numbering: (..n) => if n.pos().len() >= 2 { numbering("A.1", ..n.pos().slice(1)) },
  supplement: [Appendix])
#counter(heading).update(0)

== What is measured <measured>

All of this is taken from the source code of the versions used (lib-ncclbench `3b9d2c4`, nccl-tests v2.19.6, CoMMA
`1d5b967`, `ncclsee` `0e8433a`).

=== Benchmarks

*ncclbench* (`--blocking`, the per-call times used in the latency figures). For every call, on every rank:
`MPI_Barrier`; start of the timer (`MPI_Wtime`); the NCCL call; `cudaStreamSynchronize`; end of the timer. The time
of a call is the maximum of these intervals over all ranks, minus the mean duration of an empty
`cudaStreamSynchronize` (measured once, 1.4 µs here). So it is the wall-clock time of the call on the host: enqueue,
kernel launch, execution, and waiting for the GPU. 5 warm-up calls per size are not reported. Broadcast uses rank 0 as
root.

*nccl-tests* (default mode). Per size: one synchronizing call and a barrier; start of the timer; 20 calls enqueued
back to back without waiting; one stream synchronization at the end; end of the timer; divided by 20. The result is
averaged over the ranks (not the maximum). Since the calls are not waited for individually, the enqueue and launch of
a call overlap with the execution of the previous one: this is the time per call in a stream of calls, not the
latency of a single call, and there are no per-call times.

Both benchmark times include the profiler's own overhead on the host: for 8 B Broadcast, ncclbench measures about
21 µs in the CoMMA runs and 23 -- 29 µs in the `ncclsee` runs (a run without profiler as baseline is still open).

=== Profilers

#table(
  columns: (auto, 1fr, 1fr),
  align: left,
  stroke: 0.5pt + luma(180),
  table.header([], [*`ncclsee`* (`runtime_us`)], [*CoMMA* (`child_dur`)]),
  [What], [enqueue of the call on the host + execution of the GPU kernel],
    [the call's network transfers (NCCL proxy operations), from the earliest start to the latest end],
  [Clock], [enqueue: CPU; kernel: GPU timer, read by the kernel itself], [CPU (NCCL's proxy thread)],
  [Computed as], [CPU duration of the collective event + (last kernel channel end − first kernel channel start)],
    [latest proxy-op end − earliest proxy-op start],
  [Not included], [the gap between enqueue and kernel start (launch)],
    [enqueue, launch, GPU work before and after the transfer],
  [Ranks], [all], [only ranks with a network transfer in the call],
  [Resolution], [below 1 µs], [1 µs],
)

The `ncclsee` time thus adds two intervals measured on two different clocks; it is not an interval on a single
timeline. CoMMA's alternative field `dur` (not used) runs from the start of the call on the host to the end of the
last network transfer. With `NCCL_PROFILER_TRACK_KERNEL_CH=true` (off by default), CoMMA would also count the kernel
channels as child events.

For the comparison with ncclbench, the profiler times of a call are reduced to the maximum over the ranks, like
ncclbench's own times. For nccl-tests, all records of a size are averaged (including its warm-up, synchronization and
validation calls, which the profilers cannot tell apart).

=== Example: Broadcast, 8 B <bcast-example>

#figure(
  table(
    columns: 7,
    align: (center, center, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*rank*], [*node*], [*CoMMA* \ `child_dur`], [*CoMMA* \ `dur`], [*`ncclsee`* \ `runtime_us`],
      [*`ncclsee`* \ enqueue], [*`ncclsee`* \ kernel]),
    table.hline(stroke: 0.5pt),
    [0 (root)], [1], [--], [--], [5.9], [4.4], [1.5],
    [1], [1], [--], [--], [4.7], [1.5], [3.1],
    [2], [1], [6.0], [14.0], [15.6], [9.6], [3.5],
    [3], [1], [--], [--], [3.3], [1.6], [1.7],
    [4], [2], [8.0], [15.0], [20.7], [1.9], [18.8],
    [5], [2], [--], [--], [21.9], [1.6], [20.3],
    [6], [2], [--], [--], [22.3], [1.6], [20.4],
    [7], [2], [--], [--], [21.0], [1.6], [19.2],
    table.hline(),
  ),
  caption: [Broadcast of 8 B (ncclbench, iteration 01): median times per rank in µs, over the 50 calls. CoMMA and
    `ncclsee` come from separate runs. Medians are taken per column, so `runtime_us` is not exactly the sum of the
    two columns to its right.],
)

For small messages NCCL uses a single ring through all 8 GPUs. Rank 0 (the root) passes the data on within its node,
rank 2 sends it over the network, rank 4 receives it and passes it on to the other GPUs of the second node. The kernels
of all ranks start at about the same time (ncclbench synchronizes the ranks with a barrier before every call):

- ranks 0 -- 3 (node 1, with the root) get the data over NVLink and are done after 1.5 -- 3.5 µs of kernel time;
  rank 2's kernel only hands the data to its proxy thread, the network send itself runs on the CPU
- ranks 4 -- 7 (node 2) wait in their kernels until the data has crossed the network (rank 2's proxy sends, rank 4's
  proxy receives), about 19 -- 20 µs, then pass it on over NVLink
- `ncclsee` measures every rank's kernel; its maximum over the ranks (about 21 µs per call) is dominated by the waiting
  on node 2, as is ncclbench's time (about 21 -- 29 µs)
- CoMMA only measures the network transfer, on the only two ranks that have one: 6 µs on rank 2 (send), 8 µs on rank 4
  (receive). Its maximum over the ranks is 8 µs per call, less than half of everything else, because the waiting
  in the kernels before the transfer and the forwarding afterwards are not part of it.

The longer enqueue on rank 0 and rank 2 (the root and the network sender) was not investigated further. With more
channels (larger messages), every GPU has a network transfer and CoMMA's times converge to the others.

