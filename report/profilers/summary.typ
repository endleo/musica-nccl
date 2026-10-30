// Short summary of a run, to send around. Compile from the repository root (figures are read from results/):
//   typst compile --root . report/profilers/summary.typ
//   typst compile --root . --input run=results/full/<job> report/profilers/summary.typ     other run
//   typst watch --root . report/profilers/summary.typ                                      recompile on changes
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
) <recorded>

The missed calls are small and medium messages. CoMMA derives the time of an operation from its child events, by
default only from proxy operations (network transfers); kernel-channel events are off by default
(`NCCL_PROFILER_TRACK_KERNEL_CH`). Ranks that only exchange data with GPUs on their own node have no network transfer
in an operation, never get a time, and their records are reclaimed. With small messages NCCL uses a single channel,
so only the two GPUs where it crosses between the nodes use the network (25 % = 2 of 8 ranks). Larger messages use
more channels, which cross the nodes at different GPUs, until every GPU has a network transfer: from 256 KiB
(AllReduce) or 128 KiB -- 2 MiB (Broadcast, depending on the benchmark) on, CoMMA records 100 % (details on the
algorithms and channels: @algorithms). On Chameleon (one GPU per node, every rank on the network)
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
) <fig-latency>

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

== Note: logging experiment (run 1889653)

Added later; the rest of this summary is unchanged. The logging experiment (`report/logging/summary.typ`) ran
ncclbench with `ncclsee` and NCCL's own debug log, on other nodes (n3011-021, n3012-007). It confirms or extends the
following:

- *Algorithms, protocols, channels* (@algorithms): NCCL's per-call log shows the same choice as the two tables, for
  every call of a size. No external tuner plugin was loaded, and no `NCCL_*` variable was set from outside the
  repository (environment, `/etc/nccl.conf`, `~/.nccl.conf`).
- *AllReduce at 2 MiB*: "not visible from the data" no longer holds. NCCL's cost model (latency/bandwidth table in
  the log, `src/graph/tuning.cc`) estimates Ring/LL128 at 87.5 µs and Tree/LL128 at 94.5 µs there. Tree's bandwidth
  is scaled by a fixed correction factor per size, 0.6 from 64 KiB to 2 MiB and 0.8 from 4 MiB, which makes Ring
  cheaper between about 1.1 MB and 4 MiB. The model reproduces every other choice as well.
- *Rings and trees*: the log confirms the explanations inferred from the data. Small AllReduce messages (Tree,
  channel 0) cross between the nodes only at ranks 0 #sym.arrow.l.r 4. The 8 B Broadcast ring (channel 0: 0 3 1 2 4 7 5 6) crosses at
  rank 2 #sym.arrow 4. One correction: rank 2’s network send goes through the NIC next to GPU 0, forwarded over
  NVLink, and is carried out by the proxy thread of rank 0’s process (PXN, `NET/IB/4(0)` in the log), not by rank 2’s
  own proxy as written in @bcast-example. How CoMMA still records this transfer for rank 2 has not been checked; nor
  whether it is related to the longer enqueue on ranks 0 and 2.
- *CoMMA's 25 % for AllReduce in ncclbench at 32 -- 64 KiB* (@algorithms): the guess that all trees cross the nodes at
  the same GPUs is wrong. The four tree channels cross at four different GPU pairs (0 #sym.arrow.l.r 4,
  3 #sym.arrow.l.r 7, 1 #sym.arrow.l.r 5, 2 #sym.arrow.l.r 6), so the cause is still open.
- *Threads*: `ncclsee` also records the number of warps per call (Tree: always 20; Broadcast/LL: 3 -- 16, growing
  with the size). These follow from the same thresholds in NCCL's model.

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

=== Algorithms and protocols <algorithms>

Both profilers record, for every call, the algorithm, the protocol and the number of channels NCCL chose. The two
profilers agree on every size, and the choice is the same in nccl-tests and ncclbench, in all three iterations and on
all ranks:

#figure(
  table(
    columns: 4,
    align: (left, left, left, right),
    stroke: none,
    table.hline(),
    table.header([*message size*], [*algorithm*], [*protocol*], [*channels*]),
    table.hline(stroke: 0.5pt),
    [8 B -- 4 KiB], [tree], [LL], [1],
    [8 KiB], [tree], [LL], [2],
    [16 KiB], [tree], [LL], [4],
    [32 KiB -- 64 KiB], [tree], [LL], [8],
    [128 KiB -- 1 MiB], [tree], [LL128], [8],
    [*2 MiB*], [*ring*], [LL128], [8],
    [4 MiB -- 64 MiB], [tree], [LL128], [8],
    [128 MiB -- 1 GiB], [tree], [SIMPLE], [8],
    table.hline(),
  ),
  caption: [AllReduce: algorithm, protocol and channels NCCL 2.27.7 chose per message size (2 nodes × 4 H100).],
) <algo-allreduce>

#figure(
  table(
    columns: 4,
    align: (left, left, left, right),
    stroke: none,
    table.hline(),
    table.header([*message size*], [*algorithm*], [*protocol*], [*channels*]),
    table.hline(stroke: 0.5pt),
    [8 B -- 32 KiB], [ring], [LL], [1],
    [64 KiB], [ring], [LL], [2],
    [128 KiB], [ring], [LL], [4],
    [256 KiB -- 512 KiB], [ring], [LL], [8],
    [1 MiB -- 64 MiB], [ring], [LL128], [8],
    [128 MiB -- 1 GiB], [ring], [SIMPLE], [8],
    table.hline(),
  ),
  caption: [Broadcast: algorithm, protocol and channels per message size.],
) <algo-broadcast>

- *Algorithm*: ring passes the data from GPU to GPU along a ring; tree reduces and broadcasts along trees (inside a
  node as a chain, between the nodes as a tree). AllReduce uses tree at all sizes except 2 MiB, where it uses ring;
  this is the outlier in @fig-latency (the `ncclsee` ratio jumps to about 1.0 there). Why NCCL's tuning chooses ring
  at exactly this size is not visible from the data. Broadcast always uses ring.
- *Protocol*: LL (low latency) sends 8-byte words with a flag each, so only half of the bandwidth carries data; LL128
  uses 128-byte lines with much less overhead; SIMPLE uses the full bandwidth with the highest latency per step.
  AllReduce switches from LL to LL128 at 128 KiB, Broadcast only at 1 MiB; both switch to SIMPLE at 128 MiB.
- *Channels*: the number of rings/trees used in parallel grows with the message size before the protocol changes
  (AllReduce: 8 channels from 32 KiB, Broadcast: from 256 KiB). Each channel can cross between the nodes at a
  different GPU, so more channels give more GPUs a network transfer, and CoMMA's gaps (@recorded) close as the size
  grows: completely from 256 KiB for AllReduce (both benchmarks), from 128 KiB (ncclbench) or 2 MiB (nccl-tests) for
  Broadcast. It is not one-to-one, though: AllReduce in ncclbench stays at 25 % even with 8 channels (32 -- 64 KiB),
  so there all trees apparently cross the nodes at the same GPUs (not verified).
- Since the choice is the same in both benchmarks, it does not explain why CoMMA's coverage differs between nccl-tests
  and ncclbench (AllReduce 8 -- 128 KiB, Broadcast 64 KiB -- 1 MiB); that has to come from elsewhere (e.g. nccl-tests
  enqueues 20 calls back to back).

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
  rank 2’s kernel only hands the data to its proxy thread, the network send itself runs on the CPU
- ranks 4 -- 7 (node 2) wait in their kernels until the data has crossed the network (rank 2’s proxy sends, rank 4’s
  proxy receives), about 19 -- 20 µs, then pass it on over NVLink
- `ncclsee` measures every rank's kernel; its maximum over the ranks (about 21 µs per call) is dominated by the waiting
  on node 2, as is ncclbench's time (about 21 -- 29 µs)
- CoMMA only measures the network transfer, on the only two ranks that have one: 6 µs on rank 2 (send), 8 µs on rank 4
  (receive). Its maximum over the ranks is 8 µs per call, less than half of everything else, because the waiting
  in the kernels before the transfer and the forwarding afterwards are not part of it.

The longer enqueue on rank 0 and rank 2 (the root and the network sender) was not investigated further. With more
channels (larger messages), every GPU has a network transfer and CoMMA's times converge to the others.

=== Example: Broadcast, 1 GiB <bcast-example-large>

#figure(
  table(
    columns: 7,
    align: (center, center, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*rank*], [*node*], [*CoMMA* \ `child_dur`], [*CoMMA* \ `dur`], [*`ncclsee`* \ `runtime_us`],
      [*`ncclsee`* \ enqueue], [*`ncclsee`* \ kernel]),
    table.hline(stroke: 0.5pt),
    [0 (root)], [1], [11 162], [11 203], [10 907.5], [9.0], [10 894.6],
    [1], [1], [11 512], [11 526], [11 541.8], [8.8], [11 532.8],
    [2], [1], [11 446], [11 461], [11 469.5], [9.9], [11 454.8],
    [3], [1], [11 436], [11 457], [11 551.6], [5.8], [11 544.5],
    [4], [2], [11 429], [11 439], [11 360.2], [8.1], [11 352.2],
    [5], [2], [11 429], [11 445], [11 393.8], [14.9], [11 377.2],
    [6], [2], [11 485], [11 499], [11 379.4], [9.6], [11 365.9],
    [7], [2], [11 286], [11 295], [11 336.6], [10.5], [11 325.9],
    table.hline(),
  ),
  caption: [Broadcast of 1 GiB (ncclbench, iteration 01): median times per rank in µs, over the 50 calls. CoMMA and
    `ncclsee` come from separate runs; medians are taken per column.],
)

For large messages NCCL uses 8 rings (channels) with the SIMPLE protocol. Each ring crosses between the nodes at
different GPUs, so every GPU sends or receives part of the data over the network, and the data is cut into chunks
that move along the rings one after the other (pipelining): all ranks are busy for almost the whole call, and the time
is determined by the bandwidth (1 GiB in about 11.5 ms, about 93 GB/s).

- every rank has a CoMMA time (none of its records was reclaimed), and CoMMA's network time and `ncclsee`'s kernel time
  of a rank agree within 1 %, except for the root (2.4 %, see below)
- the per-call overheads that dominated at 8 B (enqueue 6 -- 15 µs, launch, waiting for the stream) are
  insignificant against 11.5 ms
- maximum over the ranks per call (medians, iteration 01): 11.51 ms (CoMMA) and 11.55 ms (`ncclsee`), against
  11.56 ms measured by ncclbench in both runs: the profilers report 99.6 % (CoMMA) and 99.9 % (`ncclsee`) of the
  benchmark time
- the root (rank 0) finishes first: it only sends, and its kernel is done once it has handed over its last chunk,
  while the other ranks still receive and forward it. CoMMA's time on rank 0 is about 0.27 ms longer than `ncclsee`'s
  kernel time, presumably because the proxy thread is still sending the last chunks over the network after the
  kernel has handed them over (not verified)

So for large messages both profilers measure essentially the same thing as the benchmark: the data transfer, which
takes almost all of the time.

