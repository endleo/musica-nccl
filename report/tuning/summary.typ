// Summary of the tuning experiment (experiments/tuning.sbatch). Compile from the repository root (figures are read
// from results/, written by `uv run analysis/analyze.py results/tuning/<job>`):
//   typst compile --root . report/tuning/summary.typ
//   typst compile --root . --input run=results/tuning/<job> report/tuning/summary.typ     other run
//   typst watch --root . report/tuning/summary.typ                                        recompile on changes
// The text and numbers are written for run 1893400 (from choices.txt, model_vs_measured.txt and best_vs_default.txt
// in the run folder).

#let run = sys.inputs.at("run", default: "results/tuning/1893400")
#let fig(name) = "/" + run + "/" + name

#set document(title: "NCCL's choice vs. pinned algorithms and protocols on MUSICA: tuning experiment")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#show raw: set text(size: 0.95em)

#let tbl(..args) = table(stroke: none, inset: (x: 5pt, y: 3pt), ..args)

= NCCL's choice vs. pinned algorithms and protocols: tuning experiment

#text(fill: luma(90))[Run #raw(run), #datetime.today().display()]

== Setup

Is NCCL's choice of algorithm and protocol the fastest one, and how well does its cost model (logging experiment,
`report/logging/summary.typ`) predict the measured times? Every available combination is pinned once and measured
like NCCL's own choice:

- 2 nodes with 4 H100 each (n3010-020, n3010-021; other nodes than the logging run), one rank per GPU; NCCL 2.27.7
  from the EESSI module
- lib-ncclbench (`--blocking`, 5 warm-ups + 50 iterations), AllReduce and Broadcast, 8 B -- 1 GiB in powers of two
  plus 1.5 and 3 MiB; `ncclsee` (tracer mode) records what NCCL did in every call
- 7 variants: `default` (no `NCCL_ALGO`/`NCCL_PROTO`, NCCL's choice) and the AllReduce combinations Tree and Ring
  #sym.times LL, LL128 and Simple, pinned with `NCCL_ALGO=allreduce:<algo>` and `NCCL_PROTO=allreduce:<proto>`. The
  `ring_*` variants also pin Broadcast's protocol (Broadcast only has Ring); in the `tree_*` variants Broadcast runs
  with NCCL's choice.
- 3 rounds in one job on the same nodes, each running all 7 variants once (order rotated by one per round): 42
  benchmark runs, about 11 minutes

Statistics as in the profiler experiment: ncclbench's time per call (maximum over the ranks), medians and quartiles
over the 150 calls of the 3 rounds together. The analysis also writes the full tables as text files (`choices.txt`,
`model_vs_measured.txt`, `best_vs_default.txt` in the run folder).

== Run and pinned settings

All 42 runs completed (no `FAILED` in the job output; Slurm reported a hung prolog on n3010-021 at the start, without
effect). Every benchmark output has all 30 sizes #sym.times 50 iterations, and `ncclsee` recorded exactly
8 ranks #sym.times (5 + 50) = 440 calls for every round, variant, operation and size. The variant's `NCCL_ALGO` and
`NCCL_PROTO` are in the environment of every rank (`env-rank<N>.txt`).

*Every call of a pinned variant used the pinned algorithm and protocol*, on every rank and in every round. The
number of channels and warps stays NCCL's choice and follows the thresholds of the pinned combination
(@pinned-channels). Every (variant, size) got the same algorithm, protocol, channels and warps in all three rounds.

#figure(
  tbl(
    columns: 4,
    align: (left, left, left, left),
    table.hline(),
    table.header([*pinned*], [*channels (1 #sym.arrow 8)*], [*warps*], [*Broadcast channels*]),
    table.hline(stroke: 0.5pt),
    [Tree/LL], [1 up to 4 KiB, 8 from 32 KiB], [20], [--],
    [Tree/LL128], [1 up to 8 KiB, 8 from 64 KiB], [20], [--],
    [Tree/Simple], [1 up to 32 KiB, 8 from 256 KiB], [20], [--],
    [Ring/LL], [1 up to 32 KiB, 8 from 256 KiB], [3 #sym.arrow 16], [as AllReduce],
    [Ring/LL128], [1 up to 8 KiB, 8 from 64 KiB], [10, 20 from 8 KiB], [1 up to 16 KiB, 8 from 128 KiB],
    [Ring/Simple], [1 up to 32 KiB, 8 from 256 KiB], [3 #sym.arrow 17], [as AllReduce],
    table.hline(),
  ),
  caption: [Channels and warps NCCL used with each pinned combination (`choices.txt`); warps are the same for
    Broadcast.],
) <pinned-channels>

*NCCL's choice* (`default`) is identical to the logging run (algorithm, protocol, channels and warps at every size,
on other nodes) and to the minimum of the cost model at every size. The new sizes 1.5 and 3 MiB use Ring/LL128 for
AllReduce, as the model predicted: Ring from 1.5 to 3 MiB, Tree below and from 4 MiB. The Broadcast runs of the
`tree_*` variants made the same choice as `default`.

== NCCL's estimate vs. measured time

The estimate per call is NCCL's model (`ncclTopoGetAlgoTime`, with the latency/bandwidth table of the logging run,
reproduced offline). @model-overview shows every combination over all sizes, @model-violins the distributions at
selected sizes in absolute times.

#figure(
  image(fig("model_vs_measured.pdf"), width: 100%),
  caption: [Measured time per call (box: quartiles, whiskers: 5 -- 95 %, 3 rounds) and NCCL's estimate (point), both
    divided by the measured median, so the box is centered on 1 and the point shows how far off the estimate is.
    Triangles at the bottom: sizes where NCCL chooses this combination.],
) <model-overview>

- *The estimate is usually outside the measured distribution*: the spread of the calls is a few percent, the error of
  the model larger. Of the 270 (operation, size, combination) estimates, 116 are below every measured call and 63
  above every call; 22 fall into the interquartile range, mostly Broadcast/LL up to 256 KiB, whose distribution is
  wide (see below).
- *NCCL's choice, AllReduce*: the estimate is 0.76 -- 0.93 of the measured median up to 64 KiB (below every call):
  ncclbench measures the call on the host including enqueue, launch and synchronization, which the model does not
  contain; the gap grows from 1.6 µs (8 B) to 7.8 µs (32 KiB). 128 KiB -- 64 MiB: 0.87 -- 1.06. From 128 MiB on
  (Tree/Simple) the model is pessimistic: 1.21 -- 1.30.
- *NCCL's choice, Broadcast*: LL up to 32 KiB within 0.91 -- 1.05 (inside the distribution); from 64 KiB the model is
  optimistic, 0.60 -- 0.85 between 64 KiB and 32 MiB (minimum at 4 MiB), approaching 1 at 1 GiB.
- *Combinations NCCL does not choose* are off by more: Tree/LL for AllReduce above 128 KiB is estimated up to 3.5
  times its measured time; the latencies of LL128 and Simple for Broadcast are too high (8 B: estimated 46 and
  120 µs, measured 27.5 and 44 µs). Ring/LL128 for AllReduce is underestimated by up to 35 % between 64 B and 1 MiB.

#figure(
  image(fig("model_violins.pdf"), width: 100%),
  caption: [Measured calls (violins, 3 rounds) and NCCL's estimate (points) per combination at selected sizes; `*`:
    NCCL's choice. Log scale per panel.],
) <model-violins>

== Pinned vs. NCCL's choice

#figure(
  image(fig("pinned_vs_default.pdf"), width: 100%),
  caption: [Top: median latency per combination; NCCL's choice in black. Bottom: median of the pinned combination /
    median of NCCL's choice (clipped at 1.5); below 1 = the pinned combination is faster. Colored bands: range of the
    ratio over the 3 rounds (each round's medians). Gray band: noise floor, the largest difference between two runs
    of the same configuration in the same round (NCCL's own combination pinned vs. `default`; for Broadcast also the
    `tree_*` runs). Circled: faster than NCCL's choice in every round and by more than the noise floor.],
) <pinned>

NCCL's choice is the fastest combination, or within the noise of it, at 27 of the 30 sizes of each operation. Clearly faster than NCCL's choice (in all three rounds and beyond the noise floor):

#figure(
  tbl(
    columns: 7,
    align: (left, left, right, left, right, right, right),
    table.hline(),
    table.header([*operation*], [*NCCL's choice*], [*median*], [*faster*], [*median*], [*ratio*], [*noise*]),
    table.hline(stroke: 0.5pt),
    [AllReduce 4 MiB], [Tree/LL128], [136.1], [Ring/LL128], [126.7], [0.93], [2.9 %],
    [AllReduce 32 MiB], [Tree/LL128], [682.2], [Tree/Simple], [617.1], [0.90], [0.3 %],
    [AllReduce 64 MiB], [Tree/LL128], [1168.4], [Tree/Simple], [1000.6], [0.86], [0.8 %],
    [Broadcast 256 KiB], [Ring/LL], [54.5], [Ring/LL128], [49.3], [0.90], [9.6 %],
    [Broadcast 512 KiB], [Ring/LL], [65.9], [Ring/LL128], [55.3], [0.84], [10.1 %],
    [Broadcast 128 MiB], [Ring/Simple], [1700.4], [Ring/LL128], [1632.8], [0.96], [0.1 %],
    table.hline(),
  ),
  caption: [Sizes where a pinned combination beats NCCL's choice (medians in µs over 3 rounds). In every case it
    is the model's second (or, AllReduce 32 MiB, third) choice.],
) <wins>

- *Ring vs. Tree for AllReduce around 2 MiB*: Ring/LL128 is faster than Tree/LL128 at 1.5, 2 and 3 MiB (by 5 %,
  15 % and 4 %), so NCCL's switch to Ring is right, and also at 4 MiB (by 7 %), where NCCL already uses Tree again;
  at 1 MiB and 8 MiB Tree is faster (by 8 % and 6 %). The measured Ring range extends from between 1 and 1.5 MiB to
  between 4 and 8 MiB; in the model, Tree's correction factor rises from 0.6 to 0.8 at 4 MiB, which is too early for
  this system.
- *LL128 vs. Simple for AllReduce at 32 -- 128 MiB*: the model overestimates Tree/Simple by 20 -- 25 % (Tree/LL128:
  within 7 %), so it switches to Simple only at 128 MiB; measured, Simple is faster from 32 MiB on.
- *LL vs. LL128 for Broadcast*: measured, LL128 is faster from 256 KiB on (model: from about 537 KB), and also at
  128 MiB, where NCCL switches to Simple (equal at 256 MiB, Simple faster from 512 MiB). At most sizes up to 8 KiB,
  LL128's median is up to 8 % lower than LL's, but within the noise of Broadcast/LL there.
- *The close calls of the logging report* (estimates less than 2 % apart): NCCL is right at AllReduce 1 MiB (Tree
  8 % faster than Ring), 128 MiB (Simple 18 % faster than LL128) and Broadcast 64 MiB (LL128 11 % faster than
  Simple), and wrong at AllReduce 64 MiB and Broadcast 512 KiB (@wins). The real differences are much larger than
  the model's.

== Consistency of the rounds

#figure(
  image(fig("round_consistency.pdf"), width: 100%),
  caption: [Lines: median latency of each round, as deviation from the median over the 3 rounds, per variant.
    Band: interquartile range within a round in % of its median, averaged over the rounds, centered on 0 (as
    `iteration_consistency.pdf` of the profiler experiment, with rounds instead of iterations).],
) <rounds>

- From 1 MiB on, the rounds agree within about 2 % in every variant, except AllReduce Ring/LL (round 03 up to 13 %
  slower, and 10 -- 21 % between 64 B and 1 MiB) and AllReduce Tree/LL128 (round 01 about 5 -- 7 % slower at
  128 KiB -- 4 MiB).
- Below 1 MiB, the AllReduce variants agree within 2 -- 4 % (apart from these two), the Broadcast variants with LL128
  and Simple within 3 %.
- *Broadcast with LL up to 16 KiB* (NCCL's choice, and Ring/LL) differs by up to 23 % between rounds, and the spread
  within a round is large (interquartile range up to 45 % of the median). The times are bimodal: per round, the
  calls cluster at about 20 µs and about 28 µs (8 B), and the median falls into one or the other cluster depending
  on how many calls hit the fast one. This is the same pattern as the iterations of the `ncclsee` run in the profiler
  experiment (run 1863257: Broadcast up to 4 KiB up to 27 % lower in one iteration, 4 -- 16 KiB up to 28 % higher in
  another), there attributed to the run. It may also explain the 21 µs (CoMMA runs) vs. 23 -- 29 µs (`ncclsee` runs)
  of ncclbench's 8 B Broadcast, so far read as profiler overhead (not checked).
- Two runs of the same configuration in the same round (minutes apart, same nodes) differ by up to 3 % for AllReduce
  below 128 KiB and at 1.5 -- 4 MiB, up to 6 % at 128 KiB -- 1 MiB and below 1 % from 8 MiB on; for Broadcast by up to
  28 % up to 16 KiB (LL, see above), about 10 % at 256 -- 512 KiB and below 2 % from 1 MiB on. This is the noise floor
  in @pinned.

== Open questions

- The bimodal Broadcast/LL times at small sizes: where the two clusters come from (e.g. which rank finishes last,
  `ncclsee`'s per-rank kernel times), and whether they also occur without a profiler
- The same comparison with nccl-tests (calls queued back to back), and without `ncclsee`
- A second run on other nodes, to see whether the wins in @wins are a property of the system or of these two nodes
- Whether a tuner plugin (or `NCCL_PROTO`/`NCCL_ALGO` per size range) with the measured switch points pays off in an
  application: the gains are 4 -- 16 % at single sizes
