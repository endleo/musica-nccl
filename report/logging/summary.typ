// Summary of the logging experiment (experiments/logging.sbatch). Compile from the repository root:
//   typst compile --root . report/logging/summary.typ
//   typst watch --root . report/logging/summary.typ        recompile on changes
// The text and numbers are written for run 1889653 (taken from the NCCL logs and ncclsee traces of the run); there are
// no figures yet.

#let run = sys.inputs.at("run", default: "results/logging/1889653")

#set document(title: "NCCL's decisions on MUSICA: logging experiment")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#show raw: set text(size: 0.95em)

#let tbl(..args) = table(stroke: none, inset: (x: 5pt, y: 3pt), ..args)

= NCCL's decisions on MUSICA: logging experiment

#text(fill: luma(90))[Run #raw(run), #datetime.today().display()]

== Setup

What NCCL decides (algorithm, protocol, channels), on which hardware layout, and why, taken from NCCL's own debug
log. Same hardware and software as the profiler experiment (`report/profilers/summary.typ`):

- 2 nodes with 4 H100 each (n3011-021, n3012-007), one rank per GPU; NCCL 2.27.7 from the EESSI module
- lib-ncclbench (`--blocking`, 5 warm-ups + 50 iterations), AllReduce and Broadcast, 8 B -- 1 GiB, one iteration,
  with `ncclsee` (tracer mode) to compare NCCL's log with what a profiler records
- `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING,NET`, one log per rank, and the topology and graph dumps
  (`NCCL_TOPO_DUMP_FILE`, `NCCL_GRAPH_DUMP_FILE`, written by rank 0)

The TUNING subsystem writes one line per call on rank 0, which adds to the host time of every call; the times of this
run are not comparable to the other experiments and are not evaluated here. How to read the logs: @reading.

== Environment and plugins

- No `NCCL_*` variable was set from outside the repository. NCCL reports every variable it reads that is set
  (`set by environment`, also for `/etc/nccl.conf` and `~/.nccl.conf`, whose values NCCL puts into the environment
  first); only the two dump files appear. In particular, `NCCL_ALGO` and `NCCL_PROTO` are not set, so the defaults of
  the tuning experiment are NCCL's own choice.
- Tuner: `Could not find: libnccl-tuner.so. Using internal tuner plugin.` No external tuner was involved (and none in
  the earlier runs, which had the same environment).
- Network: NCCL's built-in InfiniBand transport (no network plugin), GPUDirect RDMA on all connections (`GDRDMA`).
- Profiler: `ncclsee` loaded as `NCCLSEE-Profiler-v4`.

== Hardware as NCCL sees it

Both nodes have the same layout. Each GPU sits behind a PCIe switch together with its own 200 Gb/s InfiniBand card
(path type `PXB`), on its own NUMA node; all four GPUs are connected pairwise by NVLink (123.6 GB/s per pair, no
NVSwitch, so no NVLS).

#figure(
  tbl(
    columns: 6,
    align: (center, left, center, left, center, right),
    table.hline(),
    table.header([*GPU*], [*PCI*], [*NUMA*], [*NIC*], [*NCCL net dev*], [*NIC bandwidth*]),
    table.hline(stroke: 0.5pt),
    [0], [`06:00`], [3], [`mlx5_3`, IB], [4], [25 GB/s],
    [1], [`26:00`], [2], [`mlx5_2`, IB], [1], [25 GB/s],
    [2], [`a6:00`], [6], [`mlx5_5`, IB], [3], [25 GB/s],
    [3], [`c6:00`], [4], [`mlx5_4`, IB], [0], [25 GB/s],
    [--], [--], [0], [`mlx5_0`, RoCE], [2], [3.1 GB/s],
    table.hline(),
  ),
  caption: [GPUs and network cards per node. GPU = local rank (rank $r$ on node 1, $r + 4$ on node 2). NCCL numbers
    the network devices in the order it finds them, not by their names.],
) <hardware>

- The 25 Gb/s RoCE card `mlx5_0` is listed by NCCL but used by no channel. There is no `mlx5_1`; probably the second,
  inactive port of the same card (not checked).
- `maxBw 25.0` in the log is the bandwidth of the best path from a GPU to the network (one IB card), `totalBw 370.8`
  the largest sum of link bandwidths at one GPU (3 #sym.times 123.6 GB/s NVLink).
- The graph search finds 4 channels at 24 GB/s each, for rings and for trees alike, and NCCL uses each twice: 8
  channels, where channel $c + 4$ repeats channel $c$.

== Rings and trees

*Rings* (pattern 4, network path type `PXN`): every channel enters and leaves a node through the same NIC, but the GPU
that sends to the other node is not the one next to that NIC. It forwards the data over NVLink to the NIC's GPU, whose
process (proxy thread) does the network send (PXN). In the log: `[send] via NET/IB/4(0)` = NIC dev 4, through the
proxy of local rank 0.

#figure(
  tbl(
    columns: 4,
    align: (center, left, left, left),
    table.hline(),
    table.header([*channel*], [*ring*], [*node 1 #sym.arrow node 2*], [*node 2 #sym.arrow node 1*]),
    table.hline(stroke: 0.5pt),
    [0, 4], [0 3 1 2 4 7 5 6], [2 #sym.arrow 4 (NIC of GPU 0)], [6 #sym.arrow 0 (NIC of GPU 0)],
    [1, 5], [0 2 1 7 4 6 5 3], [1 #sym.arrow 7 (NIC of GPU 3)], [5 #sym.arrow 3 (NIC of GPU 3)],
    [2, 6], [0 2 5 7 4 6 1 3], [2 #sym.arrow 5 (NIC of GPU 1)], [6 #sym.arrow 1 (NIC of GPU 1)],
    [3, 7], [0 6 5 7 4 2 1 3], [0 #sym.arrow 6 (NIC of GPU 2)], [4 #sym.arrow 2 (NIC of GPU 2)],
    table.hline(),
  ),
  caption: [Rings (`Channel NN/08` lines of rank 0) and where they cross between the nodes (`via NET` lines of all
    ranks).],
) <rings>

*Trees* (pattern 3, path type `PXB`): every GPU uses its own NIC. In each channel, the GPUs of a node form a chain,
and the end of the chain on node 2 is connected to the end on node 1; the second tree of a channel pair (channel
$c + 4$) is the mirror image, with the root on the other node.

#figure(
  tbl(
    columns: 4,
    align: (center, center, left, center),
    table.hline(),
    table.header([*channel*], [*root*], [*chains (child #sym.arrow parent)*], [*between nodes*]),
    table.hline(stroke: 0.5pt),
    [0 (4)], [0 (4)], [2 #sym.arrow 1 #sym.arrow 3 #sym.arrow 0, #h(0.5em) 6 #sym.arrow 5 #sym.arrow 7 #sym.arrow 4],
      [0 #sym.arrow.l.r 4],
    [1 (5)], [3 (7)], [0 #sym.arrow 2 #sym.arrow 1 #sym.arrow 3, #h(0.5em) 4 #sym.arrow 6 #sym.arrow 5 #sym.arrow 7],
      [3 #sym.arrow.l.r 7],
    [2 (6)], [1 (5)], [2 #sym.arrow 3 #sym.arrow 0 #sym.arrow 1, #h(0.5em) 6 #sym.arrow 7 #sym.arrow 4 #sym.arrow 5],
      [1 #sym.arrow.l.r 5],
    [3 (7)], [2 (6)], [3 #sym.arrow 1 #sym.arrow 0 #sym.arrow 2, #h(0.5em) 7 #sym.arrow 5 #sym.arrow 4 #sym.arrow 6],
      [2 #sym.arrow.l.r 6],
    table.hline(),
  ),
  caption: [Trees per channel (`Trees` lines of all ranks); in parentheses the mirrored tree of the channel pair.],
) <trees>

Consequences for the profiler experiment:

- small AllReduce messages (Tree, channel 0 only) cross the nodes only at ranks 0 and 4, small Broadcasts (ring
  channel 0, root 0) only at rank 2 #sym.arrow 4: the ranks for which CoMMA has times
- rank 2’s network send is done by the proxy thread of rank 0’s process (PXN), not by rank 2’s own
- with 8 channels, the trees cross the nodes at all 8 GPUs; CoMMA's 25 % for AllReduce in ncclbench at 32 -- 64 KiB
  (8 channels) is not explained by the layout

== Algorithm selection

NCCL's per-call log (rank 0) and `ncclsee`'s records (all ranks) agree on every call; every call of a size gets the
same choice. Same as in the profiler experiment, on other nodes. Warps (threads per block / 32) from `ncclsee`.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    tbl(
      columns: 5,
      align: (left, left, left, right, right),
      table.hline(),
      table.header([*size*], [*algo*], [*proto*], [*ch.*], [*warps*]),
      table.hline(stroke: 0.5pt),
      [8 B -- 4 KiB], [Tree], [LL], [1], [20],
      [8 KiB], [Tree], [LL], [2], [20],
      [16 KiB], [Tree], [LL], [4], [20],
      [32 -- 64 KiB], [Tree], [LL], [8], [20],
      [128 KiB -- 1 MiB], [Tree], [LL128], [8], [20],
      [*2 MiB*], [*Ring*], [LL128], [8], [20],
      [4 -- 64 MiB], [Tree], [LL128], [8], [20],
      [128 MiB -- 1 GiB], [Tree], [Simple], [8], [20],
      table.hline(),
    ),
    caption: [AllReduce],
  ),
  figure(
    tbl(
      columns: 5,
      align: (left, left, left, right, right),
      table.hline(),
      table.header([*size*], [*algo*], [*proto*], [*ch.*], [*warps*]),
      table.hline(stroke: 0.5pt),
      [8 B -- 4 KiB], [Ring], [LL], [1], [3],
      [8 KiB], [Ring], [LL], [1], [4],
      [16 KiB], [Ring], [LL], [1], [8],
      [32 KiB], [Ring], [LL], [1], [16],
      [64 KiB], [Ring], [LL], [2], [16],
      [128 KiB], [Ring], [LL], [4], [16],
      [256 -- 512 KiB], [Ring], [LL], [8], [16],
      [1 -- 64 MiB], [Ring], [LL128], [8], [20],
      [128 MiB -- 1 GiB], [Ring], [Simple], [8], [17],
      table.hline(),
    ),
    caption: [Broadcast (only Ring is available)],
  ),
)

== NCCL's cost model

At init, rank 0 logs the model NCCL chooses with: a latency (µs) and a bandwidth (GB/s) per operation, algorithm and
protocol (@model-table). For every call, NCCL estimates the time of every available combination and takes the
minimum (NCCL 2.27.7, `ncclTopoGetAlgoTime` in `src/graph/tuning.cc`; numPipeOps = 1 here):

$ t = "lat" + "bytes" / (1000 dot "bw") $

with two corrections: for AllReduce with Tree, the bandwidth is multiplied by a fixed factor per message size and
protocol (`treeCorrectionFactor`, indexed by $log_2("bytes"\/64)$, 0.4 -- 1.0); for AllReduce with Ring/Simple across
nodes, the latency is multiplied by 1.4 once the message is larger than 64 bytes per channel and rank.

#figure(
  tbl(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([], table.cell(colspan: 3, align: center)[*Tree*], table.cell(colspan: 3, align: center)[*Ring*]),
    table.header([], [LL], [LL128], [Simple], [LL], [LL128], [Simple]),
    table.hline(stroke: 0.5pt),
    [AllReduce], [20.4 / 9.6], [38.5 / 62.4], [60.4 / 70.4], [36.0 / 12.9], [46.0 / 50.5], [108.4 / 54.9],
    [Broadcast], [--], [--], [--], [28.2 / 22.5], [46.0 / 88.3], [120.4 / 96.0],
    table.hline(),
  ),
  caption: [Latency (µs) / bandwidth (GB/s) per algorithm and protocol (TUNING table of rank 0). CollNet, NVLS
    and PAT are not available on this system.],
) <model-table>

With these numbers, the model reproduces every choice in the tables above. Estimates around the switch points (in
µs, \* = chosen):

#figure(
  tbl(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([*AllReduce*], [Tree/LL], [Tree/LL128], [Tree/Simple], [Ring/LL], [Ring/LL128], [Ring/Simple]),
    table.hline(stroke: 0.5pt),
    [64 KiB], [31.8\*], [40.3], [62.0], [41.1], [47.3], [153.0],
    [128 KiB], [47.7], [42.0\*], [64.1], [46.2], [48.6], [154.1],
    [1 MiB], [238.9], [66.5\*], [90.2], [117.3], [66.8], [170.9],
    [2 MiB], [384.5], [94.5], [110.0], [198.6], [87.5\*], [190.0],
    [4 MiB], [644.6], [122.5\*], [145.5], [361.1], [129.1], [228.2],
    [64 MiB], [7010.9], [1233.5\*], [1252.0], [5238.2], [1374.9], [1374.1],
    [128 MiB], [14001.4], [2189.4], [2178.7\*], [10440.5], [2703.8], [2596.5],
    table.hline(),
  ),
  caption: [AllReduce: NCCL's estimated time per combination.],
)

- *AllReduce at 2 MiB*: Tree's LL128 bandwidth is scaled by 0.6 from 64 KiB to 2 MiB, and by 0.8 from 4 MiB on.
  With 0.6, Ring/LL128 (7.5 µs more latency, but more effective bandwidth) is cheaper from about 1.1 MB up to 4 MiB;
  2 MiB is the only power of two in that range. 1.5 MiB or 3 MiB should use Ring as well.
- *Broadcast*: LL #sym.arrow LL128 at about 537 KB (between 512 KiB and 1 MiB), LL128 #sym.arrow Simple at about
  82 MB (between 64 and 128 MiB).
- *Close calls* (less than 2 % apart): AllReduce 1 MiB (Tree/LL128 66.5 vs. Ring/LL128 66.8), 64 MiB (Tree/LL128
  1233.5 vs. Tree/Simple 1252.0), 128 MiB (Tree/Simple 2178.7 vs. Tree/LL128 2189.4); Broadcast 512 KiB (LL 51.5
  vs. LL128 51.9), 64 MiB (LL128 806.0 vs. Simple 819.5).
- *Channels and warps* follow from the `threadThresholds` in the log (Tree 8/8/64, Ring 64/8/64 bytes per thread for
  LL/LL128/Simple): NCCL removes channels, and then halves the threads, while the message is smaller than
  channels #sym.times threads #sym.times threshold (minimum 3 warps). Tree always uses the maximum of 640 threads
  (20 warps); Simple adds one warp for Ring (17).

Whether the estimates match the measured times is open: they are compared with the ncclbench times of a run
without logging in the tuning experiment.

== Consequences for the tuning experiment

- Ring vs. Tree for AllReduce around 2 MiB: pinned `NCCL_ALGO=allreduce:ring` / `allreduce:tree`, with additional
  sizes between the powers of two (ncclbench takes any list of sizes), e.g. 1.5 and 3 MiB
- the close calls above, with pinned protocols: `NCCL_PROTO=allreduce:LL128` / `allreduce:Simple` (64 -- 128 MiB),
  `broadcast:LL` / `broadcast:LL128` (512 KiB -- 1 MiB), `broadcast:LL128` / `broadcast:Simple` (64 -- 128 MiB)
- `default` is NCCL's own choice: nothing from the environment or a conf file influences it

#pagebreak()
#set heading(numbering: (..n) => if n.pos().len() >= 2 { numbering("A.1", ..n.pos().slice(1)) },
  supplement: [Appendix])
#counter(heading).update(0)

== Reading the logs <reading>

Files: `profiler-out.tar.gz` of the run, `profiler-out/ncclsee/<test>/<host>/`: `nccl-rank<N>.log` per rank (rank 0:
about 2100 lines, the others about 500), `topo-rank0.xml` and `graph-rank0.xml` on rank 0’s node, next to
`env-rank<N>.txt`.

Every line starts with `host:pid:tid [d] NCCL INFO`, where `d` is the CUDA device (= local rank), not the rank. The
log of rank 0 contains, in this order:

+ *Version and network*: `NCCL version`, `NET/Plugin: Could not find: libnccl-net.so` (built-in transport), the
  network devices (`NET/IB : Using [0]mlx5_4:1/IB ...`), `GPU Direct RDMA Enabled for HCA ...` (HCA: host channel
  adapter, the InfiniBand term for a NIC).
+ *Topology*, from `=== System : maxBw .. totalBw .. ===`: a tree of NUMA nodes (`CPU/0-<numa>`), PCIe switches, GPUs
  (`GPU/0-<busid>`), NICs and their links with bandwidths in GB/s (`PCI[48.0]`, `NVL[123.6]`, `NET[25.0]`), then
  every path from each GPU/NIC to every other device as `(hops/bandwidth/type)`. Path types from close to far:
  `LOC`, `NVL` (NVLink), `NVB` (NVLink via another GPU), `C2C`, `PIX` (one PCIe switch), `PXB` (several PCIe
  switches), `PXN` (NVLink to another GPU, then its PCIe path), `PHB` (through the CPU's PCIe host bridge), `SYS`
  (between NUMA nodes / sockets). The same as XML in `topo-rank0.xml`.
+ *Graph search*: `Pattern <p>, crossNic .., nChannels .., bw <intra>/<inter>, type <intra>/<inter>` per graph
  (pattern 4 = ring, 3 = tree, see `src/include/graph.h`), followed by one line per channel, the path inside the node
  from NIC to NIC. The same as XML in `graph-rank0.xml`.
+ *Layout of the communicator*: `Channel NN/08 : <ranks>` (ring order over all ranks), `Ring NN : prev -> rank -> next`, `Tree c : parent -> rank -> children`, `Trees [c] children->rank->parent` (-1 = none).
+ *Plugins and model*: `PROFILER/Plugin: loaded ...`, the latency/bandwidth table (`Algorithm | Tree | Ring ...`,
  `0.0/0.0` = not available), `threadThresholds`, `TUNER/Plugin: ... Using internal tuner plugin`, `Init timings`.
+ *Connections* (also on the other ranks): `Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM` (NVLink), `Channel 00/0 : 2[2] -> 4[0] [send] via NET/IB/4(0)/GDRDMA` (network: NCCL net dev 4, through the proxy of local rank 0 = PXN; no
  parentheses = the rank's own proxy), `Connected all rings, use ring PXN 1 GDR 1`, `Connected all trees`. NCCL
  connects lazily (`NCCL_RUNTIME_CONNECT`), so these lines appear at the first call that needs the connection.
+ *Per call* (TUNING, rank 0 only): `AllReduce: 2097152 Bytes -> Algo RING proto LL128 channel{Lo..Hi}={0..7}`, one
  line per call (55 per size: 5 warm-ups, 50 iterations).

Useful one-liners, run in `profiler-out/ncclsee/`:

```sh
# settings NCCL picked up
grep -h "set by environment" */*/nccl-rank*.log
# who crosses the nodes, through which NIC
grep -h "via NET" */*/nccl-rank*.log | cut -d' ' -f5- | sort -u
# selection per size
grep -h "Bytes -> Algo" all_reduce_bench/*/nccl-rank0.log | cut -d' ' -f5- | sort | uniq -c
```
