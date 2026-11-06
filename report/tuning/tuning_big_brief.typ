#set page(paper: "a4", margin: (x: 1.5cm, y: 1.8cm), numbering: "1")
#set text(size: 9pt)
#let tbl(..args) = table(stroke: none, inset: (x: 4pt, y: 2pt), table.hline(y: 1, stroke: 0.5pt), ..args)
#set heading(numbering: "1.1")
#set document(title: [NCCL on MUSICA on 8/16 Nodes (32/64 GPUs total)])
#title()


- Two experiments for 8 Nodes (1911532 & 1912206) and one for 16 Nodes (1914035)
- Two notable weird results:
  - Broadcast is slow for 32 KiB (slower than for 64 KiB), even though algo/proto/channel is the same for both 32 and 64 (warps are different, though -- see @choices)
  - For broadcast around 4 MiB, where NCCL chooses LL128, the behavior is odd -- latency rises fairly steeply up until then, but almost plateaus for 4 MiB - 8 MiB. Also, LL actually performs better than LL128 at this point, which is unexpected (LL128 should be beating LL at this size)
- For both anomalies, there are plausible explanations relating to how NCCL chooses thread count and chunk size (see below)

#outline(depth:1)

= Plots: pinned variants vs. NCCL's choice <plots>

== 8 Nodes (1911532 & 1912206):

#figure(image("/results/tuning/1911532/pinned_vs_default.pdf", width: 80%))

#figure(image("/results/tuning/1912206/pinned_vs_default.pdf", width: 80%))

== 16 Nodes (1914035):

#figure(image("/results/tuning/1914035/pinned_vs_default.pdf", width: 80%))

= Which algo/proto NCCL chooses itself (default) <choices>

== 8 Nodes:

The choices are the same for both (8 node) experiments.

#grid(columns: 2, column-gutter: 1em, row-gutter: 1.2em,
  [*Allreduce* (NCCL's choice)
  #tbl(columns: 5, align: (left, left, left, right, right),
    [size], [algo], [proto], [channels], [warps],
    [8 B - 4 KiB], [tree], [ll], [1], [20],
    [8 KiB], [tree], [ll], [2], [20],
    [16 KiB], [tree], [ll], [4], [20],
    [32 KiB - 128 KiB], [tree], [ll], [8], [20],
    [256 KiB - 8 MiB], [tree], [ll128], [8], [20],
    [16 MiB - 256 MiB], [ring], [ll128], [8], [20],
    [512 MiB - 1 GiB], [ring], [simple], [8], [17])],
)

#grid(columns: 2, column-gutter: 1em, row-gutter: 1.2em,
  [*Broadcast* (NCCL's choice)
  #tbl(columns: 5, align: (left, left, left, right, right),
    [size], [algo], [proto], [channels], [warps],
    [8 B - 16 KiB], [ring], [ll], [1], [3],
    [32 KiB], [ring], [ll], [1], [4],
    [64 KiB], [ring], [ll], [1], [8],
    [128 KiB], [ring], [ll], [1], [16],
    [256 KiB], [ring], [ll], [2], [16],
    [512 KiB], [ring], [ll], [4], [16],
    [1 MiB], [ring], [ll], [8], [16],
    [1536 KiB - 256 MiB], [ring], [ll128], [8], [20],
    [512 MiB - 1 GiB], [ring], [simple], [8], [17])],
)


== 16 Nodes:

#grid(columns: 2, column-gutter: 1em, row-gutter: 1.2em,
  [*Allreduce* (NCCL's choice)
  #tbl(columns: 5, align: (left, left, left, right, right),
    [size], [algo], [proto], [channels], [warps],
    [8 B - 4 KiB], [tree], [ll], [1], [20],
    [8 KiB], [tree], [ll], [2], [20],
    [16 KiB], [tree], [ll], [4], [20],
    [32 KiB - 128 KiB], [tree], [ll], [8], [20],
    [256 KiB - 16 MiB], [tree], [ll128], [8], [20],
    [32 MiB - 512 MiB], [ring], [ll128], [8], [20],
    [1 GiB], [ring], [simple], [8], [17])],
)

#grid(columns: 2, column-gutter: 1em, row-gutter: 1.2em,
  [*Broadcast* (NCCL's choice)
  #tbl(columns: 5, align: (left, left, left, right, right),
    [size], [algo], [proto], [channels], [warps],
    [8 B - 32 KiB], [ring], [ll], [1], [3],
    [64 KiB], [ring], [ll], [1], [4],
    [128 KiB], [ring], [ll], [1], [8],
    [256 KiB], [ring], [ll], [1], [16],
    [512 KiB], [ring], [ll], [2], [16],
    [1 MiB], [ring], [ll], [4], [16],
    [1536 KiB], [ring], [ll], [6], [16],
    [2 MiB], [ring], [ll], [8], [16],
    [3 MiB - 512 MiB], [ring], [ll128], [8], [20],
    [1 GiB], [ring], [simple], [8], [17])],
)

= Possible explanation for the 32 KiB anomaly <explanation32K>

The theory here is that the number of warps are the problem. Warps determine the number of threads (one warp = 32 threads), and NCCL chooses them depending on the messages size (e.g., 32 KiB gets 3 warps = 96 threads). It also divides the message into chunks, which are processed in sequence. The amount of data that a single thread is responsible for is therefore size_of_chunk / number_of_threads. Both of these grow with message size, but chunk size caps out at 32 KiB, whereas number of threads keeps growing beyond that (it hits the maximum at 256 KiB). So from 32 KiB to 256 KiB, the chunk size stays the same, but the number of threads goes up, and therefore the data _per thread and chunk_ actually goes _down_. This matters, because when doing a hop over the network (not NVLink), NCCL waits for the entire chunk to be done before sending, and having two chunks (which are being processed faster because more threads are available) allows pipelining (see also @explanation4M). 

== Exact numbers for MUSICA, 16 Nodes

The number of warps NCCL uses for a given message size depends on the number of GPUs -- the examples in this section are for 16 MUSICA nodes (64 GPUs).

A _line_ is 16 B (8 B payload + 2 flags). A _chunk_ (= one step of the LL buffer) holds at most 4096 lines, i.e. 32 KiB of payload. Within a channel, the chunks are processed one after another, each by all threads of the block. Each thread handles every `nthreads`-th line of the chunk, one line after the other. Time is the median for run 1914035 (NCCL's choice of algorithm/protocol). The important column is lines/thread/chunk, which shows that the amount of data (lines) per thread and chunk peaks at 32 KiB message size.

#figure({
  tbl(columns: 6, align: (left, right, right, right, right, right),
  [message], [chunks], [lines/chunk], [threads], [lines/thread/chunk], [time (µs)],
  [8 KiB], [1], [1024], [96], [11], [189],
  [16 KiB], [1], [2048], [96], [22], [241],
  [*32 KiB*], [*1*], [*4096*], [*96*], [*43*], [*357*],
  [*64 KiB*], [*2*], [*4096*], [*128*], [*32*], [*322*],
  [128 KiB], [4], [4096], [256], [16], [258],
  [256 KiB], [8], [4096], [512], [8], [256],
)
})

== Verification

To verify this theory, we run an additional experiment that runs the broadcast benchmark (a) normally, (b) with the number of threads pinned to 512 and (c) with the number of threads pinned to 128. The expectation here is that (b) should outperform the default, until the default hits 512 threads itself, at which point they should match and (c) should outperform the default until it hits 128, where they should match, and after that it should underperform.

The results of the experiment meet this expectation: The 128-thread variant matches the default at 64 KiB and 96 KiB, being slightly better before and worse afterwards, and the 512-variant significantly outperforms the default until 256 KiB, at which point the default also uses 512 threads. From 512 KiB, the default starts to outperform both variants - this is because it starts utilizing more than one channel, which the pinned versions don't do (pinning the thread count requires also pinning the channel count).

#figure(image("/results/ll_threads/1915660/pinned_vs_default.pdf", width: 40%))

Pinning the thread count to 512 resolves the anomaly that a smaller messages size has higher latency. However, even with this fix in place, LL128 can still outperform LL (albeit by a much smaller margin). This may have to do with the way that NCCL calculates channel count for these protocols: For both of them, it has a threshold under which it runs with fewer channels than it has available, but for LL this threshold increases with the number of ranks, while for LL128 it does not. This leads to a situation where for 16 MUSICA nodes (64 GPUs/ranks), NCCL sticks with just one channel for LL up until messages size 256 KiB, whereas LL128 already utilizes two at 32 KiB (and the full 8 at 128 KiB). This explanation is untested, though.

= Possible explanation for 4 MiB <explanation4M>

At this size, NCCL picks LL128 as the protocol. Its chunks are fairly large (around 562 KiB), which leads to a similar kind of problem: When moving across the network (not NVLink), NCCL always sends an entire chunk of data at once. When a channel only has one chunk (which for 8 channels is the case for up to 8 x 562 KiB ≈ 4.4 MiB), this means that the section of data the channel is responsible for moves through the ring as one big block (at least for the network hops -- intra-node communication doesn't have this problem). At any point in time, data is moving from just one specific node to the next along the ring, and the other nodes are essentially idling (as far as this one channel is concerned).

Beyond 4 MiB, the number of chunks per channel starts going up, and this enables a form of pipelining: When a channel's data is cut into multiple chunks, these can make their way along the ring independently, so the first chunk might be already at the last node, while the final chunk is just now leaving the root. Because this only applies to network hops, a configuration of 8 nodes only requires 7 chunks to keep the entire channel busy (even if it has more than 8 GPUs).

This behavior explains why the broadcast time rises pretty sharply up until 4 MiB, but then plateaus for the next few larger message sizes: The gains from being able to pipeline compensate for the larger amount of data.

Below are the exact numbers for the relevant portion of the benchmark -- times are medians for Ring/LL128 (LL in parentheses), runs 1912206 (8 nodes) and 1914035 (16 nodes):

#figure({
tbl(columns: 5, align: (left, right, right, right, right),
  [message], [per channel], [chunks/channel], [8 nodes (µs)], [16 nodes (µs)],
  [512 KiB], [64 KiB], [1], [150 (142)], [284 (261)],
  [1 MiB], [128 KiB], [1], [196 (179)], [372 (288)],
  [2 MiB], [256 KiB], [1],  [292 (236)], [558 (355)],
  [*4 MiB*], [*512 KiB*], [*1*],  [*485 (338)*], [*898 (470)*],
  [*8 MiB*], [*1 MiB*], [*2*],  [*551 (540)*], [*897 (685)*],
  [16 MiB], [2 MiB], [4],  [661 (930)], [1041 (1119)],
)})

This diagram illustrates the difference between having one or two chunks -- note the two-chunk version's ability to pipeline:

// Pipelining of one channel across the network hops (time runs left to right, 1 column = half a chunk-sized transfer)
#let hop-diagram(title, bars, ncols: 6) = {
  let c0 = rgb("#4c78a8")
  let c1 = rgb("#f58518")
  let hops = ("N1 → N2", "N2 → N3", "N3 → N4")
  let cells = ()
  for (y, h) in hops.enumerate() {
    cells.push(table.cell(x: 0, y: y, stroke: none, align(right + horizon, h)))
  }
  for (y, x, len, label) in bars {
    cells.push(table.cell(x: x + 1, y: y, colspan: len, fill: if label == "c0" { c0 } else { c1 },
      align(center + horizon, text(fill: white, weight: "bold", label))))
  }
  block(breakable: false, {
    text(weight: "bold", title)
    v(-0.4em)
    table(columns: (auto,) + (1fr,) * ncols, rows: 1.6em, inset: 2pt,
      stroke: (x, y) => if x > 0 { (left: 0.3pt + luma(200)) }, ..cells)
  })
}

#figure({
  hop-diagram([1 chunk: total ≈ hops × T(whole per-channel share)],
    ((0, 0, 2, "c0"), (1, 2, 2, "c0"), (2, 4, 2, "c0")))
  hop-diagram([2 chunks: total ≈ (hops + 1) × T(one chunk)],
    ((0, 0, 1, "c0"), (0, 1, 1, "c1"), (1, 1, 1, "c0"), (1, 2, 1, "c1"), (2, 2, 1, "c0"), (2, 3, 1, "c1")))
}) <fig-pipelining>

== Verification

It is more difficult to confirm this theory with an experiment, but NCCL has the environment variable `NCCL_LL128_BUFFSIZE`. We can use it to make the chunk used by LL128 smaller, which does not eliminate the effect, but should move it left in the diagram (towards smaller messages sizes). 

TODO: evaluation (experiment is still running)