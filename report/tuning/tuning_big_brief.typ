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

There are two things happening here: First, NCCL gradually increases the number of threads (32 threads = 1 warp) as the message size grows. For example, everything up until 32 KiB gets 3 warps (the minimum), which is 96 threads, but 64 KiB gets 4 warps (= 128 threads). For LL, this keeps going up to a maximum of 16 warps (512 threads). This counteracts the increased cost of sending larger messages, but not completely -- latency still grows, just slower than it would otherwise.

Second, NCCL cuts messages into chunks, which it processes sequentially. The maximum size of a chunk for LL is 32 KiB #footnote[Technically the chunk has 64 KiB, but half of that are flags.], so messages up to 32 KiB fit into one chunk, and everything beyond that needs multiple. As it turns out, using multiple chunks is actually beneficial for performance: When NCCL has to send data over the network (inter-node), it always sends an entire chunk. If there is only one chunk, then all data makes its way through the ring as one big block, and only one node can be active at any given moment. When there are multiple chunks, however, NCCL sends the first one over the network as soon as the root node is done with it, allowing the next node in the ring to start working on it while the root processes the second chunk. This is a form of pipelining, and it can significantly reduce the overall time spent on a broadcast operation. The following diagram illustrates the difference between a message that is processed in one chunk and the same message, but processed in two chunks:


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

What happens at 32 KiB is that both these effects coincide: The number of threads is growing (we only reach the peak number of 512 threads at 256 KiB), and NCCL starts to use multiple chunks and therefore pipelining across nodes. Together, they end up having a larger effect than the increasing message size, and so between 32 KiB and 256 KiB, the latency _decreases_.

== Exact numbers for MUSICA, 16 Nodes

The number of warps NCCL uses for a given message size depends on the number of GPUs -- the examples in this section are for 16 MUSICA nodes (64 GPUs).

A _line_ is 16 B (8 B payload + 2 flags). A _chunk_ (= one step of the LL buffer) holds at most 4096 lines, i.e. 32 KiB of payload. Within a channel, the chunks are processed one after another, each by all threads of the block. Each thread handles every `nthreads`-th line of the chunk, one line after the other. Time is the median for run 1914035 (NCCL's choice of algorithm/protocol).

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

To verify this theory, we would ideally control the number of threads and the number of chunks. The former is much easier to do, so for now, that is what we focus on. To this end, we run an additional experiment that runs the broadcast benchmark (a) normally, (b) with the number of threads pinned to 512 and (c) with the number of threads pinned to 128. The expectation here is that (b) should outperform the default, until the default hits 512 threads itself, at which point they should match and (c) should outperform the default until it hits 128, where they should match, and after that it should underperform.

The results of the experiment meet this expectation: The 128-thread variant matches the default at 64 KiB and 96 KiB, being slightly better before and worse afterwards, and the 512-variant significantly outperforms the default until 256 KiB, at which point the default also uses 512 threads. From 512 KiB, the default starts to outperform both variants - this is because it starts utilizing more than one channel, which the pinned versions don't do (pinning the thread count requires also pinning the channel count).

#figure(image("/results/ll_threads/1915660/pinned_vs_default.pdf", width: 40%))

Pinning the thread count to 512 partially resolves the anomaly that a smaller messages (32 KiB) has higher latency than a larger one (64 KiB). What remains is the pipelining effect when using multiple chunks, which can be seen in the figure above for both version with pinned number of threads: The latency steadily rises up to 32 KiB, at which it suddenly plateaus (there is a noticeable kink in the line). This behavior is exactly in line with what we would expect, since this is the point where NCCL starts using multiple chunks.

Ideally, we would test this as well, by configuring LL to use a larger chunk with the `NCCL_LL_BUFFSIZE` parameter. This remains an open point for this scenario, but the idea is taken up again for the other anomaly (see @explanation4M).


One more thing is worth noting here: For these small message sizes, NCCL only uses one channel. For larger messages, using multiple channels adds an extra level of parallelism to the operation. The way NCCL calculates channel count differs between LL and LL128: For both of them, it has a threshold under which it runs with fewer channels than it has available, but for LL this threshold increases with the number of ranks, while for LL128 it does not. This leads to a situation where for 16 MUSICA nodes (64 GPUs/ranks), NCCL sticks with just one channel for LL up until messages size 256 KiB, whereas LL128 already utilizes two at 32 KiB (and the full 8 at 128 KiB). This can theoretically lead to LL128 having better performance when running with many ranks, purely because it can utilize more channels -- this is untested though.

= Possible explanation for 4 MiB <explanation4M>

The explanation here is essentially the pipelining of chunks again, only for the LL128 protocol. Its chunks are much larger (around 562 KiB), and at this point we are also utilizing all 8 available channels #footnote[For MUSICA, that is. Available channels may vary from system to system.]. Because of this, messages up to 8 x 562 KiB ≈ 4.4 MiB only utilize one chunk (per channel). Beyond 4 MiB, the number of chunks per channel starts going up, which enables the pipelining, and the effect is the same as for 32 KiB on LL: The growth in latency suddenly plateaus, with a noticeable kink in the plot.

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


== Verification

It is more difficult to confirm this theory with an experiment, but NCCL has the environment variable `NCCL_LL128_BUFFSIZE`. We can use it to make the chunk used by LL128 smaller, which does not eliminate the effect, but should move it left in the plot (towards smaller messages sizes). 

The result of the experiment shows exactly this behavior: With a buffer half the size, the number of chunks starts going up at 2 MiB instead of 4 MiB, which is exactly where that variant starts to plateau in the graph. The same thing is true for a buffer a quarter the size at 1 MiB (although the effect is less pronounced there).

Interestingly, reducing the buffer to a quarter of its original size is, in this case at least, strictly better in terms of performance for almost all message sizes where NCCL would by default choose LL128 (3 MiB - 512 MiB). It is also the only configuration that is not outperformed by LL at around 4 MiB.

#figure(image("/results/ll128_chunks/1917408/pinned_vs_default.pdf", width: 40%))
