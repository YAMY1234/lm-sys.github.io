<!-- Review-only section draft. Integrate into the launch article after its target branch is confirmed. -->

## PLE: A Large Sparse Memory That Does Not Need to Live in HBM

Qwen4-Exp adds **Per-Layer Embeddings (PLE)** to the second decoder layer. PLE is not another attention block, and it is not part of the MTP draft model. Instead, it is a learned, content-addressed memory that augments the four Hyper-Connection (HC) residual branches before the layer's attention mixer.

The checkpoint makes this memory unusually large but very sparse at inference time. With `ngram_size=3` and `heads_per_ngram=8`, PLE hashes both 2-gram and 3-gram contexts into 16 independent embedding heads. Each head returns 160 BF16 values, and their concatenation forms a 2,560-dimensional embedding. Across the 16 prime-sized hash tables, the embedding alone contains about 51.2 billion parameters, or 95.4 GiB in BF16. However, each token reads only 16 rows, totaling 2,560 BF16 values, from that table.

<p align="center">
  <img src="/images/blog/qwen4/qwen4-ple-offload.svg" width="100%">
</p>
<p align="center" style="color:gray; text-align:center;"><em>Figure X. PLE dataflow and SGLang's sparse pinned-host offload path. PLE is configured only at decoder layer 2 in the evaluated checkpoint.</em></p>

### From N-gram memory to an HC residual update

For each token, SGLang retains the two preceding token IDs, respecting EOS boundaries, and hashes the resulting 2-gram and 3-gram contexts into the 16 embedding heads. The concatenated embedding is projected into a key and a shared value. In parallel, the current HC state provides four branch-specific queries. Grouped RMS normalization followed by a query-key dot product produces one gate per HC branch, so the same retrieved value can be injected with a different strength into each branch.

The gated value then passes through a depthwise short convolution with kernel size 4 and dilation 3. Its persistent state spans nine positions. The sum of the gated value and convolution output is added to the four-branch HC state; only then does the ordinary HC mixer produce the 2,560-wide input to the layer's GDN attention block. This ordering matters: PLE enriches the residual branches before attention, rather than bypassing HC or directly replacing the attention input.

### Sparse pinned-host offload

Keeping the full PLE table in HBM is wasteful because almost all of it is idle for any given token. SGLang therefore keeps each rank's existing vocabulary-parallel shard in pinned host memory and uses a Triton UVA gather kernel to read only the selected BF16 rows into a small GPU buffer. It does not copy or all-gather the full table. The gathered rows still follow the original tensor-parallel reduction, so the numerical and sharding semantics are unchanged.

The gather is issued on a dedicated CUDA stream while the preceding decoder layer is running. SGLang reuses preallocated eager or CUDA Graph buffers and waits only when layer 2 consumes the result. Under DP Attention, this storage choice remains orthogonal to PLE sharding: the default global-TP path replicates token IDs across DP ranks and reduces over the global TP group, while the attention-TP mode shards and reduces within each attention-TP group. In both cases, offload changes where the table is stored, not which ranks own its rows.

This is a specialized PLE weight offload, not KV-cache offload or generic layer offload. It is enabled by default only for BF16 Qwen4-Exp on CUDA, where pinned memory and UVA are available; unsupported dtypes and platforms do not silently enter the path.

### Capacity without a throughput tax

On H200 with TP4 and the selected MTP-213 configuration, PLE offload reduced target-model weight residency from 83.91 GiB to 60.45 GiB per GPU. At the same memory fraction, the freed HBM increased the allocated KV capacity from 1,835,584 to 3,277,376 tokens, a 78.54% gain. The matched C1/C2/C4 throughput geometric mean changed by -0.07%, within run-to-run precision; the measured tradeoff was 12.02 seconds of additional one-time model loading.

| H200 TP4 metric | GPU-resident PLE | Pinned-host PLE | Change |
| :--- | ---: | ---: | ---: |
| Target-model weights per GPU | 83.91 GiB | 60.45 GiB | **-23.46 GiB** |
| Allocated KV capacity | 1,835,584 tokens | 3,277,376 tokens | **+78.54%** |
| C1 output throughput | 236.38 tok/s | 240.43 tok/s | +1.72% |
| C2 output throughput | 381.34 tok/s | 383.56 tok/s | +0.58% |
| C4 output throughput | 580.13 tok/s | 565.91 tok/s | -2.45% |
| C1/C2/C4 throughput, geometric mean | - | - | **-0.07%** |
| Target-model load time | 24.92 s | 36.94 s | +12.02 s |

We also optimized the compute surrounding the lookup. A shape-guarded, bitwise-exact decode fast path fuses the N-gram hash, branch gate/value broadcast, and short-convolution state movement while retaining the native depthwise convolution. On GB300 TP4, this reduced the PLE core from roughly 80 launches to 17 or 18 and cut PLE-core time by 69.56% across batch sizes 1, 16, 64, and 256. The corresponding profiler-off serving throughput improved by 0.54% to 2.64%. The end-to-end gain is intentionally smaller because this checkpoint invokes PLE only once, at layer 2, while the rest of the 48-layer model remains unchanged.
