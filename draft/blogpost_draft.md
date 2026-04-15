# Achieving Up to 67% Cost Savings with Prefill-Decode Disaggregation Using Ray + vLLM on AMD MI325X


In LLM serving, the optimization objective is deceptively simple: given a set of latency SLA targets -- time to first token (TTFT), time per output token (TPOT), end-to-end latency (E2E) -- maximize the queries per second (QPS) you can sustain. Higher QPS on the same hardware means lower cost per token. Whether your bottleneck is TTFT or TPOT depends on the shape of your workload: the input/output length ratio, KV cache hit rates, and multi-turn conversation patterns all shift the pressure between the prefill and decode phases of inference.

One of the most powerful levers for breaking through the throughput ceiling is **Prefill-Decode (PD) disaggregation**. Instead of running both phases on the same GPUs -- where they compete for compute, memory bandwidth, and scheduling budget -- PD separates them onto dedicated hardware. Prefill nodes handle prompt processing. Decode nodes handle token generation. By eliminating mutual interference, each phase runs closer to its theoretical throughput, and the system as a whole serves more requests under the same SLA constraints.

PD adds operational complexity: KV cache must be transferred across nodes and the prefill-to-decode ratio must be tuned per workload. We show results where under the same GPU budget and SLA, Prefill-Decode disaggregation on Ray + vLLM can serve **1.3x to 2.3x more QPS** than aggregated serving -- depending on the workload (up to **67% compute cost reduction**) and also results where it does not help -- so you can make the right decision for your workload.


![PD vs Aggregated: Max Sustainable QPS Under SLA](../figures/fig_1_hero_bar.png)
*PD vs Aggregated max sustainable QPS under SLA across 5 workload scenarios (same GPU count). Validated on Qwen3-235B and DeepSeek-V3 on AMD MI325X.*

We tested two large MoE models across a range of workloads -- varying input/output lengths, KV cache hit rates, and P:D ratios -- to find where PD saves cost and where it doesn't. This post walks through the core intuition, the AMD-specific stack (RIXL for KV transfer), how to set it up with Ray Serve, and when to use aggregated instead. For a managed solution you can use [Anyscale]() + [Digital ocean]() for bringing similar cost savings to your workloads.

---

## Core Intuition -- Why PD Works (and When It Doesn't)

This section covers the five key insights you need to reason about PD for any workload. Each insight includes data from our experiments, plus clear guidance on when PD loses.

### Insight 1: PD does NOT make prefill faster -- it can actually hurt TTFT

The most common misconception about PD is that it speeds up everything. It does not. On the metric that matters most for interactive responsiveness -- time to first token -- PD is consistently slower than aggregated serving on the same GPU footprint.

**Why aggregated TTFT is already good.** In vLLM's scheduler, there is no separate "prefill phase" or "decode phase." The scheduler runs all currently-active requests -- both prefill and decode -- before admitting new requests from the waiting queue. Chunked prefill is enabled by default for all decoder-only models: long prompts are split into chunks sized by `max_num_batched_tokens` (defaults to 8192). Each chunk runs as one scheduler iteration. Critically, decode steps consume trivially little budget per iteration. A batch of 128 concurrent decode requests uses at most 128 tokens out of the 8192-token budget, leaving the vast majority of each iteration available for prefill tokens. TTFT is therefore dominated by the raw compute time of the prefill forward pass (attention + MoE routing), not by contention with decode.


![Aggregated scheduler timeline](../figures/insight_1_agg_scheduler.png)
*In aggregated serving, decode tokens consume only ~1.6% of the scheduler's token budget per iteration — TTFT is dominated by prefill compute, not decode contention.*

**What PD changes.** PD adds a KV cache transfer step after prefill completes. The prefill node sends KV data over the network (RDMA/RoCE) to the decode node. This transfer has inherent overhead that depends on model architecture, KV cache size, and network conditions. Under high load and kv-cache pressure, prefill nodes can also queue up, adding queuing delay on top of transfer overhead.


![PD KV transfer overhead](../figures/insight_1_pd_kv_transfer.png)
*In PD serving, the KV cache transfer step between prefill and decode nodes adds overhead that inflates TTFT.*


**The net effect.** On the same GPU footprint, aggregated consistently achieves equal or lower TTFT than PD.


![TTFT vs QPS for PD and Agg](../figures/fig_3_ttft_vs_qps.png)
*Figure 3: TTFT vs QPS — Agg TTFT stays flat while PD TTFT rises under load.*

Could you solve the TTFT gap by adding more prefill GPUs? Generally, no. TTFT improves sub-linearly with additional prefill capacity, so the added GPU cost rarely justifies the marginal TTFT improvement compared to what aggregated delivers natively.

#### When this means PD loses -- strictly TTFT-limited SLAs

If your SLA is measured purely on time-to-first-token (e.g., interactive search, auto-complete), aggregated will consistently beat PD. As Figure 3 shows, PD's TTFT baseline on DeepSeek-V3 is ~330ms (due to KV transfer overhead), while Agg stays at ~260ms across all QPS levels:

- Under a **TTFT < 300ms** SLA, PD **cannot serve any traffic** (baseline TTFT exceeds the target), while Agg sustains **7.0+ QPS**.
- Under a **TTFT < 500ms** SLA, Agg sustains **7.0+ QPS** vs PD's **5.0 QPS** -- Agg wins by at least 1.4x.

Agg's advantage here is structural: no KV transfer step, and prefill load is naturally distributed across replicas. If you need *both* fast TTFT and fast TPOT, consider accepting a slightly relaxed TTFT target -- even a small relaxation can unlock major TPOT and E2E improvements through PD.

**Bottom line:** If your SLA is strictly TTFT-limited, aggregated is the simpler and better choice.

---

### Insight 2: PD's real win is flat, stable TPOT under load

This is the core mechanism behind PD's value. In aggregated serving, prefill and decode share the same GPU. Each scheduler iteration that includes prefill tokens is compute-heavier than a pure-decode iteration. As QPS rises, more prefill work stacks up, and decode tokens wait longer. We call this the **"TPOT death spiral"** -- TPOT degrades linearly (or worse) with increasing QPS, because every new prefill request steals compute from all in-flight decode steps.

PD eliminates this entirely. Decode runs on dedicated GPUs that never see a prefill token. TPOT stays nearly flat regardless of how much prefill work is happening on other nodes.

![TPOT vs QPS — the death spiral](../figures/fig_4_tpot_vs_qps.png)
*Figure 4: TPOT vs QPS — Agg TPOT rises steeply ("death spiral") while PD stays flat. The most important chart in this post.*

**Qwen3-235B (TP8, 24 GPU, ISL=16K, OSL=1K, 0% HR — 2P1D vs 3Agg):**

| QPS | Agg TPOT | PD TPOT | PD Advantage |
|-----|----------|---------|--------------|
| 1.0 | 23.1ms | 19.5ms | ~1.2x |
| 2.0 | 25.5ms | 22.8ms | ~1.1x |
| 3.0 | 27.6ms | 24.2ms | **1.1x** |
| 4.0 | 29.9ms | 24.8ms | **1.2x** |

**DeepSeek-V3 (TP8, ISL=5.4K, OSL=140, 30% hit rate):**

| QPS | Agg TPOT | PD TPOT | PD Advantage |
|-----|----------|---------|--------------|
| 3 | 23.7ms | 22.6ms | ~1.0x |
| 5 | 35.1ms | 26.1ms | **1.3x** |
| 7 | 50.6ms | 27.3ms | **1.9x** |

The pattern is striking: PD's TPOT stays well-controlled (~22--27ms across the entire QPS range for DeepSeek-V3), while Agg's TPOT degrades linearly with load.

---

### Insight 3: TPOT savings compound over output sequence length

The per-token TPOT advantage may look modest in isolation -- 10-20ms per token. But it multiplies across every output token. The mechanism is simple arithmetic:

> **Total TPOT savings = (Agg TPOT - PD TPOT) x output_length**

This compounding is the reason PD wins on E2E latency even though it loses on TTFT. The longer the output, the more the savings accumulate.

**Concrete example (Qwen3-235B TP8, 24 GPU, QPS=4):**

- TTFT penalty from PD: ~130ms (PD is slower to produce the first token)
- TPOT savings per token: 5.1ms (Agg 29.9ms vs PD 24.8ms)
- At **OSL=1,024**: 5.1ms x 1,024 = **5.2 seconds of total TPOT savings** vs 0.13s TTFT penalty --> clear E2E win
- At **OSL=140**: 5.1ms x 140 = **0.7 seconds of savings** -- still wins, but the margin is smaller

The E2E win percentage at a given QPS depends on many factors beyond just output length -- the P:D ratio, the specific QPS point, the model architecture. But the core mechanism holds universally: TPOT delta compounds over output tokens. Longer output means bigger PD win.

**Cross-validation across workloads:**
- Qwen3-235B at OSL=140, 80% cache hit rate: PD wins E2E by only **5%** at QPS=4 -- barely worth the complexity.
- Qwen3-235B at OSL=1024, 80% cache hit rate: PD wins E2E by **24%** at the same conditions.
- DeepSeek-V3 at OSL=140: TPOT advantage is 1.3x at QPS=5, but short output limits the E2E gain. At OSL=1K, the 9.0ms per-token delta (at QPS=5) would compound to ~9.0 seconds of total savings -- a significant E2E win.

#### When this means PD loses -- short output sequences

When output is short, the per-token TPOT savings do not accumulate enough to overcome the TTFT penalty. For short-output workloads -- classification, entity extraction, short QA -- use aggregated serving. The operational simplicity outweighs the marginal gains.

**Takeaway:** The TPOT delta per token is the unit of PD's value. How that value translates to E2E improvement depends on your workload -- benchmark it. But directionally: longer output = bigger PD win.

---

### Insight 4: The optimal P:D ratio depends on your workload

This is the most practical insight for practitioners. The P:D ratio determines how GPU resources are split between prefill and decode. Getting it wrong can make PD *worse* than aggregated.

**Key findings across workloads:**

| Workload | Cache Hit Rate | Bottleneck | Optimal Ratio | Evidence |
|----------|---------------|-----------|--------------|----------|
| Long input, short output (ISL=16K, OSL=1K) | 0% | Prefill throughput | **2P:1D** | 2P1D TP8 beats 3x Agg TP8 at 24 GPUs |
| Long input, long output (ISL=16K, OSL=4K) | 0% | Decode throughput | **1P:3D** | Single-decode PD loses to Agg by 6--40%; 1P3D flips it to 14% win |
| Multi-turn with high cache reuse | 80% | Decode throughput | **1P:2D** | Cached prefill is cheap, shift GPUs to decode |
| Multi-turn with moderate cache reuse | 30--60% | Mixed | **1P:1D** to **1P:2D** | DeepSeek study: 1P1D achieves 1.40--2.33x capacity advantage |

**Why 2P:1D for long-input short-output.** At ISL=16K, OSL=1K with 0% cache hit rate, the prefill phase is compute-heavy and a single prefill replica saturates quickly. From our Qwen3-235B experiments:

- **1P1D (16 GPU):** At QPS=3, TTFT explodes to 7,021ms -- single prefill cannot keep up.
- **2P1D (24 GPU):** At QPS=3, TTFT is only 1,187ms -- the second prefill absorbs the load.
- TPOT is similar between the two (28.8ms vs 33.9ms) because both have a single decode node. The bottleneck shifted from prefill to decode at higher QPS.

The second prefill replica buys 75% more QPS capacity. That is the value of matching the ratio to the bottleneck.

**Rule of thumb:** The marginal GPU should go to wherever the bottleneck is. High cache hit rates make prefill cheap -- allocate more to decode. Low cache hit rates with long inputs -- allocate more to prefill.

#### When this means PD loses -- wrong P:D ratio is worse than aggregated

The most common PD pitfall: deploying with a ratio that does not match the workload. This can make PD strictly worse than aggregated on every metric.

![Wrong P:D ratio impact](../figures/fig_6_wrong_pd_ratio.png)
*Figure 6: Wrong P:D ratio can be worse than aggregated — 3P:1D is 67% worse, while 1P:3D is 14% better.*

From our Qwen3-235B experiments (ISL=16K, OSL=4K at QPS=1.5):

| Config | GPUs | E2E | vs 4x Agg (32 GPU) |
|--------|------|-----|---------------------|
| 3P1D TP8 | 32 | 227.0s | **67% worse** |
| 2P1D TP8 | 24 | 205.3s | **51% worse** |
| 1P1D TP8 | 16 | 192.2s | **41% worse** |
| **1P3D TP8** | **32** | **117.0s** | **14% better** |
| 4x Agg TP8 | 32 | 135.8s | baseline |

All single-decode PD configs *lose* to Agg -- badly. Only the decode-heavy 1P3D configuration wins. The difference between worst PD (3P1D) and best PD (1P3D) is **94%**. Always benchmark your specific workload before deploying PD. Start with a 1:1 ratio, then adjust based on whether TTFT or TPOT hits the SLA first.

---

### Insight 5: PD delivers the same performance with fewer GPUs (or more performance with the same GPUs)

The primary perspective: **same GPU footprint, higher QPS capacity under SLA.** This means you serve more traffic on the same hardware.

**More QPS on the same GPUs (DeepSeek-V3, TPOT < 30ms SLA, 16 GPUs):**

| Mode | Max Sustainable QPS (30% HR) | Max Sustainable QPS (60% HR) |
|------|------------------------------|------------------------------|
| PD (1P1D) | 7.0 | 7.0 |
| Agg (2x) | 3.0 | 5.0 |
| **PD Advantage** | **2.33x** | **1.40x** |

The alternative perspective: **same QPS target, fewer GPUs required.** This directly reduces cost.

**Same QPS, fewer GPUs (Qwen3-235B TP8, ISL=16K, OSL=1K):**

| QPS | PD 2P1D (24 GPU) E2E | Agg 4x (32 GPU) E2E | Verdict |
|-----|----------------------|----------------------|---------|
| 2.0 | 29.0s | 29.5s | Same E2E, **PD uses 25% fewer GPUs** |
| 3.0 | 35.9s | 40.0s | PD 10% better E2E **and** 25% fewer GPUs |
| 4.0 | 47.3s | 51.8s | PD 9% better E2E **and** 25% fewer GPUs |

**Translated to cost (DeepSeek-V3, TPOT < 30ms SLA):**

| Target QPS | PD node-hours/day | Agg node-hours/day | Savings |
|-----------|-------------------|-------------------|---------|
| 3 | 48 | 48 | 0% |
| 5 | 48 | 96 | **50%** |
| 7 | 48 | 144 | **67%** |

These cost savings are real and operational -- fewer nodes provisioned, fewer node-hours billed. Under a TPOT SLA, PD is always equal or cheaper than Agg because PD sustains 7.0 QPS vs Agg's 3.0 QPS per 2-node set (2.3x more efficient). At high QPS targets where Agg needs multiple 2-node sets (each capped at 3.0 QPS), the savings compound -- up to 67% at QPS=7.

---

### Decision Framework

The right deployment mode depends on your workload. There are no universal thresholds -- the breakpoints shift with model, hardware, and traffic patterns. The framework below gives directional guidance; **always benchmark your specific workload** before committing to PD in production.

![PD vs Aggregated decision framework](../figures/fig_7_decision_flowchart.png)
*Figure 7: Decision framework — when to use PD vs Aggregated.*

**Step 1: What is your primary SLA constraint?**
- **TPOT or E2E** --> PD likely wins. Continue to Step 2.
- **Strictly TTFT** --> Aggregated likely wins. Stop here.

**Step 2: Is your output long enough for TPOT savings to compound?**
- The longer the output, the more PD's per-token advantage accumulates.
- **Short output** (classification, extraction) --> Agg is simpler and nearly as good.
- **Moderate to long output** (chat, generation, reasoning) --> PD advantage grows. Continue.

**Step 3: What is your KV cache hit rate?**
- **Lower hit rates** --> Larger PD advantage, favor prefill-heavy ratios.
- **Higher hit rates** --> Smaller PD advantage (but still positive), favor decode-heavy ratios.
- The exact crossover is workload-specific. Use Ray Serve to deploy both configs side-by-side and benchmark.

---

## What's Special About AMD -- RIXL and the KV Transfer Stack

PD disaggregation requires high-bandwidth KV cache transfer between prefill and decode nodes. On NVIDIA hardware, this is handled by **NIXL** (NVIDIA Interconnect eXchange Library) over NVLink, InfiniBand, or EFA. On AMD, we use **RIXL** (ROCm Interconnect eXchange Library) -- a plug-and-play replacement for NIXL that uses UCX transport over RDMA/RoCE InfiniBand.

The key point: RIXL exposes the same `NixlConnector` interface in vLLM. Zero code changes are needed in the serving layer. If your vLLM config says `kv_connector: NixlConnector`, it works on both NVIDIA (via NIXL) and AMD (via RIXL). Cross-node transfer bandwidth with RIXL is 10--12 GB/s, comparable to NVIDIA InfiniBand.

### Container and Software Stack

Our container image is built from `Dockerfile.v0.18.dev` with the following components:

| Component | Version / Source |
|-----------|-----------------|
| Base image | `anyscale/ray:nightly-py312-cu128` (or `rayproject/ray:nightly-py312-cu128` for OSS) |
| ROCm | 7.0 (`rocm/dev-ubuntu-22.04:7.0-complete` for build stages) |
| vLLM | 0.18.0 via `wheels.vllm.ai/rocm/0.18.0/rocm700` prebuilt wheels |
| RIXL | Built from source (`ROCm/RIXL` commit `f33a5599`) |
| UCX | Built from source (`ROCm/ucx` commit `da3fac2a`) with `--with-rocm`, `--with-verbs`, `--enable-mt` |
| ROCm Triton | Built from source (`ROCm/triton` commit `f9e5bf54`) |
| Python | 3.12 |

### ROCm Runtime Optimizations

Key environment variables enabled at runtime for optimal AMD performance:

```bash
VLLM_ROCM_USE_AITER=1                        # AMD AI Tensor Engine Runtime
VLLM_ROCM_USE_AITER_MOE=1                    # Optimized MoE kernels
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4     # Fast all-reduce with INT4 quantization
VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1  # BF16-to-FP16 cast for quick-reduce
```

### RDMA Transport -- A Critical Operational Note

**Network transport quality is everything.** With proper UCX/RDMA configuration, cross-node KV transfer performs comparably to intra-node. With TCP fallback, throughput degrades catastrophically -- we observed up to **19x degradation** in testing. Always validate your RDMA transport layer before benchmarking PD.

The UCX transport configuration in our deployments:

```bash
UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
```

This configures 8x Mellanox ConnectX interfaces for RoCE fabric, using reliable connected (RC) transport with shared memory and ROCm GPU direct paths. Hardware tested: AMD MI325X with 288GB HBM3e, 8 GPUs per node.

---

## PD in Ray Serve -- It's a Config Change

The entire PD deployment is defined in a single YAML file. Switching between aggregated and PD serving requires changing the config -- not the application code, not the model, not the infrastructure.

### The YAML Config

Here is the actual config used for our Qwen3-235B-A22B experiments (abbreviated for clarity):

```yaml
applications:
  - name: pd-qwen235b
    import_path: pd_app:build_pd_openai_app
    route_prefix: /

    args:
      prefill_config:
        model_loading_config:
          model_id: test-model
          model_source: Qwen/Qwen3-235B-A22B-Instruct-2507
        engine_kwargs:
          tensor_parallel_size: 4
          enable_expert_parallel: true
          enable_prefix_caching: true
          max_model_len: 16384
          gpu_memory_utilization: 0.7
          max_num_batched_tokens: 32768    # Large budget for prompt processing
          kv_cache_dtype: fp8
          quantization: fp8
          enforce_eager: true
          kv_transfer_config:
            kv_connector: NixlConnector    # Same connector for NVIDIA (NIXL) or AMD (RIXL)
            kv_role: kv_both

      decode_config:
        model_loading_config:
          model_id: test-model
          model_source: Qwen/Qwen3-235B-A22B-Instruct-2507
        engine_kwargs:
          tensor_parallel_size: 4
          enable_expert_parallel: true
          enable_prefix_caching: true
          max_model_len: 16384
          gpu_memory_utilization: 0.7
          max_num_batched_tokens: 32768
          kv_cache_dtype: fp8
          quantization: fp8
          compilation_config:
            cudagraph_mode: FULL_DECODE_ONLY  # Decode-only CUDA graph optimization
          kv_transfer_config:
            kv_connector: NixlConnector
            kv_role: kv_both
```

The key design choices in this config:

- **`max_num_batched_tokens: 32768`** on the prefill side gives a large budget to process long prompts efficiently. The decode side can use a smaller budget optimized for pure token generation.
- **`kv_connector: NixlConnector`** with **`kv_role: kv_both`** enables bidirectional KV cache transfer. The connector name is the same regardless of whether NIXL (NVIDIA) or RIXL (AMD) is the underlying transport.
- **`cudagraph_mode: FULL_DECODE_ONLY`** on the decode side enables CUDA/HIP graph capture for decode-only batches, reducing kernel launch overhead.
- **`enable_prefix_caching: true`** on both sides enables KV cache reuse for repeated prefixes, which is critical for multi-turn workloads.

### The App Builder -- One Function Call

The application entry point is a single function:

```python
from pd_app import build_pd_openai_app

app = build_pd_openai_app(prefill_config=..., decode_config=...)
```

This returns a fully wired Ray Serve application with an OpenAI-compatible API. The `build_pd_openai_app` function handles replica creation, ingress routing, and KV transfer coordination. You get `/v1/chat/completions` and `/v1/completions` endpoints out of the box.

### Session-Aware Routing for Cache Affinity

For multi-turn workloads, KV cache reuse across turns is critical for performance. Our deployment uses session-aware routing: an `X-Session-Id` header in the request routes multi-turn requests to the same decode replica, maximizing KV cache hits.

This is currently implemented via `@serve.multiplexed` in our custom app code (`SessionAwareIngress` and `SessionAwareLLMServer` classes in the config above). Native Ray Serve support for session-affinity routing is in progress -- see [RFC link TBD] for the design. Once available, session routing will be a built-in config option rather than a custom implementation.

![PD architecture diagram](../figures/fig_8_pd_architecture.png)
*Figure 8: Ray Serve PD topology — Ingress routes to Decode nodes, which forward to Prefill nodes. KV cache transferred via RIXL over RDMA.*

### Coordinated Autoscaling

The config also includes a coordinated autoscaling policy (`CoordinatedPDPolicy`) that scales prefill and decode replicas together based on the configured ratio:

```yaml
autoscaling_policy:
  policy_function: pd_app:CoordinatedPDPolicy
  policy_kwargs:
    ingress_ratio: 4
    prefill_ratio: 1
    decode_ratio: 1
    target_qps_per_prefill: 10.0
    target_qps_per_decode: 10.0
```

This ensures the P:D ratio stays fixed as the system scales up or down with traffic, avoiding the wrong-ratio trap described in Insight 4.

---

## How to Reproduce

Everything needed to reproduce these results is consolidated in a single repository: Dockerfile, serve configs, and benchmark scripts.

### Cluster Requirements

- **Minimum:** 2 nodes x 8 AMD MI325X GPUs (16 GPUs total) for a 1P1D configuration
- **Recommended:** 4 nodes x 8 GPUs (32 GPUs) for exploring different P:D ratios
- **Networking:** RDMA-capable RoCE fabric (8x Mellanox ConnectX interfaces per node). Validate RDMA connectivity before benchmarking -- TCP fallback will invalidate results.
- **Container:** Build from `Dockerfile.v0.18.dev` in the repo, or use the pre-built image: `kouroshhahkha/rocm-vllm-ray:main-nightly-v6`

You can deploy on Kubernetes via KubeRay, or on Anyscale for a managed experience that handles cluster provisioning and image management.

### Deploy

Deploy PD or aggregated with a single command:

```bash
# PD disaggregated
ray serve deploy serve_configs/pd/qwen235b.yaml

# Aggregated baseline
ray serve deploy serve_configs/agg/qwen235b.yaml
```

On Anyscale:

```bash
anyscale service deploy -f serve_configs/pd/qwen235b.yaml
```

### Run Benchmarks

We use the Ray LLM in-tree benchmark CLI in interactive mode, which lets you adjust workload parameters and QPS on the fly without restarting:

```bash
python -m ray.llm._internal.serve.benchmark -i
```

In interactive mode:

```
> workload --isl 5400 --osl 140 --hit-rate 0.3 --num-turns 1
> rate 5
> status          # wait for inflight requests to stabilize
> measure 100     # collect 100 request measurements
> save results/pd_5400_140_hr30_qps5.json
```

Typical workflow for a QPS sweep:

1. Set the workload parameters once.
2. Start at low QPS (e.g., `rate 1`), wait for steady state, measure.
3. Increase QPS incrementally (`rate 2`, `rate 3`, ...), measuring at each level.
4. Save results at each QPS point for later comparison.

### Compare Results

Compare PD vs Agg results across QPS levels by loading the saved JSON files. The key metrics to compare:
- **TTFT** (p50, p99) -- PD will generally be higher
- **TPOT** (p50, p99) -- PD will generally be lower and flatter
- **E2E latency** (p50, p99) -- depends on output length
- **Max sustainable QPS** under your SLA target

### Full Reproduction Repository

All artifacts are available at: [Link TBD]

For a quick start, Anyscale + Digital Ocean provides a managed environment where you can skip cluster setup and go directly to deploying the serve config.

---

## Conclusion

### Key Takeaways

1. **PD disaggregation on Ray + vLLM delivers up to 67% cost savings** for TPOT- and E2E-sensitive workloads on AMD MI325X GPUs. Under a TPOT < 30ms SLA, PD sustains up to 2.3x more QPS than aggregated on the same hardware.

2. **The savings are workload-dependent.** Match your P:D ratio to your workload's ISL/OSL ratio and cache hit rate. Long inputs with short outputs need more prefill capacity (2P:1D). Long outputs need more decode capacity (1P:3D). The wrong ratio can make PD 67% worse than aggregated.

3. **AMD MI325X is a first-class platform for PD** via RIXL -- a plug-and-play replacement for NVIDIA's NIXL. Zero code changes in the serving layer. Same `NixlConnector` config, same performance characteristics.

4. **Ray Serve makes PD a config change, not a rewrite.** Same API, same deployment flow, same autoscaling. Switch between aggregated and PD by swapping a YAML file. Session-aware routing for multi-turn cache affinity is supported today and will become a built-in feature.

5. **Know when NOT to use PD.** Strictly TTFT-limited SLAs, short outputs, or a mismatched P:D ratio can make PD worse. Always benchmark your specific workload. The decision framework in this post gives you the structure to evaluate quickly.

### Call to Action

- **Try it yourself.** Clone the reproduction repository [Link TBD], deploy on your AMD MI325X cluster, and benchmark with your own workload parameters.
- **Managed experience.** Anyscale + Digital Ocean provides a turnkey environment for PD disaggregation -- no cluster setup required.
- **Join the community.** Questions, results, and feedback are welcome in the Ray community. If you find workloads where PD behaves differently than our framework predicts, we want to hear about it.
