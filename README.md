# Prefill-Decode Disaggregation with Ray + vLLM on AMD MI325X

Reproduction repository for the blog post: *Achieving Up to 67% Cost Savings with Prefill-Decode Disaggregation Using Ray + vLLM on AMD MI325X*.

## Hardware Requirements

- **GPUs:** AMD Instinct MI325X (288 GB HBM3e), 8 per node
- **Minimum:** 2 nodes (16 GPUs) for 1P1D configurations
- **Recommended:** 4 nodes (32 GPUs) for exploring P:D ratios
- **Networking:** RDMA-capable RoCE fabric with 8x Mellanox ConnectX interfaces per node

**Prerequisites:** These instructions assume you already have a Ray cluster running on AMD MI325X nodes (via Anyscale, KubeRay, or bare metal). The serve configs and benchmarks run on top of an existing cluster.

## Container Image

The same `Dockerfile` produces images for either OSS Ray (KubeRay / bare
metal) or Anyscale services. The only difference is the **base image**,
which is exposed as a build argument:

| Target | `BASE_IMAGE` |
|--------|--------------|
| OSS Ray (default) | `rayproject/ray:nightly-py312-cu128` |
| Anyscale services | `anyscale/ray:nightly-py312-cu128` |

**Build for OSS Ray (default):**

```bash
docker build --platform linux/amd64 -t pd-vllm-rocm .
```

**Build for Anyscale:**

```bash
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=anyscale/ray:nightly-py312-cu128 \
  -t pd-vllm-rocm .
```

Push the resulting image to a registry your Ray cluster (or Anyscale cloud)
can pull from before deploying.

See [Anyscale base images](https://docs.anyscale.com/reference/base-images)
for the list of supported `anyscale/ray:*` tags.

## Quick Start

### Option A: Run on Anyscale

If you're new to Anyscale, start here:
- [Anyscale getting started](https://docs.anyscale.com/get-started)
- [Deploy Anyscale on Kubernetes](https://docs.anyscale.com/admin/cloud/kubernetes)
- [Anyscale Service API](https://docs.anyscale.com/reference/service-api)

**1. Build and push the image** using the Anyscale `BASE_IMAGE` (see
[Container Image](#container-image) above). Set `IMAGE` to the registry URL
your Anyscale cloud can pull from.

**2. Create a compute config on Anyscale:**

```bash
anyscale compute-config create compute_configs/pd-amd-mi325x-autoscale.yaml --name pd-amd-mi325x
```

**3. Deploy:**

Pass `--cloud`, `--compute-config`, and `--image-uri` at deploy time — the YAML files contain only the application config:

```bash
anyscale service deploy -f serve_configs/pd/qwen235b_2p1d_tp8.yaml \
  --cloud YOUR_CLOUD_NAME \
  --compute-config pd-amd-mi325x \
  --image-uri $IMAGE
```

**4. Benchmark:**

Launch the Ray LLM benchmark CLI in interactive mode against your service URL:

```bash
python -m ray.llm._internal.serve.benchmark -i \
  -u YOUR_SERVICE_URL -m qwen3-235b --tokenizer Qwen/Qwen3-235B-A22B-FP8
```

Interactive mode commands:

```
> workload --isl 16000 --osl 1024 --hit-rate 0.0 --num-turns 1
> rate 3
> status          # wait for inflight to stabilize
> measure 50      # collect 50 measurements
> save results/my_experiment.json
```

### Option B: Run on OSS Ray (KubeRay or bare metal)

**1. Build the image** with the default OSS Ray `BASE_IMAGE` (see
[Container Image](#container-image) above) and push it to a registry your
cluster can pull from.

**2. Start a Ray cluster:**

**With KubeRay** — install the operator, then create a RayCluster or RayService. Use `compute_configs/pd-amd-mi325x-autoscale.yaml` as a reference for the pod spec.

Resources:
- [KubeRay quickstart](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)
- [KubeRay RayService guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html) — deploys Ray Serve apps directly
- [KubeRay Helm chart](https://github.com/ray-project/kuberay/tree/master/helm-chart/kuberay-operator)

Key pod spec requirements for PD on AMD:

```yaml
spec:
  hostIPC: true           # Required for RDMA shared memory
  hostNetwork: true       # Required for RDMA RoCE networking
  securityContext:
    privileged: true
    supplementalGroups: [991]  # AMD GPU device access (/dev/kfd, /dev/dri)
  containers:
  - resources:
      limits:
        amd.com/gpu: 8
        rdma/fabric0: 1   # 8x RoCE fabric interfaces
        rdma/fabric1: 1
        rdma/fabric2: 1
        rdma/fabric3: 1
        rdma/fabric4: 1
        rdma/fabric5: 1
        rdma/fabric6: 1
        rdma/fabric7: 1
```

**With bare metal:**

```bash
# Head node
ray start --head

# Worker nodes (run on each)
ray start --address=HEAD_NODE_IP:6379
```

**3. Deploy:**

```bash
ray serve deploy serve_configs/pd/qwen235b_2p1d_tp8.yaml
```

**4. Benchmark:**

Launch the Ray LLM benchmark CLI in interactive mode against your local cluster:

```bash
python -m ray.llm._internal.serve.benchmark -i \
  -u http://localhost:8000 -m qwen3-235b --tokenizer Qwen/Qwen3-235B-A22B-FP8
```

Interactive mode commands:

```
> workload --isl 16000 --osl 1024 --hit-rate 0.0 --num-turns 1
> rate 3
> status          # wait for inflight to stabilize
> measure 50      # collect 50 measurements
> save results/my_experiment.json
```

## Serve Configs

### PD Disaggregated

| Config | Model | P:D Ratio | GPUs | Use Case |
|--------|-------|-----------|------|----------|
| `pd/qwen235b_1p1d_tp8.yaml` | Qwen3-235B | 1P:1D | 16 | Minimum PD setup |
| `pd/qwen235b_2p1d_tp8.yaml` | Qwen3-235B | 2P:1D | 24 | Long input, short output |
| `pd/qwen235b_1p2d_tp8.yaml` | Qwen3-235B | 1P:2D | 24 | Moderate output |
| `pd/qwen235b_1p2d_tp8_cache.yaml` | Qwen3-235B | 1P:2D | 24 | Multi-turn with prefix caching |
| `pd/qwen235b_1p3d_tp8.yaml` | Qwen3-235B | 1P:3D | 32 | Long output |
| `pd/qwen235b_2p2d_tp8.yaml` | Qwen3-235B | 2P:2D | 32 | Balanced |
| `pd/deepseek_v3_1p1d_tp8.yaml` | DeepSeek-V3 | 1P:1D | 16 | DeepSeek baseline (prefix caching) |

### Aggregated Baselines

| Config | Model | Replicas | GPUs |
|--------|-------|----------|------|
| `agg/qwen235b_2agg_tp8.yaml` | Qwen3-235B | 2 | 16 |
| `agg/qwen235b_3agg_tp8.yaml` | Qwen3-235B | 3 | 24 |
| `agg/qwen235b_3agg_tp8_cache.yaml` | Qwen3-235B | 3 | 24 |
| `agg/qwen235b_4agg_tp8.yaml` | Qwen3-235B | 4 | 32 |
| `agg/deepseek_v3_2agg_tp8.yaml` | DeepSeek-V3 | 2 | 16 |

## Software Stack

| Component | Version |
|-----------|---------|
| vLLM | 0.18.0 (ROCm wheels) |
| Ray | 3.0.0.dev0 (nightly) / 2.55 (release) |
| ROCm | 7.0 |
| RIXL | Built from source (`ROCm/RIXL` commit `f33a5599`) |
| UCX | Built from source (`ROCm/ucx` commit `da3fac2a`) |
| Python | 3.12 |
| Model | Qwen/Qwen3-235B-A22B-FP8 (TP8, dummy weights for benchmarking) |
