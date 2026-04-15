# Prefill-Decode Disaggregation with Ray + vLLM on AMD MI325X

Reproduction repository for the blog post: *Achieving Up to 67% Cost Savings with Prefill-Decode Disaggregation Using Ray + vLLM on AMD MI325X*.

## Repository Structure

```
serve_configs/
  pd/                       # PD disaggregated configs (1P1D, 2P1D, 1P2D, 1P3D, 2P2D)
  agg/                      # Aggregated baseline configs (2Agg, 3Agg, 4Agg)
compute_configs/            # Anyscale compute config templates for AMD MI325X
Dockerfile                  # Container image: vLLM 0.18.0 + RIXL + UCX on ROCm 7.0
results/                    # Benchmark results (JSON)
figures/                    # Charts and diagrams for the blog post
scripts/
  run_experiments.sh        # Full experiment runner (deploy → benchmark → save)
  generate_serve_configs.py # Regenerate all serve configs programmatically
  generate_blog_charts.py   # Regenerate all matplotlib charts
  extract_sla_capacity.py   # Extract max QPS under SLA from result JSONs
draft/
  blogpost_draft.md         # Blog post markdown
```

## Hardware Requirements

- **GPUs:** AMD Instinct MI325X (288 GB HBM3e), 8 per node
- **Minimum:** 2 nodes (16 GPUs) for 1P1D configurations
- **Recommended:** 4 nodes (32 GPUs) for exploring P:D ratios
- **Networking:** RDMA-capable RoCE fabric with 8x Mellanox ConnectX interfaces per node

## Quick Start

### Option A: Run on Anyscale

**1. Build or use the pre-built container image:**

```bash
# Pre-built:
IMAGE="kouroshhahkha/anyscale-rayllm:nightly-py312-rocm700"

# Or build from Dockerfile:
docker build --platform linux/amd64 -t $IMAGE .
```

**2. Create a compute config on Anyscale:**

Edit `compute_configs/pd-amd-mi325x-autoscale.yaml` to set your cloud name, then register it:

```bash
# Edit the cloud field in the compute config
sed -i 's/cloud: amd2/cloud: YOUR_CLOUD_NAME/' compute_configs/pd-amd-mi325x-autoscale.yaml

# Create the compute config on Anyscale
anyscale compute-config create compute_configs/pd-amd-mi325x-autoscale.yaml --name pd-amd-mi325x
```

**3. Update serve configs with your cloud and compute config:**

Edit the bottom of any serve config YAML:

```yaml
cloud: YOUR_CLOUD_NAME
compute_config: pd-amd-mi325x
image_uri: kouroshhahkha/anyscale-rayllm:nightly-py312-rocm700
```

Or regenerate all configs at once:

```bash
# Edit CLOUD and COMPUTE_CONFIG in scripts/generate_serve_configs.py, then:
python scripts/generate_serve_configs.py
```

**4. Deploy:**

```bash
anyscale service deploy -f serve_configs/pd/qwen235b_2p1d_tp8.yaml
```

**5. Benchmark:**

```bash
python -m ray.llm._internal.serve.benchmark -i \
  -u YOUR_SERVICE_URL -m qwen3-235b --tokenizer Qwen/Qwen3-235B-A22B-FP8
```

### Option B: Run on OSS Ray (KubeRay or bare metal)

**1. Build the container image:**

```bash
# For OSS Ray, change the base image in the Dockerfile:
# FROM rayproject/ray:nightly-py312-cu128  (instead of anyscale/ray:...)
docker build -t pd-vllm-ray .
```

**2. Start a Ray cluster:**

With KubeRay, use the compute config as a reference for your RayCluster spec — the key requirements are:
- `hostIPC: true` and `hostNetwork: true` for RDMA
- 8x `rdma/fabric` resources per node for RoCE
- `supplementalGroups: [991]` for AMD GPU device access
- `privileged: true` for RDMA operations

With bare metal:

```bash
# Head node
ray start --head

# Worker nodes (run on each)
ray start --address=HEAD_NODE_IP:6379
```

**3. Deploy:**

The serve configs have Anyscale-specific fields (`cloud`, `compute_config`, `image_uri`) at the bottom — `ray serve deploy` ignores these and uses only the `applications` section:

```bash
ray serve deploy serve_configs/pd/qwen235b_2p1d_tp8.yaml
```

**4. Benchmark:**

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

| Config | P:D Ratio | GPUs | Use Case |
|--------|-----------|------|----------|
| `pd/qwen235b_1p1d_tp8.yaml` | 1P:1D | 16 | Minimum PD setup |
| `pd/qwen235b_2p1d_tp8.yaml` | 2P:1D | 24 | Long input, short output |
| `pd/qwen235b_1p2d_tp8.yaml` | 1P:2D | 24 | Moderate output |
| `pd/qwen235b_1p2d_tp8_cache.yaml` | 1P:2D | 24 | Multi-turn with prefix caching |
| `pd/qwen235b_1p3d_tp8.yaml` | 1P:3D | 32 | Long output |
| `pd/qwen235b_2p2d_tp8.yaml` | 2P:2D | 32 | Balanced |

### Aggregated Baselines

| Config | Replicas | GPUs |
|--------|----------|------|
| `agg/qwen235b_2agg_tp8.yaml` | 2 | 16 |
| `agg/qwen235b_3agg_tp8.yaml` | 3 | 24 |
| `agg/qwen235b_4agg_tp8.yaml` | 4 | 32 |

## Running the Full Experiment Suite

```bash
# All experiments sequentially (~4-6 hours):
./scripts/run_experiments.sh

# Or run individual experiments:
./scripts/run_experiments.sh exp1a_pd    # 2P1D PD, ISL=16K, OSL=1K
./scripts/run_experiments.sh exp1a_agg   # 3Agg baseline for above
```

See `scripts/run_experiments.sh` for the complete list of experiments and exact benchmark commands.

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
