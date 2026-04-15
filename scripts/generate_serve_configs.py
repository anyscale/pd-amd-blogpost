#!/usr/bin/env python3
"""Generate all serve configs for PD disaggregation blog experiments.

Based on the working thesis_5 config pattern (ray.serve.llm built-in PD,
Qwen3-235B-A22B-FP8 pre-quantized model, no custom pd_app).

Run: python scripts/generate_serve_configs.py
"""

import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD_DIR = os.path.join(REPO_ROOT, "serve_configs", "pd")
AGG_DIR = os.path.join(REPO_ROOT, "serve_configs", "agg")

# Shared engine kwargs
BASE_ENGINE = {
    "tensor_parallel_size": 8,
    "enable_expert_parallel": True,
    "enable_prefix_caching": False,
    "max_model_len": 40960,
    "kv_cache_dtype": "fp8",
    "load_format": "dummy",
}

RUNTIME_ENV = {
    "env_vars": {
        "SAFETENSORS_FAST_GPU": "1",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "UCX_TLS": "rc,sm,self,rocm_copy,rocm_ipc",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random",
    }
}

SERVICE_ENV_VARS = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_THROUGHPUT_OPTIMIZED": "1",
    "RAY_SERVE_HAPROXY_TCP_NODELAY": "0",
    "RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP": "1",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_USE_TRITON_FLASH_ATTN": "0",
    "UCX_TLS": "rc_mlx5,ud_mlx5,dc_mlx5",
    "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
    "UCX_MEM_MMAP_HOOK_MODE": "none",
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
}

IMAGE_URI = "kouroshhahkha/anyscale-rayllm:nightly-py312-rocm700"
CLOUD = "amd2"
COMPUTE_CONFIG = "amd-short-autoscale-2-10"


def make_pd_config(name: str, num_prefill: int, num_decode: int,
                   prefix_caching: bool = False, comment: str = "") -> dict:
    """Generate a PD disaggregated serve config."""
    total_gpus = (num_prefill + num_decode) * 8
    num_ingress = max(4, (num_prefill + num_decode) * 4)

    prefill_engine = {
        **BASE_ENGINE,
        "enforce_eager": True,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 32,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
        },
        "enable_prefix_caching": prefix_caching,
    }

    decode_engine = {
        **BASE_ENGINE,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
        },
        "enable_prefix_caching": prefix_caching,
    }

    config = {
        "applications": [{
            "name": "pd-qwen235b",
            "import_path": "ray.serve.llm:build_pd_openai_app",
            "route_prefix": "/",
            "args": {
                "prefill_config": {
                    "model_loading_config": {
                        "model_id": "qwen3-235b",
                        "model_source": "Qwen/Qwen3-235B-A22B-FP8",
                    },
                    "engine_kwargs": prefill_engine,
                    "deployment_config": {
                        "num_replicas": num_prefill,
                        "max_ongoing_requests": 1000000,
                        "logging_config": {"enable_access_log": False},
                    },
                    "runtime_env": RUNTIME_ENV,
                },
                "decode_config": {
                    "model_loading_config": {
                        "model_id": "qwen3-235b",
                        "model_source": "Qwen/Qwen3-235B-A22B-FP8",
                    },
                    "engine_kwargs": decode_engine,
                    "deployment_config": {
                        "num_replicas": num_decode,
                        "max_ongoing_requests": 1000000,
                        "logging_config": {"enable_access_log": False},
                    },
                    "runtime_env": RUNTIME_ENV,
                },
                "ingress_deployment_config": {
                    "num_replicas": num_ingress,
                    "logging_config": {"enable_access_log": False},
                },
            },
        }],
        "name": f"qwen235b_pd_{num_prefill}p{num_decode}d_tp8_service",
        "working_dir": ".",
        "cloud": CLOUD,
        "compute_config": COMPUTE_CONFIG,
        "image_uri": IMAGE_URI,
        "query_auth_token_enabled": False,
        "env_vars": SERVICE_ENV_VARS,
    }
    return config


def make_agg_config(name: str, num_replicas: int,
                    prefix_caching: bool = False, comment: str = "") -> dict:
    """Generate an aggregated serve config."""
    total_gpus = num_replicas * 8

    engine = {
        **BASE_ENGINE,
        "max_num_batched_tokens": 32768,
        "enable_prefix_caching": prefix_caching,
    }

    config = {
        "applications": [{
            "name": "aggregated-qwen235b",
            "import_path": "ray.serve.llm:build_openai_app",
            "route_prefix": "/",
            "args": {
                "llm_configs": [{
                    "model_loading_config": {
                        "model_id": "qwen3-235b",
                        "model_source": "Qwen/Qwen3-235B-A22B-FP8",
                    },
                    "engine_kwargs": engine,
                    "deployment_config": {
                        "num_replicas": num_replicas,
                        "logging_config": {"enable_access_log": False},
                    },
                    "runtime_env": RUNTIME_ENV,
                }],
                "ingress_deployment_config": {
                    "autoscaling_config": {
                        "min_replicas": 16,
                        "max_replicas": 16,
                    },
                },
            },
        }],
        "name": f"qwen235b_agg_{num_replicas}_tp8_service",
        "working_dir": ".",
        "cloud": CLOUD,
        "compute_config": COMPUTE_CONFIG,
        "image_uri": IMAGE_URI,
        "query_auth_token_enabled": False,
        "env_vars": SERVICE_ENV_VARS,
    }
    return config


def write_yaml(config: dict, path: str, comment: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if comment:
            for line in comment.strip().split("\n"):
                f.write(f"# {line}\n")
            f.write("\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {path}")


def main():
    print("Generating serve configs...\n")

    # PD configs
    configs = [
        ("pd/qwen235b_2p1d_tp8.yaml", make_pd_config("2p1d", 2, 1),
         "PD — Qwen3-235B-A22B-FP8 (2P1D, TP8/EP8)\n24 GPUs (3 nodes). Prefill-heavy ratio for long-input short-output.\nUsed for: EXP-1a, EXP-3b"),
        ("pd/qwen235b_1p2d_tp8.yaml", make_pd_config("1p2d", 1, 2),
         "PD — Qwen3-235B-A22B-FP8 (1P2D, TP8/EP8)\n24 GPUs (3 nodes). Decode-heavy ratio for moderate output.\nUsed for: EXP-1d, EXP-3c"),
        ("pd/qwen235b_1p2d_tp8_cache.yaml", make_pd_config("1p2d_cache", 1, 2, prefix_caching=True),
         "PD — Qwen3-235B-A22B-FP8 (1P2D, TP8/EP8, prefix caching)\n24 GPUs (3 nodes). Decode-heavy with KV cache for multi-turn warm workloads.\nUsed for: EXP-1c"),
        ("pd/qwen235b_1p3d_tp8.yaml", make_pd_config("1p3d", 1, 3),
         "PD — Qwen3-235B-A22B-FP8 (1P3D, TP8/EP8)\n32 GPUs (4 nodes). Decode-heavy ratio for long output.\nUsed for: EXP-1b, EXP-3f"),
        ("pd/qwen235b_2p2d_tp8.yaml", make_pd_config("2p2d", 2, 2),
         "PD — Qwen3-235B-A22B-FP8 (2P2D, TP8/EP8)\n32 GPUs (4 nodes). Balanced ratio.\nUsed for: EXP-3d"),
    ]

    # Agg configs
    configs += [
        ("agg/qwen235b_3agg_tp8.yaml", make_agg_config("3agg", 3),
         "Aggregated — Qwen3-235B-A22B-FP8 (3 replicas, TP8/EP8)\n24 GPUs (3 nodes). Agg baseline for 24-GPU comparisons.\nUsed for: EXP-1a, EXP-1c, EXP-1d, EXP-3h"),
        ("agg/qwen235b_4agg_tp8.yaml", make_agg_config("4agg", 4),
         "Aggregated — Qwen3-235B-A22B-FP8 (4 replicas, TP8/EP8)\n32 GPUs (4 nodes). Agg baseline for 32-GPU comparisons.\nUsed for: EXP-1b, EXP-3i"),
    ]

    for relpath, config, comment in configs:
        path = os.path.join(REPO_ROOT, "serve_configs", relpath)
        write_yaml(config, path, comment)

    print(f"\nDone — {len(configs)} configs generated.")


if __name__ == "__main__":
    main()
