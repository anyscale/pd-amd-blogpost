"""Aggregated serving wrapper with vision/architecture detection fix.

Works around a transformers version incompatibility where
_infer_supports_vision and _set_model_architecture fail on DeepSeek
models due to missing max_position_embeddings attribute.
"""

import logging
from ray.serve.llm import build_openai_app
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from pd_app import SessionAwareLLMServer

logger = logging.getLogger(__name__)

# Patch both failing methods to be fault-tolerant
_orig_infer_vision = LLMConfig._infer_supports_vision
_orig_set_arch = LLMConfig._set_model_architecture


def _safe_infer_supports_vision(self, model_id_or_path):
    try:
        _orig_infer_vision(self, model_id_or_path)
    except Exception as e:
        logger.warning("Vision detection failed for %s: %s", model_id_or_path, e)
        self._supports_vision = False


def _safe_set_model_architecture(self, model_id_or_path=None, model_architecture=None):
    try:
        _orig_set_arch(self, model_id_or_path, model_architecture)
    except Exception as e:
        logger.warning("Architecture detection failed for %s: %s", model_id_or_path, e)
        # Leave _model_architecture as default


LLMConfig._infer_supports_vision = _safe_infer_supports_vision
LLMConfig._set_model_architecture = _safe_set_model_architecture


class PatchedSessionAwareLLMServer(SessionAwareLLMServer):
    """SessionAwareLLMServer with HF config detection fixes applied at import time."""
    pass


def build_agg_openai_app(args: dict):
    return build_openai_app(args)
