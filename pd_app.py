"""PD (Prefill-Decode) disaggregated serving for Ray Serve LLM.

See README.md for architecture overview and request flow.
"""

import asyncio
import logging
import math
import os
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator, Dict, Hashable, List, Optional, Tuple, Type, Union

from fastapi import Request
from ray import serve
from ray.serve.config import AutoscalingContext
from ray.serve._private.common import DeploymentID
from ray.serve.deployment import Application
from ray.serve.exceptions import DeploymentUnavailableError
from ray.serve.handle import DeploymentHandle
from ray.serve.llm import LLMConfig, build_llm_deployment

# --- Monkey-patch: work around transformers version incompatibility ---
# _infer_supports_vision and _set_model_architecture fail on DeepSeek-V3
# due to missing max_position_embeddings attribute in newer transformers.
_orig_infer_vision = LLMConfig._infer_supports_vision
_orig_set_arch = LLMConfig._set_model_architecture


def _safe_infer_supports_vision(self, model_id_or_path):
    try:
        _orig_infer_vision(self, model_id_or_path)
    except Exception:
        self._supports_vision = False


def _safe_set_model_architecture(self, model_id_or_path=None, model_architecture=None):
    try:
        _orig_set_arch(self, model_id_or_path, model_architecture)
    except Exception:
        pass


LLMConfig._infer_supports_vision = _safe_infer_supports_vision
LLMConfig._set_model_architecture = _safe_set_model_architecture
# --- End monkey-patch ---

# --- Monkey-patch: transformers 5.x Qwen3MoeConfig attribute_map compat ---
# transformers >= 5.x remaps 'num_experts' -> 'num_local_experts' via attribute_map.
# vLLM 0.18.0 accesses config.num_experts directly, which breaks in subprocesses
# when the __getattribute__ override doesn't work as expected. Remove the remap
# so that 'num_experts' stays as a real attribute on the config instance.
try:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig as _Qwen3MoeConfig,
    )
    if hasattr(_Qwen3MoeConfig, 'attribute_map') and 'num_experts' in _Qwen3MoeConfig.attribute_map:
        _Qwen3MoeConfig.attribute_map = {
            k: v for k, v in _Qwen3MoeConfig.attribute_map.items()
            if k != 'num_experts'
        }
except Exception:
    pass
# --- End Qwen3MoeConfig compat ---

from ray.llm._internal.common.dict_utils import maybe_apply_llm_deployment_config_defaults
from ray.llm._internal.serve.core.configs.openai_api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)
from ray.llm._internal.serve.core.ingress.ingress import (
    OpenAiIngress,
    _sanitize_chat_completion_request,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.core.protocol import RawRequestInfo
from ray.llm._internal.serve.core.server.llm_server import LLMServer
from ray.llm._internal.serve.serving_patterns.prefill_decode.pd_server import RequestType
from ray.llm._internal.serve.utils.broadcast import broadcast
from ray.llm._internal.serve.utils.lora_serve_utils import LoraModelLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Default max sessions tracked per replica via the multiplexing LRU.
DEFAULT_MAX_SESSIONS_PER_REPLICA = 1000

# Session ID header name — must match the client convention.
SESSION_ID_HEADER = os.environ.get("SESSION_ID_HEADER", "x-session-id")

# QPS window for autoscaling stats.
QPS_WINDOW_SECONDS = 10

# Pre-warm constants for DecodeServer startup.
_PREWARM_PROMPT = " x"
_PREWARM_MAX_TOKENS = 1
_PREWARM_RETRY_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# Base: SessionAwareLLMServer — register sessions via @serve.multiplexed
# ---------------------------------------------------------------------------


class SessionAwareLLMServer(LLMServer):
    """LLMServer that registers session IDs via the multiplexing mechanism.

    This makes each replica report its active session IDs to the controller,
    which propagates them to the router. The router then prefers replicas
    that already own a given session.
    """

    async def _start_engine(self):
        """Override to patch HF config detection before engine starts."""
        from ray.llm._internal.serve.core.configs.llm_config import LLMConfig as _LC
        _orig_vision = _LC._infer_supports_vision.__wrapped__ if hasattr(_LC._infer_supports_vision, '__wrapped__') else None

        if not getattr(_LC, '_hf_patched', False):
            _real_infer = _LC._infer_supports_vision
            _real_arch = _LC._set_model_architecture

            def _safe_vision(self_cfg, path):
                try:
                    _real_infer(self_cfg, path)
                except Exception:
                    self_cfg._supports_vision = False

            def _safe_arch(self_cfg, path=None, arch=None):
                try:
                    _real_arch(self_cfg, path, arch)
                except Exception:
                    pass

            _LC._infer_supports_vision = _safe_vision
            _LC._set_model_architecture = _safe_arch
            _LC._hf_patched = True

        return await super()._start_engine()

    def _init_multiplex_loader(
        self, model_downloader_cls: Optional[Type[LoraModelLoader]] = None
    ):
        """Override: if no LoRA config, set up a no-op @serve.multiplexed
        handler that just registers session IDs with the replica."""

        mx_config = self._llm_config.multiplex_config()

        if mx_config is not None:
            # LoRA is configured — use the normal LoRA multiplex loader.
            super()._init_multiplex_loader(model_downloader_cls)
            return

        # No LoRA — set up session-only multiplexing.
        max_sessions = self._llm_config.experimental_configs.get(
            "max_sessions_per_replica", DEFAULT_MAX_SESSIONS_PER_REPLICA
        )

        async def _register_session(session_id: str) -> None:
            """No-op 'load' — the @serve.multiplexed decorator handles
            registering this session_id with the controller so the router
            knows this replica owns this session."""
            logger.debug(
                f"[SessionAwareLLMServer] Registered session: {session_id!r}"
            )
            return None

        self._load_model = serve.multiplexed(
            max_num_models_per_replica=max_sessions,
        )(_register_session)

    async def _maybe_resolve_lora_from_multiplex(self) -> None:
        """Override: if no LoRA config, just call _load_model to trigger
        the @serve.multiplexed registration, but skip LoRA resolution."""
        multiplexed_model_id = serve.get_multiplexed_model_id()
        if not multiplexed_model_id:
            return

        if self._llm_config.lora_config is not None:
            # LoRA path — delegate to parent.
            await super()._maybe_resolve_lora_from_multiplex()
        else:
            # Session path — call _load_model to register the session ID
            # with the multiplexing system (no actual model loading).
            await self._load_model(multiplexed_model_id)


# ---------------------------------------------------------------------------
# Ingress: SessionAwareIngress
# ---------------------------------------------------------------------------


class SessionAwareIngress(OpenAiIngress):
    """Ingress with session-affinity routing.

    Extracts X-Session-Id from request headers and uses it as
    multiplexed_model_id for session-aware routing to replicas.
    """

    async def _get_response(
        self,
        *,
        body: Union[
            CompletionRequest,
            ChatCompletionRequest,
        ],
        call_method: str,
        raw_request: Optional[Request] = None,
    ):
        """Override to add session-aware routing.

        If X-Session-Id header is present, configures the handle with
        multiplexed_model_id=session_id for session affinity.
        """
        model_id = await self._get_model_id(body.model)

        # Get base handle (without session routing yet)
        base_handle = self._get_configured_serve_handle(model_id)

        # Extract session ID from request headers
        session_id = None
        if raw_request is not None:
            session_id = raw_request.headers.get(SESSION_ID_HEADER)

        # If session ID present, configure handle for session routing
        if session_id:
            model_handle = base_handle.options(multiplexed_model_id=session_id)
            logger.debug(
                f"[SessionAwareIngress] Routing to session={session_id} for model={model_id}"
            )
        else:
            model_handle = base_handle

        # TODO(seiji): Remove when we update to Pydantic v2.11+
        if isinstance(body, ChatCompletionRequest):
            body = _sanitize_chat_completion_request(body)

        # Convert request to RawRequestInfo
        raw_request_info: Optional[RawRequestInfo] = None
        if raw_request is not None:
            raw_request_info = RawRequestInfo.from_starlette_request(raw_request)

        # Call the model with the session-aware handle
        async for response in getattr(model_handle, call_method).remote(
            body, raw_request_info
        ):
            yield response


# ---------------------------------------------------------------------------
# Mixin: QPSTrackingMixin — per-replica QPS for coordinated autoscaling
# ---------------------------------------------------------------------------


class QPSTrackingMixin:
    """Per-replica QPS tracking consumed by CoordinatedPDPolicy.

    Must appear before SessionAwareLLMServer in MRO so _run_request
    is intercepted via cooperative super().
    """

    def _init_qps_tracking(self) -> None:
        self._qps_timestamps: deque = deque()

    def _record_request(self) -> None:
        self._qps_timestamps.append(time.time())

    def _get_current_qps(self) -> float:
        cutoff = time.time() - QPS_WINDOW_SECONDS
        while self._qps_timestamps and self._qps_timestamps[0] < cutoff:
            self._qps_timestamps.popleft()
        return len(self._qps_timestamps) / QPS_WINDOW_SECONDS

    async def record_autoscaling_stats(self) -> Dict[str, float]:
        return {"qps": self._get_current_qps()}

    async def _run_request(self, request, *, engine_method, batch_output_stream=False, raw_request_info=None):
        self._record_request()
        return await super()._run_request(
            request,
            engine_method=engine_method,
            batch_output_stream=batch_output_stream,
            raw_request_info=raw_request_info,
        )


# ---------------------------------------------------------------------------
# Server: PrefillServer
# ---------------------------------------------------------------------------


class PrefillServer(QPSTrackingMixin, SessionAwareLLMServer):
    """Prefill-side server.

    Adds prewarm_prefill() — called by each DecodeServer replica at startup
    to complete the NIXL connector handshake before serving real traffic.
    """

    async def __init__(self, *args, **kwargs):
        self._init_qps_tracking()
        await super().__init__(*args, **kwargs)

    async def prewarm_prefill(self, prefill_request: CompletionRequest) -> Optional[dict]:
        """Run one prefill pass and return kv_transfer_params as a dict.

        Returns None on error.
        """
        async for chunk in self.engine.completions(prefill_request, None):
            if hasattr(chunk, "kv_transfer_params") and chunk.kv_transfer_params:
                return chunk.kv_transfer_params
            if isinstance(chunk, ErrorResponse):
                logger.warning("[PrefillServer] prewarm_prefill got error: %s", chunk)
                return None
        return None


# ---------------------------------------------------------------------------
# Server: DecodeServer
# ---------------------------------------------------------------------------


class DecodeServer(QPSTrackingMixin, SessionAwareLLMServer):
    """Decode-side server that also orchestrates prefill.

    Holds a handle to PrefillServer. On startup, runs one dummy
    prefill→decode round-trip per P replica to complete the NIXL
    handshake before accepting real traffic.
    """

    async def __init__(
        self,
        llm_config: LLMConfig,
        prefill_server: DeploymentHandle,
        *,
        engine_cls=None,
        model_downloader=None,
    ):
        self._init_qps_tracking()
        self._prefill_server: DeploymentHandle = prefill_server.options(stream=True)

        # Start the local vLLM engine (blocking — sets self.engine).
        await super().__init__(
            llm_config,
            engine_cls=engine_cls,
            model_downloader=model_downloader,
        )

        # Block until pre-warm completes. Ray Serve won't call check_health
        # until __init__ returns, so this naturally holds the replica in
        # STARTING state without needing a _nixl_ready gate.
        await self._prewarm_all_prefill_replicas()

    # ------------------------------------------------------------------
    # Pre-warm: one dummy round-trip per P replica
    # ------------------------------------------------------------------

    def _make_dummy_request(self, model_id: str) -> CompletionRequest:
        """Build the smallest valid completion request."""
        return CompletionRequest(
            model=model_id,
            prompt=_PREWARM_PROMPT,
            max_tokens=_PREWARM_MAX_TOKENS,
            stream=False,
            request_id=f"prewarm-{uuid.uuid4()}",
        )

    async def _prewarm_all_prefill_replicas(self) -> None:
        """Run one prefill→decode round-trip per P replica to complete
        the NIXL handshake on both sides before traffic arrives."""
        logger.info("[DecodeServer] Starting pre-warm across all P replicas.")

        # 1. Dummy request — smallest valid prompt.
        model_id = self._llm_config.model_id
        dummy = self._make_dummy_request(model_id)

        # 2. Prefill-side request (do_remote_decode=True so P pushes KV to us).
        prefill_req = self._prepare_prefill_request(dummy)

        # 3. Broadcast to every live P replica; each runs prewarm_prefill()
        #    and returns kv_transfer_params as a plain dict.
        #    broadcast() is synchronous internally — run it in an executor.
        #    P replicas start in parallel with D, so retry until they are up.
        kv_params_list: List[Any] = []
        attempt = 0
        while True:
            attempt += 1
            try:
                kv_params_list = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: broadcast(
                        self._prefill_server,
                        method_name="prewarm_prefill",
                        args=[prefill_req],
                    ),
                )
                break  # success
            except DeploymentUnavailableError:
                logger.info(
                    "[DecodeServer] PrefillServer not available yet (attempt %d); "
                    "retrying in %.0fs...",
                    attempt,
                    _PREWARM_RETRY_INTERVAL_S,
                )
                await asyncio.sleep(_PREWARM_RETRY_INTERVAL_S)
            except Exception as exc:
                logger.warning(
                    "[DecodeServer] broadcast() attempt %d failed with %s: %s; "
                    "retrying in %.0fs...",
                    attempt,
                    type(exc).__name__,
                    exc,
                    _PREWARM_RETRY_INTERVAL_S,
                )
                await asyncio.sleep(_PREWARM_RETRY_INTERVAL_S)
        logger.info(
            "[DecodeServer] broadcast() reached %d P replica(s); "
            "driving local decode to complete the handshake.",
            len(kv_params_list),
        )

        # 4. Build one decode request per P replica result.
        decode_reqs: List[CompletionRequest] = []
        for idx, kv_params in enumerate(kv_params_list):
            if not kv_params:
                logger.warning(
                    "[DecodeServer] P replica %d returned empty kv_params; skipping.", idx
                )
                continue
            req = dummy.model_copy(deep=True)
            req.kv_transfer_params = kv_params
            decode_reqs.append(req)

        # 5 & 6. Send all decode requests to the local engine concurrently and
        #        await them all — this triggers add_remote_agent on D's NIXL side
        #        for each P, completing the full connector handshake.
        async def _decode_one(req: CompletionRequest, idx: int) -> None:
            async for _ in self.engine.completions(req, None):
                pass
            logger.info("[DecodeServer] Pre-warm handshake done for P replica %d.", idx)

        await asyncio.gather(*[_decode_one(r, i) for i, r in enumerate(decode_reqs)])

        logger.info("[DecodeServer] Pre-warm complete — all P replicas registered.")

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _extract_session_id(
        self, raw_request_info: Optional[RawRequestInfo]
    ) -> Optional[str]:
        if raw_request_info is None:
            return None
        return raw_request_info.headers.get(SESSION_ID_HEADER) or None

    def _prepare_prefill_request(self, request: RequestType) -> RequestType:
        prefill_request = request.model_copy(deep=True)
        prefill_request.kv_transfer_params = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_request.max_tokens = 1
        prefill_request.stream = False
        return prefill_request

    def _prepare_decode_request(
        self,
        request: RequestType,
        prefill_chunk: Union[ChatCompletionResponse, CompletionResponse],
    ) -> RequestType:
        decode_request = request.model_copy(deep=True)
        decode_request.kv_transfer_params = prefill_chunk.kv_transfer_params
        return decode_request

    async def _handle_request_with_prefill_handle(
        self,
        request: RequestType,
        *,
        engine_method: str,
        prefill_handle: DeploymentHandle,
        raw_request_info: Optional[RawRequestInfo] = None,
    ) -> AsyncGenerator[
        Union[str, ChatCompletionResponse, CompletionResponse, ErrorResponse], None
    ]:
        # Prefill phase — one remote hop.
        prefill_request = self._prepare_prefill_request(request)
        prefill_gen = getattr(prefill_handle, engine_method).remote(
            prefill_request, raw_request_info
        )
        prefill_chunk = await prefill_gen.__anext__()

        if isinstance(prefill_chunk, ErrorResponse):
            logger.error("[DecodeServer] Prefill error: %s", prefill_chunk)
            yield prefill_chunk
            return

        # Decode phase — directly on the local engine, no remote hop.
        decode_request = self._prepare_decode_request(request, prefill_chunk)
        async for chunk in getattr(self.engine, engine_method)(decode_request, raw_request_info):
            yield chunk

    async def _run_request(self, request, *, engine_method, batch_output_stream=False, raw_request_info=None):
        """Override LLMServer._run_request to inject the prefill→local-decode flow."""
        self._record_request()
        await self._maybe_add_request_id_to_request(request)
        await self._maybe_resolve_lora_from_multiplex()

        session_id = self._extract_session_id(raw_request_info)
        prefill_handle = self._prefill_server
        if session_id:
            prefill_handle = prefill_handle.options(multiplexed_model_id=session_id)
            logger.debug("[DecodeServer] session_id=%r", session_id)

        return self._handle_request_with_prefill_handle(
            request,
            engine_method=engine_method,
            prefill_handle=prefill_handle,
            raw_request_info=raw_request_info,
        )


# ---------------------------------------------------------------------------
# Builder: Ingress → DecodeServer ← PrefillServer
# ---------------------------------------------------------------------------


def build_pd_openai_app(pd_serving_args: dict) -> Application:
    """Build the PD application from a PDServingArgs dict (loaded from YAML).

    Builds Prefill first, injects it into Decode, then wires up the Ingress.
    """
    from ray.llm._internal.serve.serving_patterns.prefill_decode.builder import PDServingArgs

    pd_config = PDServingArgs.model_validate(pd_serving_args)

    prefill_dp_size = pd_config.prefill_config.engine_kwargs.get("data_parallel_size", 1)
    decode_dp_size = pd_config.decode_config.engine_kwargs.get("data_parallel_size", 1)

    if prefill_dp_size > 1 or decode_dp_size > 1:
        raise NotImplementedError(
            "build_pd_openai_app does not yet support data_parallel_size > 1."
        )

    # 1. Build prefill — PrefillServer adds prewarm_prefill() for broadcast().
    prefill_deployment = build_llm_deployment(
        pd_config.prefill_config,
        name_prefix="Prefill:",
        deployment_cls=PrefillServer,
    )

    # 2. Build decode — DecodeServer receives the prefill handle at bind time.
    decode_deployment = build_llm_deployment(
        pd_config.decode_config,
        name_prefix="Decode:",
        deployment_cls=DecodeServer,
        bind_kwargs={"prefill_server": prefill_deployment},
    )

    # 3. Ingress — routes directly to DecodeServer (no proxy layer).
    ingress_cls_config = pd_config.ingress_cls_config
    default_ingress_options = ingress_cls_config.ingress_cls.get_deployment_options(
        [pd_config.prefill_config, pd_config.decode_config]
    )
    ingress_options = maybe_apply_llm_deployment_config_defaults(
        default_ingress_options, pd_config.ingress_deployment_config
    )
    ingress_cls = make_fastapi_ingress(ingress_cls_config.ingress_cls)
    return serve.deployment(ingress_cls, **ingress_options).bind(
        llm_deployments=[decode_deployment],
        **ingress_cls_config.ingress_extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Autoscaling: LogOnChange — value-based log deduplication
# ---------------------------------------------------------------------------


class LogOnChange:
    """Deduplicates log messages by value *and* time.

    Logs when either:
    - the *value* for a given key changes, or
    - at least *min_interval_s* seconds have elapsed since the last log
      for that key (periodic re-log for operator visibility).

    Usage::

        _log = LogOnChange(logger, min_interval_s=10)

        # Inside a hot loop:
        _log("scaling", desired, "Scaling to %s", desired)
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        default_level: int = logging.INFO,
        min_interval_s: float = 10.0,
    ):
        self._logger = logger
        self._default_level = default_level
        self._min_interval_s = min_interval_s
        self._last_value: Dict[str, Any] = {}
        self._last_time: Dict[str, float] = {}

    def __call__(
        self,
        key: str,
        value: Hashable,
        msg: str,
        *args,
        level: Optional[int] = None,
    ) -> bool:
        """Log *msg* if *value* changed or *min_interval_s* elapsed.

        Returns True if the message was logged, False if suppressed.
        """
        now = time.monotonic()
        prev_value = self._last_value.get(key)
        prev_time = self._last_time.get(key, 0.0)

        value_changed = prev_value != value or key not in self._last_value

        if not value_changed and (now - prev_time) < self._min_interval_s:
            return False

        self._last_value[key] = value
        self._last_time[key] = now
        self._logger.log(level or self._default_level, msg, *args)
        return True

    def reset(self, key: Optional[str] = None) -> None:
        """Clear tracked state so the next call logs unconditionally."""
        if key is None:
            self._last_value.clear()
            self._last_time.clear()
        else:
            self._last_value.pop(key, None)
            self._last_time.pop(key, None)


# ---------------------------------------------------------------------------
# Autoscaling: CoordinatedPDPolicy
# ---------------------------------------------------------------------------


class CoordinatedPDPolicy:
    """Autoscaling policy that scales Ingress, Prefill, and Decode in lockstep.

    Ratio parameters define relative replica counts within one "set"
    (e.g. ingress_ratio=4, prefill_ratio=1, decode_ratio=1 → 4:1:1).
    """

    def __init__(
        self,
        *,
        ingress_ratio: int,
        prefill_ratio: int,
        decode_ratio: int,
        target_qps_per_prefill: float = 10.0,
        target_qps_per_decode: float = 10.0,
    ):
        for name, value in [
            ("ingress_ratio", ingress_ratio),
            ("prefill_ratio", prefill_ratio),
            ("decode_ratio", decode_ratio),
        ]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "%s must be an integer >= 1, got %r" % (name, value)
                )

        self.ingress_ratio = ingress_ratio
        self.prefill_ratio = prefill_ratio
        self.decode_ratio = decode_ratio

        self.target_qps_per_prefill = target_qps_per_prefill
        self.target_qps_per_decode = target_qps_per_decode

        # Use ray.serve logger so autoscaling logs appear in the right namespace.
        self._serve_logger = logging.getLogger("ray.serve")
        self._log = LogOnChange(self._serve_logger)

        self._serve_logger.info(
            "CoordinatedPDPolicy initialized — ratios %d:%d:%d (I:P:D), "
            "targets P=%.1f D=%.1f qps/replica",
            ingress_ratio, prefill_ratio, decode_ratio,
            target_qps_per_prefill, target_qps_per_decode,
        )

    def _find_deployments(
        self, contexts: Dict[DeploymentID, AutoscalingContext]
    ) -> Tuple:
        ingress_id = prefill_id = decode_id = None
        for deployment_id in contexts:
            name = deployment_id.name
            if "Prefill" in name:
                prefill_id = deployment_id
            elif "Decode" in name:
                decode_id = deployment_id
            else:
                ingress_id = deployment_id
        return ingress_id, prefill_id, decode_id

    def __call__(
        self,
        contexts: Dict[DeploymentID, AutoscalingContext],
    ) -> Tuple[Dict[DeploymentID, int], Dict]:
        ingress_id, prefill_id, decode_id = self._find_deployments(contexts)

        if not (ingress_id and prefill_id and decode_id):
            found = [d.name for d in contexts]
            self._serve_logger.warning(
                "CoordinatedPDPolicy: not all tiers found (%s). "
                "Maintaining current replicas.", found,
            )
            return {
                dep_id: ctx.current_num_replicas
                for dep_id, ctx in contexts.items()
            }, {}

        prefill_ctx = contexts[prefill_id]
        decode_ctx = contexts[decode_id]

        sets_needed = 1
        for ctx, target_qps_per_replica, ratio in [
            (prefill_ctx, self.target_qps_per_prefill, self.prefill_ratio),
            (decode_ctx, self.target_qps_per_decode, self.decode_ratio),
        ]:
            qps_metric = ctx.aggregated_metrics.get("qps", {})
            if not qps_metric:
                continue
            total_qps = sum(qps_metric.values())
            if target_qps_per_replica > 0 and ratio > 0:
                replicas_needed = math.ceil(total_qps / target_qps_per_replica)
                sets_needed = max(
                    sets_needed,
                    math.ceil(replicas_needed / ratio),
                )

        new_sets = max(1, sets_needed)

        target = {
            ingress_id: new_sets * self.ingress_ratio,
            prefill_id: new_sets * self.prefill_ratio,
            decode_id: new_sets * self.decode_ratio,
        }

        desired = {}
        for dep_id, count in target.items():
            ctx = contexts[dep_id]
            desired[dep_id] = max(
                ctx.capacity_adjusted_min_replicas,
                min(ctx.capacity_adjusted_max_replicas, count),
            )

        current = {
            dep_id: contexts[dep_id].current_num_replicas
            for dep_id in desired
        }

        desired_tuple = (
            desired[ingress_id], desired[prefill_id], desired[decode_id],
        )
        if desired != current:
            if desired == target:
                self._log(
                    "scaling", desired_tuple,
                    "CoordinatedPDPolicy: "
                    "I=%d→%d  P=%d→%d  D=%d→%d",
                    current[ingress_id], desired[ingress_id],
                    current[prefill_id], desired[prefill_id],
                    current[decode_id], desired[decode_id],
                )
            else:
                self._log(
                    "scaling", desired_tuple,
                    "CoordinatedPDPolicy: "
                    "policy wants I=%d P=%d D=%d, "
                    "clamped to I=%d P=%d D=%d  "
                    "(current I=%d P=%d D=%d)",
                    target[ingress_id], target[prefill_id], target[decode_id],
                    desired[ingress_id], desired[prefill_id], desired[decode_id],
                    current[ingress_id], current[prefill_id], current[decode_id],
                )

        return desired, {}
