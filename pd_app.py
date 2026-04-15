"""PD (Prefill-Decode) disaggregated serving with session-aware routing for Ray Serve LLM.

Provides:
- SessionAwareLLMServer: registers session IDs via @serve.multiplexed for cache affinity
- SessionAwareIngress: extracts X-Session-Id header for session-aware routing
- PrefillServer: prefill-side server with NIXL handshake pre-warming
- DecodeServer: decode-side server that orchestrates prefill → KV transfer → local decode
- build_pd_openai_app: wires the Ingress → Decode ← Prefill topology
"""

import asyncio
import logging
import os
import uuid
from typing import Any, AsyncGenerator, List, Optional, Type, Union

from fastapi import Request
from ray import serve
from ray.serve.deployment import Application
from ray.serve.exceptions import DeploymentUnavailableError
from ray.serve.handle import DeploymentHandle
from ray.serve.llm import LLMConfig, build_llm_deployment

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
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_SESSIONS_PER_REPLICA = 1000
SESSION_ID_HEADER = os.environ.get("SESSION_ID_HEADER", "x-session-id")

_PREWARM_PROMPT = " x"
_PREWARM_MAX_TOKENS = 1
_PREWARM_RETRY_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# SessionAwareLLMServer — session registration via @serve.multiplexed
# ---------------------------------------------------------------------------


class SessionAwareLLMServer(LLMServer):
    """LLMServer that registers session IDs via the multiplexing mechanism.

    Each replica reports its active session IDs to the controller,
    which propagates them to the router. The router then prefers replicas
    that already own a given session — enabling KV cache reuse across
    multi-turn requests.
    """

    def _init_multiplex_loader(
        self, model_downloader_cls: Optional[Type[LoraModelLoader]] = None
    ):
        mx_config = self._llm_config.multiplex_config()

        if mx_config is not None:
            super()._init_multiplex_loader(model_downloader_cls)
            return

        max_sessions = self._llm_config.experimental_configs.get(
            "max_sessions_per_replica", DEFAULT_MAX_SESSIONS_PER_REPLICA
        )

        async def _register_session(session_id: str) -> None:
            logger.debug("Registered session: %s", session_id)
            return None

        self._load_model = serve.multiplexed(
            max_num_models_per_replica=max_sessions,
        )(_register_session)

    async def _maybe_resolve_lora_from_multiplex(self) -> None:
        multiplexed_model_id = serve.get_multiplexed_model_id()
        if not multiplexed_model_id:
            return

        if self._llm_config.lora_config is not None:
            await super()._maybe_resolve_lora_from_multiplex()
        else:
            await self._load_model(multiplexed_model_id)


# ---------------------------------------------------------------------------
# SessionAwareIngress — extracts X-Session-Id for affinity routing
# ---------------------------------------------------------------------------


class SessionAwareIngress(OpenAiIngress):
    """Ingress that routes requests to the same replica based on session ID."""

    async def _get_response(
        self,
        *,
        body: Union[CompletionRequest, ChatCompletionRequest],
        call_method: str,
        raw_request: Optional[Request] = None,
    ):
        model_id = await self._get_model_id(body.model)
        base_handle = self._get_configured_serve_handle(model_id)

        session_id = None
        if raw_request is not None:
            session_id = raw_request.headers.get(SESSION_ID_HEADER)

        if session_id:
            model_handle = base_handle.options(multiplexed_model_id=session_id)
        else:
            model_handle = base_handle

        if isinstance(body, ChatCompletionRequest):
            body = _sanitize_chat_completion_request(body)

        raw_request_info: Optional[RawRequestInfo] = None
        if raw_request is not None:
            raw_request_info = RawRequestInfo.from_starlette_request(raw_request)

        async for response in getattr(model_handle, call_method).remote(
            body, raw_request_info
        ):
            yield response


# ---------------------------------------------------------------------------
# PrefillServer
# ---------------------------------------------------------------------------


class PrefillServer(SessionAwareLLMServer):
    """Prefill-side server.

    Adds prewarm_prefill() for NIXL connector handshake at startup.
    """

    async def prewarm_prefill(self, prefill_request: CompletionRequest) -> Optional[dict]:
        """Run one prefill pass and return kv_transfer_params."""
        async for chunk in self.engine.completions(prefill_request, None):
            if hasattr(chunk, "kv_transfer_params") and chunk.kv_transfer_params:
                return chunk.kv_transfer_params
            if isinstance(chunk, ErrorResponse):
                logger.warning("prewarm_prefill error: %s", chunk)
                return None
        return None


# ---------------------------------------------------------------------------
# DecodeServer
# ---------------------------------------------------------------------------


class DecodeServer(SessionAwareLLMServer):
    """Decode-side server that orchestrates prefill → KV transfer → local decode.

    Holds a handle to PrefillServer. On startup, runs one dummy prefill→decode
    round-trip per P replica to complete the NIXL handshake.
    """

    async def __init__(
        self,
        llm_config: LLMConfig,
        prefill_server: DeploymentHandle,
        *,
        engine_cls=None,
        model_downloader=None,
    ):
        self._prefill_server: DeploymentHandle = prefill_server.options(stream=True)
        await super().__init__(
            llm_config, engine_cls=engine_cls, model_downloader=model_downloader,
        )
        await self._prewarm_all_prefill_replicas()

    # --- Pre-warm: NIXL handshake ---

    def _make_dummy_request(self, model_id: str) -> CompletionRequest:
        return CompletionRequest(
            model=model_id,
            prompt=_PREWARM_PROMPT,
            max_tokens=_PREWARM_MAX_TOKENS,
            stream=False,
            request_id=f"prewarm-{uuid.uuid4()}",
        )

    async def _prewarm_all_prefill_replicas(self) -> None:
        logger.info("Starting pre-warm across all P replicas.")
        model_id = self._llm_config.model_id
        dummy = self._make_dummy_request(model_id)
        prefill_req = self._prepare_prefill_request(dummy)

        # Broadcast to every live P replica; retry until they're up.
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
                break
            except (DeploymentUnavailableError, Exception) as exc:
                logger.info(
                    "PrefillServer not ready (attempt %d, %s); retrying in %.0fs...",
                    attempt, type(exc).__name__, _PREWARM_RETRY_INTERVAL_S,
                )
                await asyncio.sleep(_PREWARM_RETRY_INTERVAL_S)

        logger.info("Reached %d P replica(s); driving local decode for handshake.", len(kv_params_list))

        # Build decode requests from prefill results.
        decode_reqs: List[CompletionRequest] = []
        for idx, kv_params in enumerate(kv_params_list):
            if not kv_params:
                logger.warning("P replica %d returned empty kv_params; skipping.", idx)
                continue
            req = dummy.model_copy(deep=True)
            req.kv_transfer_params = kv_params
            decode_reqs.append(req)

        # Run all decode requests concurrently to complete handshakes.
        async def _decode_one(req: CompletionRequest, idx: int) -> None:
            async for _ in self.engine.completions(req, None):
                pass
            logger.info("Pre-warm handshake done for P replica %d.", idx)

        await asyncio.gather(*[_decode_one(r, i) for i, r in enumerate(decode_reqs)])
        logger.info("Pre-warm complete — all P replicas registered.")

    # --- Request handling ---

    def _extract_session_id(self, raw_request_info: Optional[RawRequestInfo]) -> Optional[str]:
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
        prefill_request = self._prepare_prefill_request(request)
        prefill_gen = getattr(prefill_handle, engine_method).remote(
            prefill_request, raw_request_info
        )
        prefill_chunk = await prefill_gen.__anext__()

        if isinstance(prefill_chunk, ErrorResponse):
            logger.error("Prefill error: %s", prefill_chunk)
            yield prefill_chunk
            return

        decode_request = self._prepare_decode_request(request, prefill_chunk)
        async for chunk in getattr(self.engine, engine_method)(decode_request, raw_request_info):
            yield chunk

    async def _run_request(self, request, *, engine_method, batch_output_stream=False, raw_request_info=None):
        await self._maybe_add_request_id_to_request(request)
        await self._maybe_resolve_lora_from_multiplex()

        session_id = self._extract_session_id(raw_request_info)
        prefill_handle = self._prefill_server
        if session_id:
            prefill_handle = prefill_handle.options(multiplexed_model_id=session_id)

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
    """Build the PD application from a serve config dict.

    Wires: Ingress → DecodeServer ← PrefillServer
    """
    from ray.llm._internal.serve.serving_patterns.prefill_decode.builder import PDServingArgs

    pd_config = PDServingArgs.model_validate(pd_serving_args)

    prefill_dp_size = pd_config.prefill_config.engine_kwargs.get("data_parallel_size", 1)
    decode_dp_size = pd_config.decode_config.engine_kwargs.get("data_parallel_size", 1)
    if prefill_dp_size > 1 or decode_dp_size > 1:
        raise NotImplementedError("data_parallel_size > 1 not supported.")

    prefill_deployment = build_llm_deployment(
        pd_config.prefill_config,
        name_prefix="Prefill:",
        deployment_cls=PrefillServer,
    )

    decode_deployment = build_llm_deployment(
        pd_config.decode_config,
        name_prefix="Decode:",
        deployment_cls=DecodeServer,
        bind_kwargs={"prefill_server": prefill_deployment},
    )

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
