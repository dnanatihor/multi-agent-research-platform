"""Shared service platform: secrets, guardrails, cost, resilience, and model wiring.

Copied into each bounded context so a service can be built and deployed on its own.
The conversation, corpus, or audit store in the service package remains the system of
record. This module does not persist business entities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)

SERVICE_VERSION = "0.1.0"
# Pinned snapshot. Do not switch this to an auto-updating alias.
PINNED_CHAT_MODEL = "gpt-4o-mini-2024-07-18"
SCRIPTED_MODEL_ID = "portfolio:scripted-v1"

# Published mini-snapshot rate card used when the provider does not return usage.
# Finance can override the per-service settings without changing call sites.
DEFAULT_INPUT_USD_PER_TOKEN = 0.15 / 1_000_000
DEFAULT_OUTPUT_USD_PER_TOKEN = 0.60 / 1_000_000

# Tools the deepagents harness adds by default. A portfolio agent must opt in
# to a tool by leaving it off this list. `execute` and `delete` are never opted in.
HARNESS_TOOLS_DENIED = frozenset(
    {
        "execute",
        "delete",
        "write_file",
        "edit_file",
        "task",
        "ls",
        "read_file",
        "glob",
        "grep",
    }
)

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
trace_id: ContextVar[str] = ContextVar("trace_id", default="")

_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|bearer)\s*[:=]\s*\S+"
)
_INJECTION = (
    ("ignore-instructions", re.compile(r"ignore (all|any|previous|above) instructions", re.I)),
    ("role-hijack", re.compile(r"you are now\b", re.I)),
    ("control-token", re.compile(r"<\s*\|?\s*im_start\s*\|?\s*>", re.I)),
    ("prompt-exfil", re.compile(r"reveal (the )?(system|hidden|developer) prompt", re.I)),
    ("jailbreak", re.compile(r"\b(jailbreak|do anything now)\b", re.I)),
    ("admin-override", re.compile(r"\badmin override\b", re.I)),
)
_KEY_MATERIAL = (
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
)

POLICIES: dict[str, Callable[[list[BaseMessage], list[str]], AIMessage]] = {}


class AppError(Exception):
    """Domain error rendered as a problem+json response at the API edge."""

    def __init__(self, status: int, title: str, detail: str, type_slug: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail
        self.type_slug = type_slug


def redact(text: str) -> str:
    cleaned = _EMAIL.sub("[redacted-email]", text)
    cleaned = _SECRET.sub(r"\1=[redacted]", cleaned)
    for pattern in _KEY_MATERIAL:
        cleaned = pattern.sub("[redacted-secret]", cleaned)
    return cleaned


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "service": getattr(record, "service", ""),
            "version": SERVICE_VERSION,
            "environment": getattr(record, "environment", ""),
            "correlationId": correlation_id.get(),
            "traceId": trace_id.get(),
            "message": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["errorType"] = record.exc_info[0].__name__ if record.exc_info[0] else "error"
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(service: str, environment: str) -> logging.Logger:
    logger = logging.getLogger(service)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            extra = kwargs.setdefault("extra", {})
            extra.setdefault("service", service)
            extra.setdefault("environment", environment)
            return msg, kwargs

    return _Adapter(logger, {})  # type: ignore[return-value]


def screen_input(text: str, max_chars: int) -> None:
    """Reject empty, oversized, or injection-shaped input before any model call."""

    if text is None or not str(text).strip():
        raise AppError(400, "Empty input", "The request content is empty.", "empty-input")
    if len(text) > max_chars:
        raise AppError(
            413,
            "Input too large",
            f"Input exceeds the {max_chars} character limit.",
            "input-too-large",
        )
    for code, pattern in _INJECTION:
        if pattern.search(text):
            raise AppError(
                400,
                "Input rejected",
                "The message was rejected by the input guard.",
                f"prompt-injection:{code}",
            )


def scrub_output(text: str) -> str:
    cleaned = text
    for pattern in _KEY_MATERIAL:
        cleaned = pattern.sub("[redacted-secret]", cleaned)
    return cleaned


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def cost_usd(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> float:
    return round(input_tokens * input_rate + output_tokens * output_rate, 8)


def message_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def transcript(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(message_text(message) for message in messages)


def tool_message_names(messages: Sequence[BaseMessage]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if getattr(message, "type", "") == "tool":
            names.append(getattr(message, "name", "") or "")
    return names


def last_human(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return message_text(message)
    return ""


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }
        ],
    )


def register_policy(name: str, policy: Callable[[list[BaseMessage], list[str]], AIMessage]) -> None:
    POLICIES[name] = policy


def _tool_name(tool: Any) -> str:
    if hasattr(tool, "name"):
        return str(tool.name)
    if isinstance(tool, dict):
        if "name" in tool:
            return str(tool["name"])
        function = tool.get("function") or {}
        return str(function.get("name") or "")
    return ""


class ScriptedChatModel(BaseChatModel):
    """Offline model used when no vault key is configured.

    Behaviour is a named policy so tests and local runs exercise the same graph
    as the pinned OpenAI snapshot without sending data to a provider.
    """

    model_name: str = SCRIPTED_MODEL_ID
    policy_name: str = "default"
    tool_names: list[str] = Field(default_factory=list)
    sleep_s: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "portfolio-scripted"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "policy_name": self.policy_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        names = [name for name in (_tool_name(tool) for tool in tools) if name]
        return self.model_copy(update={"tool_names": names})

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        policy = POLICIES.get(self.policy_name)
        if policy is None:
            raise AppError(500, "Model policy missing", "No scripted policy is registered.", "model-policy")
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return policy(messages, list(self.tool_names))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterable[ChatGenerationChunk]:
        message = self._decide(messages)
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_calls=message.tool_calls)
            )
            return
        parts = re.findall(r"\S+\s*", message.content or "")
        if not parts and message.content:
            parts = [message.content]
        for index, part in enumerate(parts):
            position = "last" if index == len(parts) - 1 else None
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=part, chunk_position=position)
            )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk
            await asyncio.sleep(0)


def install_harness_profiles() -> None:
    """Deny harness tools that are outside this portfolio's least-privilege boundary.

    Profiles are selected by model id. `task` is excluded so specialists cannot
    recursively spawn general-purpose subagents; orchestration is an explicit graph.
    """

    profile = HarnessProfile(
        excluded_tools=HARNESS_TOOLS_DENIED,
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    register_harness_profile(SCRIPTED_MODEL_ID, profile)
    register_harness_profile(f"openai:{PINNED_CHAT_MODEL}", profile)


def build_chat_model(settings: Any, policy_name: str) -> BaseChatModel:
    mode = settings.llm_mode
    api_key = settings.resolved_api_key
    use_scripted = mode == "scripted" or (mode == "auto" and not api_key)
    if use_scripted:
        return ScriptedChatModel(
            model_name=SCRIPTED_MODEL_ID,
            policy_name=policy_name,
            sleep_s=getattr(settings, "script_sleep_s", 0.0) or 0.0,
        )
    if not api_key:
        raise AppError(
            500,
            "Model key missing",
            "OpenAI mode requires a key from the vault file.",
            "secret-missing",
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=PINNED_CHAT_MODEL,
        api_key=api_key.get_secret_value(),
        temperature=0,
        timeout=settings.llm_timeout_s,
        max_retries=0,
    )


def build_deep_agent(
    model: BaseChatModel,
    *,
    system_prompt: str,
    tools: Sequence[Any] | None = None,
    checkpointer: Any = None,
    name: str | None = None,
) -> Any:
    install_harness_profiles()
    return create_deep_agent(
        model=model,
        tools=list(tools or []),
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        name=name,
    )


def agent_config(thread_id: str, recursion_limit: int) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
        "metadata": {
            "correlation_id": correlation_id.get(),
            "trace_id": trace_id.get(),
        },
    }


async def stream_agent_text(
    agent: Any,
    messages: list[dict[str, str]],
    *,
    thread_id: str,
    recursion_limit: int = 8,
) -> Any:
    config = agent_config(thread_id, recursion_limit)
    async for item in agent.astream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        message = item[0] if isinstance(item, tuple) else item
        if getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None):
            continue
        text = message_text(message)
        if text:
            yield text


async def invoke_agent(
    agent: Any,
    messages: list[dict[str, str]] | list[BaseMessage],
    *,
    thread_id: str,
    recursion_limit: int = 30,
) -> str:
    result = await agent.ainvoke(
        {"messages": messages},
        config=agent_config(thread_id, recursion_limit),
    )
    return message_text(result["messages"][-1])


class CircuitBreaker:
    """Opens when the failure ratio in the current window reaches the threshold."""

    def __init__(self, failure_ratio: float = 0.5, window: int = 10, open_seconds: float = 30.0) -> None:
        self.failure_ratio = failure_ratio
        self.window = window
        self.open_seconds = open_seconds
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.open_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record(self, success: bool) -> None:
        if self.state == "half_open":
            self._opened_at = None if success else time.monotonic()
            self._outcomes.clear()
            if success:
                return
        self._outcomes.append(success)
        if len(self._outcomes) == self.window:
            failures = self._outcomes.count(False)
            if failures / len(self._outcomes) >= self.failure_ratio:
                self._opened_at = time.monotonic()
                self._outcomes.clear()


async def retry_async(
    operation: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
) -> Any:
    """Retry an idempotent call with exponential backoff and jitter."""

    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:
            last = exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt) * (0.5 + random.random() * 0.5)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window_s:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


@dataclass
class Span:
    name: str
    trace_id: str
    correlation_id: str
    started_at: float
    attributes: dict[str, Any]
    duration_ms: float = 0.0


@dataclass
class Tracer:
    """In-process trace buffer.

    Production exports these spans to the platform OTEL collector. The buffer is
    the local adapter so every query is traceable before that collector is wired.
    """

    service: str
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Any:
        current = Span(
            name=name,
            trace_id=trace_id.get() or uuid.uuid4().hex,
            correlation_id=correlation_id.get(),
            started_at=time.perf_counter(),
            attributes={key: value for key, value in attributes.items()},
        )
        try:
            yield current
        finally:
            current.duration_ms = round((time.perf_counter() - current.started_at) * 1000, 3)
            self.spans.append(current)

    def for_correlation(self, cid: str) -> list[dict[str, Any]]:
        return [
            {
                "name": span.name,
                "traceId": span.trace_id,
                "correlationId": span.correlation_id,
                "durationMs": span.duration_ms,
                "attributes": span.attributes,
            }
            for span in self.spans
            if span.correlation_id == cid
        ]


class Metrics:
    def __init__(self, service: str) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["service", "method", "path", "status"],
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "llm_cost_usd_total",
            "Estimated model spend in USD",
            ["service"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_latency_seconds",
            "HTTP latency",
            ["service", "path"],
            registry=self.registry,
        )
        self.service = service

    def observe_request(self, method: str, path: str, status: int, elapsed: float) -> None:
        self.requests.labels(self.service, method, path, str(status)).inc()
        self.latency.labels(self.service, path).observe(elapsed)

    def add_cost(self, amount: float) -> None:
        if amount > 0:
            self.llm_cost.labels(self.service).inc(amount)

    def render(self) -> bytes:
        return generate_latest(self.registry)


class CorrelationMiddleware:
    """ASGI middleware so server-sent events are not buffered."""

    def __init__(self, app: Any, service: str, environment: str, metrics: Metrics, limiter: RateLimiter | None) -> None:
        self.app = app
        self.service = service
        self.environment = environment
        self.metrics = metrics
        self.limiter = limiter
        self.logger = configure_logging(service, environment)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        cid = headers.get("x-correlation-id") or str(uuid.uuid4())
        tid = headers.get("x-trace-id") or uuid.uuid4().hex
        path = scope.get("path") or ""
        client = scope.get("client")
        host = client[0] if client else "unknown"
        cid_token = correlation_id.set(cid)
        tid_token = trace_id.set(tid)
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                raw = list(message.get("headers") or [])
                raw.append((b"x-correlation-id", cid.encode("ascii")))
                raw.append((b"x-trace-id", tid.encode("ascii")))
                message = {**message, "headers": raw}
            await send(message)

        try:
            if self.limiter is not None and path.startswith("/api/") and not self.limiter.allow(host):
                response = JSONResponse(
                    problem("rate-limited", "Rate limit exceeded", "Too many requests from this client.", 429, cid),
                    status_code=429,
                    media_type="application/problem+json",
                )
                await response(scope, receive, send_wrapper)
            else:
                await self.app(scope, receive, send_wrapper)
        except Exception:
            self.logger.exception(
                "request failed path=%s",
                path,
                extra={"service": self.service, "environment": self.environment},
            )
            response = JSONResponse(
                problem("internal-error", "Internal error", "The request could not be completed.", 500, cid),
                status_code=500,
                media_type="application/problem+json",
            )
            await response(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            self.metrics.observe_request(scope.get("method", "GET"), path, status_holder["status"], elapsed)
            self.logger.info(
                "request completed path=%s status=%s latency_ms=%.2f correlation_id=%s",
                path,
                status_holder["status"],
                elapsed * 1000,
                cid,
                extra={"service": self.service, "environment": self.environment},
            )
            correlation_id.reset(cid_token)
            trace_id.reset(tid_token)


def problem(type_slug: str, title: str, detail: str, status: int, cid: str) -> dict[str, Any]:
    return {
        "type": f"https://architecture.local/problems/{type_slug}",
        "title": title,
        "status": status,
        "detail": detail,
        "correlationId": cid,
    }


def error_response(exc: AppError) -> Response:
    body = problem(exc.type_slug, exc.title, exc.detail, exc.status, correlation_id.get())
    return Response(content=json.dumps(body), status_code=exc.status, media_type="application/problem+json")


def require_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError) as exc:
        raise AppError(
            400,
            "Invalid idempotency key",
            f"{field_name} must be a UUID.",
            "invalid-idempotency-key",
        ) from exc
    return str(parsed)


def sse(event_id: int, event: str, data: dict[str, Any]) -> str:
    return f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def load_vault_key(path: str | None, *, app_env: str, env_key: str | None) -> Any:
    """Resolve the provider key. Production reads the vault file and ignores the environment."""

    from pydantic import SecretStr

    file_key: SecretStr | None = None
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            if app_env == "prod":
                raise AppError(
                    500,
                    "Vault unavailable",
                    "The configured vault secret file does not exist.",
                    "vault-missing",
                ) from exc
        except json.JSONDecodeError as exc:
            raise AppError(500, "Vault unreadable", "The vault secret file is not valid JSON.", "vault-invalid") from exc
        else:
            raw = payload.get("llm_api_key") or ""
            if raw:
                file_key = SecretStr(raw)
    if app_env == "prod":
        if env_key:
            logging.getLogger("secrets").warning("ignoring environment model key because APP_ENV=prod")
        if file_key is None:
            raise AppError(
                500,
                "Vault key missing",
                "Production requires llm_api_key in the vault secret file.",
                "vault-key-missing",
            )
        return file_key
    if file_key is not None:
        return file_key
    if env_key:
        return SecretStr(env_key)
    return None
