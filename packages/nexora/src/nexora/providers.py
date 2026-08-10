"""Ordered provider selection for LangChain chat models.

The wrapper preserves LangChain's model and message interfaces. Nexora only owns the selection
policy: which already-constructed model is tried, and when moving to the next one is safe.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Literal

ProviderErrorKind = Literal[
    "rate_limit", "authentication", "server", "network", "abort", "unknown"
]
OnFallback = Callable[[str, str, ProviderErrorKind, Exception], Awaitable[None] | None]
OnAuthError = Callable[[str], Awaitable[bool] | bool]

__all__ = [
    "FallbackChatModel",
    "ModelProvider",
    "ProviderErrorKind",
    "classify_provider_error",
]


@dataclass(frozen=True, slots=True)
class ModelProvider:
    """A named LangChain chat model in fallback priority order."""

    name: str
    model: Any


def classify_provider_error(error: Exception) -> ProviderErrorKind:
    """Classify structured provider errors without binding to one SDK."""
    if type(error).__name__.lower() in {"aborterror", "cancellederror"}:
        return "abort"
    status = _status_code(error)
    message = str(error).lower()
    if status == 429 or "rate limit" in message or "too many requests" in message:
        return "rate_limit"
    if status in {401, 403} or any(
        marker in message
        for marker in ("unauthorized", "invalid api key", "authentication")
    ):
        return "authentication"
    if status is not None and status >= 500:
        return "server"
    if any(
        marker in message
        for marker in (
            "econnrefused",
            "enotfound",
            "timeout",
            "network",
            "econnreset",
            "socket",
            "fetch failed",
            "connection",
        )
    ) or getattr(error, "provider_error", False):
        return "network"
    return "unknown"


class FallbackChatModel:
    """Try named LangChain models in order without translating their messages or chunks.

    Same-provider retries happen only before a chunk becomes visible. Once streaming has started,
    an error is propagated because trying another provider would duplicate visible output.
    """

    def __init__(
        self,
        providers: list[ModelProvider],
        *,
        on_fallback: OnFallback | None = None,
        on_auth_error: OnAuthError | None = None,
        rate_limit_retry_seconds: float = 1.0,
        transient_retry_seconds: float = 0.5,
        transient_retry_max_attempts: int = 2,
    ) -> None:
        """Configure ordered candidates and bounded same-provider recovery."""
        if not providers:
            raise ValueError("FallbackChatModel requires at least one provider")
        if min(
            rate_limit_retry_seconds,
            transient_retry_seconds,
            transient_retry_max_attempts,
        ) < 0:
            raise ValueError("fallback retry settings cannot be negative")
        self._providers = list(providers)
        self._on_fallback = on_fallback
        self._on_auth_error = on_auth_error
        self._rate_limit_retry_seconds = rate_limit_retry_seconds
        self._transient_retry_seconds = transient_retry_seconds
        self._transient_retry_max_attempts = transient_retry_max_attempts

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Stable identity used by the durable model request key."""
        return {
            "providers": [
                {
                    "name": entry.name,
                    "model": getattr(entry.model, "_identifying_params", {}),
                    "class": f"{type(entry.model).__module__}.{type(entry.model).__qualname__}",
                }
                for entry in self._providers
            ]
        }

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FallbackChatModel":
        """Bind the same LangChain tool declarations to every candidate model."""
        return FallbackChatModel(
            [
                ModelProvider(entry.name, entry.model.bind_tools(tools, **kwargs))
                for entry in self._providers
            ],
            on_fallback=self._on_fallback,
            on_auth_error=self._on_auth_error,
            rate_limit_retry_seconds=self._rate_limit_retry_seconds,
            transient_retry_seconds=self._transient_retry_seconds,
            transient_retry_max_attempts=self._transient_retry_max_attempts,
        )

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream from the first healthy provider and fall through on pre-output failure."""
        last_error: Exception | None = None
        for index, entry in enumerate(self._providers):
            rate_limit_retried = False
            auth_retried = False
            transient_retries = 0
            while True:
                received_any = False
                try:
                    async for chunk in entry.model.astream(messages, **kwargs):
                        received_any = True
                        yield chunk
                    if received_any or index == len(self._providers) - 1:
                        return
                    await self._announce(
                        entry,
                        self._providers[index + 1],
                        "unknown",
                        RuntimeError("empty response"),
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
                    kind = classify_provider_error(error)
                    if kind == "abort" or received_any:
                        raise
                    if (
                        kind == "rate_limit"
                        and self._rate_limit_retry_seconds > 0
                        and not rate_limit_retried
                    ):
                        rate_limit_retried = True
                        await asyncio.sleep(self._rate_limit_retry_seconds)
                        continue
                    if kind == "authentication" and self._on_auth_error and not auth_retried:
                        auth_retried = True
                        refreshed = self._on_auth_error(entry.name)
                        if isawaitable(refreshed):
                            refreshed = await refreshed
                        if refreshed:
                            continue
                    if (
                        kind in {"network", "server"}
                        and self._transient_retry_seconds > 0
                        and transient_retries < self._transient_retry_max_attempts
                    ):
                        transient_retries += 1
                        await asyncio.sleep(self._transient_retry_seconds * transient_retries)
                        continue
                    if index == len(self._providers) - 1:
                        raise
                    await self._announce(entry, self._providers[index + 1], kind, error)
                    break
        if last_error is not None:
            raise last_error

    async def _announce(
        self,
        source: ModelProvider,
        target: ModelProvider,
        kind: ProviderErrorKind,
        error: Exception,
    ) -> None:
        if self._on_fallback is None:
            return
        result = self._on_fallback(source.name, target.name, kind, error)
        if isawaitable(result):
            await result


def _status_code(error: Exception) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        value = body.get("status") or body.get("status_code")
        if isinstance(value, int):
            return value
    return None
