"""Dependency-free ``web_fetch`` implementation (not web search)."""

from __future__ import annotations

import asyncio
import html
import math
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from ..workspace import ToolContext
from ._types import (
    BuiltinToolState,
    ToolResult,
    WebFetchResponse,
    WebFetchToolOptions,
    error_result,
    text_result,
)

DEFAULT_MAX_RESULT_CHARS = 30_000
ALLOWED_CONTENT_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
    "application/ld+json",
)


class UrllibWebFetchTransport:
    """Standard-library HTTP transport for ``web_fetch``."""

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        """Fetch a URL off the event loop and cap the response body."""

        def request() -> WebFetchResponse:
            req = urllib.request.Request(url, headers=dict(headers), method="GET")
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    response_headers = dict(response.headers.items())
                    return WebFetchResponse(
                        status=response.status,
                        reason=response.reason,
                        url=response.geturl(),
                        headers=response_headers,
                        body=response.read(max_bytes),
                    )
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"HTTP {error.code} {error.reason}") from error

        return await asyncio.to_thread(request)


async def web_fetch_tool(
    _call_id: str,
    arguments: object,
    context: ToolContext,
    state: BuiltinToolState,
    options: WebFetchToolOptions,
) -> ToolResult:
    """Fetch readable content, porting ``createWebFetchTool().execute``."""
    del context
    params = arguments if isinstance(arguments, dict) else {}
    raw_url = params.get("url")
    url = raw_url.strip() if isinstance(raw_url, str) else ""
    if not url:
        return error_result("url is required")
    normalized = _normalize_url(url)
    if isinstance(normalized, ToolResultError):
        return error_result(normalized.message)
    prompt_value = params.get("prompt")
    prompt = prompt_value.strip() if isinstance(prompt_value, str) else ""
    max_chars = _max_chars(params.get("max_chars"))
    cache_key = f"{normalized}::{prompt}"
    now = options.now()
    cached = state.web_cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return text_result(cached[1])
    if cached is not None:
        state.web_cache.pop(cache_key, None)

    try:
        transport = options.transport or UrllibWebFetchTransport()
        response = await transport.get(
            normalized,
            headers={
                "User-Agent": "Semora-WebFetch/1.0",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "text/plain;q=0.9,application/json;q=0.9,*/*;q=0.5"
                ),
            },
            timeout_seconds=options.fetch_timeout_ms / 1000,
            max_bytes=options.max_bytes,
        )
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as error:
        return error_result(f"web_fetch failed: {error}")
    if not 200 <= response.status < 300:
        return error_result(f"web_fetch failed: HTTP {response.status} {response.reason}")
    content_type = _header(response.headers, "content-type") or "text/plain"
    content_type = content_type.lower()
    if not any(content_type.startswith(prefix) for prefix in ALLOWED_CONTENT_PREFIXES):
        return error_result(f"web_fetch failed: Unsupported content-type: {content_type}")

    cleaned = _clean_content(_decode(response.body, content_type), content_type)
    if options.summarizer is not None and prompt:
        summary = await options.summarizer.summarize(cleaned, prompt)
        result_text = (
            f"URL: {response.url}\nContent-Type: {content_type}\n\n{summary.strip() or '(empty)'}"
        )
    else:
        result_text = _format_raw(response.url, content_type, cleaned, max_chars)
    state.web_cache[cache_key] = (now + options.cache_ttl_ms / 1000, result_text)
    return text_result(result_text)


class ToolResultError:
    """Internal URL validation outcome without raising past tool input handling."""

    def __init__(self, message: str) -> None:
        self.message = message


def _normalize_url(raw: str) -> str | ToolResultError:
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ToolResultError(f"Invalid URL: {raw}")
    if not parsed.scheme:
        return ToolResultError(f"Invalid URL: {raw}")
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    elif parsed.scheme != "https":
        return ToolResultError(f"Unsupported URL scheme: {parsed.scheme}:")
    if not parsed.netloc:
        return ToolResultError(f"Invalid URL: {raw}")
    return urlunsplit(parsed)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lower = name.lower()
    return next((value for key, value in headers.items() if key.lower() == lower), None)


def _decode(body: bytes, content_type: str) -> str:
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    charset = match.group(1) if match else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _clean_content(raw: str, content_type: str) -> str:
    if content_type.startswith(("text/html", "application/xhtml")):
        return _html_to_text(raw)
    return raw.replace("\r\n", "\n").strip()


def _html_to_text(value: str) -> str:
    text = re.sub(r"<!--[\s\S]*?-->", "", value)
    text = re.sub(
        r"<(script|style|noscript|template|svg)\b[^>]*>[\s\S]*?</\1>",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"</?(p|div|section|article|header|footer|li|tr|br|hr|h[1-6])\b[^>]*>",
        "\n",
        text,
        flags=re.I,
    )
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _format_raw(url: str, content_type: str, body: str, max_chars: int) -> str:
    truncated = len(body) > max_chars
    selected = body[:max_chars]
    header = f"URL: {url}\nContent-Type: {content_type}"
    if truncated:
        header += f"\nTruncated: yes ({max_chars} of {len(body)} chars)"
    return f"{header}\n\n{selected or '(empty)'}"


def _max_chars(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return DEFAULT_MAX_RESULT_CHARS
    return min(max(int(value), 500), 100_000)
