"""Opt-in raw SSE validation BEFORE OpenAI/LiteLLM normalizes stream endings.

Pass the returned client through LLM.completion(..., client=client). Configure
bounded retries on LLM, not both SDKs. Partial on_token callbacks are display-only;
only a successfully returned completion may enter history or execute a tool.
"""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from ._sse_integrity import IncompleteStream, SSEGate


class CheckedBytes(httpx.SyncByteStream):
    def __init__(self, source: httpx.SyncByteStream, request: httpx.Request, n: int):
        self.source, self.request, self.gate = source, request, SSEGate(n)

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self.source:
                yield from self.gate.feed(chunk)
                if self.gate.done:
                    break
            self.gate.finish()
        except IncompleteStream as exc:
            raise httpx.RemoteProtocolError(str(exc), request=self.request) from exc
        finally:
            self.source.close()

    def close(self) -> None:
        self.source.close()


class CheckedAsyncBytes(httpx.AsyncByteStream):
    def __init__(self, source: httpx.AsyncByteStream, request: httpx.Request, n: int):
        self.source, self.request, self.gate = source, request, SSEGate(n)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.source:
                for frame in self.gate.feed(chunk):
                    yield frame
                if self.gate.done:
                    break
            self.gate.finish()
        except IncompleteStream as exc:
            raise httpx.RemoteProtocolError(str(exc), request=self.request) from exc
        finally:
            await self.source.aclose()

    async def aclose(self) -> None:
        await self.source.aclose()


def choice_count(request: httpx.Request, response: httpx.Response) -> int | None:
    if response.status_code != 200:
        return None
    if request.url.path not in ["/v1/chat/completions", "/v1/completions"]:
        return None
    payload = json.loads(request.content)
    if not payload.get("stream"):
        return None
    if "text/event-stream" not in response.headers.get("content-type", ""):
        raise httpx.RemoteProtocolError("Expected SSE response", request=request)
    return payload.get("n", 1)


class CheckedTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.transport = transport or httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.transport.handle_request(request)
        try:
            n = choice_count(request, response)
            if n is not None:
                assert isinstance(response.stream, httpx.SyncByteStream)
                response.stream = CheckedBytes(response.stream, request, n)
            return response
        except Exception:
            response.close()
            raise

    def close(self) -> None:
        self.transport.close()


class CheckedAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.transport.handle_async_request(request)
        try:
            n = choice_count(request, response)
            if n is not None:
                assert isinstance(response.stream, httpx.AsyncByteStream)
                response.stream = CheckedAsyncBytes(response.stream, request, n)
            return response
        except Exception:
            await response.aclose()
            raise

    async def aclose(self) -> None:
        await self.transport.aclose()


def client(base_url: str, **kwargs: Any) -> OpenAI:
    headers = {"Accept-Encoding": "identity", **kwargs.pop("default_headers", {})}
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        default_headers=headers,
        http_client=httpx.Client(transport=CheckedTransport(), timeout=1800),
        **kwargs,
    )


def async_client(base_url: str, **kwargs: Any) -> AsyncOpenAI:
    headers = {"Accept-Encoding": "identity", **kwargs.pop("default_headers", {})}
    return AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        default_headers=headers,
        http_client=httpx.AsyncClient(transport=CheckedAsyncTransport(), timeout=1800),
        **kwargs,
    )
