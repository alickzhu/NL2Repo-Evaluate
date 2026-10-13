"""Incremental validation for this SGLang OpenAI SSE protocol, without token edits."""

import json
import re


class IncompleteStream(RuntimeError):
    pass


class SSEGate:
    """Hold terminal events until every choice terminates and wire [DONE] arrives.

    Partial deltas may be displayed, but callers must discard an attempt on error.
    Keeping terminal events back also prevents SDKs accepting an early finish.
    """

    def __init__(self, choices=1):
        if not isinstance(choices, int) or choices < 1:
            raise ValueError("Invalid choice count")
        self.expected = set(range(choices))
        self.finished = {}
        self.buffer = b""
        self.held = []
        self.held_bytes = 0
        self.done = False
        self.stream_id = None

    def feed(self, chunk):
        self.buffer += chunk
        output = []
        while True:
            boundary = re.search(rb"\r?\n\r?\n", self.buffer)
            if boundary is None:
                break
            raw, self.buffer = (
                self.buffer[: boundary.end()],
                self.buffer[boundary.end() :],
            )
            lines = raw.replace(b"\r\n", b"\n").split(b"\n")
            data = b"\n".join(
                line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:")
            )
            if not data:
                if not self.done:
                    output.append(
                        raw
                    )  # Heartbeats can pass even while terminal events wait.
                continue
            if self.done:
                raise IncompleteStream("Data after [DONE]")
            if data == b"[DONE]":
                if set(self.finished) != self.expected:
                    raise IncompleteStream("[DONE] without terminal choice event")
                self.done = True
                output.extend(self.held)
                self.held.clear()
                output.append(raw)
                continue
            try:
                event = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise IncompleteStream("Malformed SSE JSON") from exc
            if not isinstance(event, dict) or event.get("error"):
                raise IncompleteStream("Upstream SSE error")
            sid = event.get("id")
            if sid:
                if self.stream_id and sid != self.stream_id:
                    raise IncompleteStream("Stream identity changed")
                self.stream_id = sid
            choices = event.get("choices")
            if not isinstance(choices, list):
                raise IncompleteStream("Missing choices in SSE event")
            terminal = False
            for choice in choices:
                if not isinstance(choice, dict):
                    raise IncompleteStream("Invalid choice")
                index = choice.get("index")
                if index not in self.expected:
                    raise IncompleteStream("Unexpected choice index")
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in {
                        "stop",
                        "length",
                        "tool_calls",
                        "function_call",
                        "content_filter",
                    }:
                        raise IncompleteStream("Unknown terminal finish reason")
                    if index in self.finished:
                        raise IncompleteStream("Duplicate terminal choice")
                    self.finished[index] = reason
                    terminal = True
                elif index in self.finished:
                    raise IncompleteStream("Choice data after termination")
            if terminal or self.held:
                self.held.append(raw)
                self.held_bytes += len(raw)
                if self.held_bytes > 4 * 1024 * 1024:
                    raise IncompleteStream("Excessive terminal metadata")
            else:
                output.append(raw)
        if len(self.buffer) > 4 * 1024 * 1024:
            raise IncompleteStream("Oversized or unterminated SSE event")
        return output

    def finish(self):
        if not self.done or self.buffer.strip():
            raise IncompleteStream(
                "Stream ended without complete terminal event and [DONE]"
            )
