"""Exercise the real SDK/LiteLLM wire protocol against a local fake provider."""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import resolve_tool

from openhands_research.model_profile import ModelProfile
from openhands_research.responses_records import projection
from openhands_research.runtime import TaskContainer
from openhands_research.tools import build_task_tools
from openhands_research.trajectories import audit_records, sft_records


def profile(endpoint):
    return ModelProfile(
        endpoint,
        model="gpt-5.6-luna",
        model_revision="endpoint-defined",
        api_mode="responses",
        temperature=None,
        seed=None,
        reasoning_effort="high",
        request_retries=0,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming, subscriber, usage", [
    (False, False, "solver"),
    (True, True, "solver"),
    (True, False, "solver"),
    (True, False, "condenser"),
])
@pytest.mark.parametrize("truncated", [0, 3])
@pytest.mark.asyncio
async def test_native_responses_roundtrip(
    tmp_path, asynchronous, streaming, subscriber, usage, truncated
):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            index = len(received)
            is_length = index <= truncated
            output = [
                {
                    "id": "rs_one",
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Visible reasoning summary"}
                    ],
                    "content": [],
                    "encrypted_content": "opaque_provider_value",
                },
                {
                    "id": "fc_one",
                    "type": "function_call",
                    "call_id": "call_one",
                    "name": "terminal",
                    "arguments": '{"command":"pwd", "summary":"Inspect working directory"}',
                    "status": "completed",
                },
            ]
            if index > truncated + 1:
                output = [
                    {
                        "id": "msg_two",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Finished",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            payload = {
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.6-luna",
                "status": "incomplete" if is_length else "completed",
                "output": [] if is_length else output,
                "incomplete_details": {"reason": "max_output_tokens"}
                if is_length
                else None,
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            }
            if body.get("stream"):
                kind = "response.incomplete" if is_length else "response.completed"
                wire = (
                    "event: "
                    + kind
                    + "\ndata: "
                    + json.dumps(
                        {"type": kind, "sequence_number": 0, "response": payload}
                    )
                    + "\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                wire, content_type = json.dumps(payload).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model_profile = profile(f"http://127.0.0.1:{server.server_port}/v1")
        llm = (model_profile.build_agent_llm(tmp_path, "dummy") if usage == "solver"
               else model_profile.build_condenser(tmp_path, "dummy").llm)
        assert llm.stream
        if not streaming:
            llm = llm.model_copy(update={"stream": False})
        import docker

        client = MagicMock(spec=docker.DockerClient)
        client.api = MagicMock(spec=docker.APIClient)
        client.api.base_url = "http+docker://localhost"
        target = TaskContainer("cached", "native-test", "task", client=client)
        tool_definitions = [
            resolve_tool(spec, MagicMock())[0] for spec in build_task_tools(target)
        ]
        history = [
            Message(
                role="system",
                content=[TextContent(text="Private controller instruction")],
            ),
            Message(role="user", content=[TextContent(text="Implement README.md")]),
        ]
        chunks = []

        async def call():
            if asynchronous:
                return await llm.aresponses(
                    history,
                    tools=tool_definitions,
                    on_token=chunks.append if subscriber else None,
                )
            return llm.responses(
                history,
                tools=tool_definitions,
                on_token=chunks.append if subscriber else None,
            )

        first = await call()
        condenser = profile(
            f"http://127.0.0.1:{server.server_port}/v1"
        ).build_condenser(tmp_path, "dummy")
        summary_input = "\n".join(
            condenser._event_strings(
                [MessageEvent(source="agent", llm_message=first.message)]
            )
        )
        assert "Visible reasoning summary" in summary_input
        assert "opaque_provider_value" not in summary_input
        assert "<think>" not in summary_input
        history += [
            first.message,
            Message(
                role="tool",
                tool_call_id="call_one",
                name="terminal",
                content=[TextContent(text="/workspace")],
            ),
        ]
        final = await call()
        assert final.message.content[0].text == "Finished"
        assert len(received) == truncated + 2
        assert all(bool(body.get("stream")) == streaming for _, body in received)
        assert all(path == "/v1/responses" for path, _ in received)
        assert all(body["model"] == "gpt-5.6-luna" for _, body in received)
        for _, body in received:
            assert body["store"] is False
            assert (
                not {
                    "messages",
                    "seed",
                    "temperature",
                    "chat_template_kwargs",
                    "max_completion_tokens",
                }
                & body.keys()
            )
            assert body["reasoning"]["effort"] == "high"
            assert body["tools"][0]["type"] == "function"
            assert body["tools"][0]["name"] == "terminal"
            assert (
                "parameters" in body["tools"][0] and "function" not in body["tools"][0]
            )
        assert [b["max_output_tokens"] for _, b in received] == (
            [32768] * 3 + [65536, 32768] if truncated else [32768] * 2
        )
        sent = received[-1][1]["input"]
        assert any(
            i.get("type") == "function_call_output" and i["call_id"] == "call_one"
            for i in sent
        )
        assert any(
            i.get("type") == "function_call" and i["call_id"] == "call_one"
            for i in sent
        )
        assert any(
            i.get("type") == "reasoning"
            and i.get("encrypted_content") == "opaque_provider_value"
            for i in sent
        )
        logs = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
        logs.sort(key=lambda r: r["timestamp"])
        assert len(logs) == len(received)
        for n, (record, (_, body)) in enumerate(zip(logs, received)):
            assert record["llm_path"] == "responses"
            assert record["input"] == body["input"]
            record.update(
                sequence=n,
                usage_id="solver",
                source_log=f"{n}.json",
                request_sha256="test",
            )
        events = [
            {"event_type": "ActionEvent", "event": {"llm_response_id": first.id}},
            {"event_type": "MessageEvent", "event": {"llm_response_id": final.id}},
        ]
        audit = audit_records(logs, events)
        assert audit["ok"], audit
        assert audit["native_history_items_checked"] == 2
        corrupted = copy.deepcopy(logs)
        for item in corrupted[-1]["input"]:
            if item.get("type") == "function_call":
                item["arguments"] = '{"command":"tampered"}'
        assert not audit_records(corrupted, events)["ok"]
        rows = sft_records(logs, events)
        assert len(rows) == 2
        assert rows[0]["api_mode"] == "responses"
        assert "<think>" not in json.dumps(rows)
        assert rows[0]["message_loss_mask"] == [0, 0, 1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_projection_preserves_native_records_and_empty_length_response():
    raw = {
        "llm_path": "responses",
        "input": "hi",
        "response": {
            "id": "r",
            "status": "incomplete",
            "output": [],
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }
    original = copy.deepcopy(raw)
    assert projection(raw)["response"]["choices"][0]["finish_reason"] == "length"
    assert raw == original


def test_profile_rejects_chat_sampling_for_responses():
    with pytest.raises(ValueError, match="temperature and seed"):
        ModelProfile("http://localhost/v1", model="gpt-5.6-luna", api_mode="responses")
