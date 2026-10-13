"""Real LocalConversation/condenser calls must not downgrade at the HTTP wire."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openhands.sdk import Agent
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent

from openhands_research.model_profile import ModelProfile


@pytest.mark.parametrize("mode", ["chat", "responses"])
@pytest.mark.asyncio
async def test_conversation_and_summary_without_token_subscribers(tmp_path, mode):
    requests, events = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if not body.get("stream"):
                self.send_error(400, "production calls must use SSE")
                return
            if mode == "chat":
                payload = {
                    "id": "wire-chat", "object": "chat.completion.chunk",
                    "created": 0, "model": "qwen3.8-flash-next",
                    "choices": [{"index": 0, "finish_reason": None, "delta": {
                        "role": "assistant", "content": "Done.",
                        "reasoning_content": "WIRE_REASONING",
                    }}],
                }
                wire = "data: " + json.dumps(payload) + "\n\n"
                payload["choices"][0].update(delta={}, finish_reason="stop")
                wire += "data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n"
            else:
                payload = {"type": "response.completed", "sequence_number": 0,
                    "response": {
                        "id": "wire-response", "object": "response", "created_at": 0,
                        "model": "gpt-5.6-luna", "status": "completed",
                        "output": [{"id": "msg", "type": "message", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                                "text": "Done.", "annotations": []}]}],
                        "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
                    }}
                wire = "event: response.completed\ndata: " + json.dumps(payload) + "\n\n"
            raw = wire.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conversation = None
    try:
        options = dict(request_retries=1)
        if mode == "responses":
            options.update(model="gpt-5.6-luna", model_revision="endpoint-defined",
                           api_mode=mode, temperature=None, seed=None, reasoning_effort="high")
        else:
            options.update(api_model="qwen3.8-flash-next")
        profile = ModelProfile(f"http://127.0.0.1:{server.server_port}/v1", **options)
        conversation = LocalConversation(
            agent=Agent(llm=profile.build_agent_llm(tmp_path / "calls", "dummy"), tools=[]),
            workspace=str(tmp_path / "workspace"), persistence_dir=str(tmp_path / "conversation"),
            callbacks=[events.append], max_iteration_per_run=1, visualizer=None,
            profile_store_dir=tmp_path / "profiles",
            # Deliberately no token_callbacks or stream_callbacks, like runner.py.
        )
        conversation.send_message("Say Done.")
        await conversation.arun()
        assert len(requests) == 1
        assert events
        if mode == "chat":
            assert "WIRE_REASONING" in json.dumps([e.model_dump(mode="json") for e in events])
        condenser = profile.build_condenser(tmp_path / "summaries", "dummy")
        summary = await condenser._agenerate_condensation([
            MessageEvent(source="user", llm_message=Message(role="user", content=[TextContent(text="Summarize.")]))
        ], 0)
        assert summary.summary == "Done."
        assert len(requests) == 2
        assert all(r.get("stream") is True for r in requests)
    finally:
        if conversation:
            conversation.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
