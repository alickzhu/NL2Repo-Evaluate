"""Real SDK streaming regression with an isolated, truncating HTTP origin."""

import asyncio
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.exceptions.types import LLMServiceUnavailableError

from openhands_research.model_profile import ModelProfile
from openhands_research.sft_transport import GUARDED_MODELS, SFTTransportLLM


def event(delta, finish=None):
    return (
        "data: "
        + json.dumps(
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )
        + "\n\n"
    ).encode()


class SFTTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.always_fail = False

        async def handler(req):
            raw = await req.read()
            assert json.loads(raw)["stream"]
            self.requests.append(
                (hashlib.sha256(raw).hexdigest(), req.headers.get("X-Request-ID"))
            )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(req)
            if self.always_fail or len(self.requests) == 1:
                await response.write(
                    event(
                        {
                            "content": "REJECTED_PARTIAL",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "bad",
                                    "type": "function",
                                    "function": {
                                        "name": "must_not_run",
                                        "arguments": '{"partial":',
                                    },
                                }
                            ],
                        }
                    )
                )
            else:
                await response.write(event({"content": "ACCEPTED"}))
                await response.write(event({}, "stop"))
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.server = TestServer(app)
        await self.server.start_server()

    async def asyncTearDown(self):
        await self.server.close()

    def llm(self, retries=2):
        return SFTTransportLLM(
            model="openai/" + min(GUARDED_MODELS),
            base_url=str(self.server.make_url("/v1")),
            api_key="EMPTY",
            stream=True,
            num_retries=retries,
            retry_min_wait=0,
            retry_max_wait=0,
            retry_multiplier=0,
            max_input_tokens=262144,
            max_output_tokens=128,
            log_completions=False,
        )

    async def test_retry_discards_partial_assistant_and_tools(self):
        seen = []
        llm = self.llm()
        with patch.object(SFTTransportLLM, "get_token_count", return_value=100):
            response = await asyncio.to_thread(
                llm.completion,
                messages=[Message(role="user", content=[TextContent(text="fixture")])],
                on_token=seen.append,
            )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertTrue(self.requests[0][1].startswith("generation-"))
        self.assertEqual(response.raw_response.choices[0].message.content, "ACCEPTED")
        self.assertNotIn("REJECTED_PARTIAL", json.dumps([c.model_dump() for c in seen]))
        self.assertNotIn("must_not_run", json.dumps([c.model_dump() for c in seen]))

    async def test_async_all_fail_publishes_nothing(self):
        self.always_fail = True
        seen = []
        with (
            patch.object(SFTTransportLLM, "get_token_count", return_value=100),
            self.assertRaises(LLMServiceUnavailableError),
        ):
            await self.llm().acompletion(
                messages=[Message(role="user", content=[TextContent(text="fixture")])],
                on_token=seen.append,
            )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(seen, [])

    def test_only_target_models_use_guard(self):
        self.assertIn("autonl2repo-qwen3.5-35b-a3b-v17-step288", GUARDED_MODELS)
        self.assertIn("autonl2repo-qwen3.5-35b-a3b-base-v22-step1350", GUARDED_MODELS)
        for model in GUARDED_MODELS:
            p = ModelProfile(endpoint="http://127.0.0.1:1234/v1", api_model=model)
            self.assertIsInstance(
                p.build_agent_llm(Path("/tmp/test-unused"), "EMPTY"), SFTTransportLLM
            )
        p = ModelProfile(endpoint="http://127.0.0.1:1234/v1")
        self.assertNotIsInstance(
            p.build_agent_llm(Path("/tmp/test-unused"), "EMPTY"), SFTTransportLLM
        )
