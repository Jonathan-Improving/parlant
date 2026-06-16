# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, cast

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.loop.base_loop import _LoopState, _PROVIDER_DATA_KEY
from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
    Usage,
)
from parlant.core.sessions import EventKind, EventSource, ToolEventData
from parlant.core.tools import ToolId, ToolResult
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import create_engine_context


def _make_blocking_loop() -> BlockingLoop:
    # _update_message only touches the session event emitter and the loop state,
    # so the heavier collaborators aren't exercised here.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    return BlockingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        optimization_policy=cast(Any, None),
        react=cast(Any, None),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, None),
        hooks=EngineHooks(),
    )


class _NoReplayReact:
    """Stands in for a generator that can't natively replay a stored tool turn
    (e.g. Gemini when the persisted thought_signature is unusable)."""

    def deserialize_tool_messages(self, deserializer: Any) -> None:
        return None


class _NoopToolMessageReact:
    def serialize_tool_messages(self, messages: list[Message], serializer: Any) -> None:
        return None


class _StubToolRunner:
    async def run_tool(self, context: Any, tool_id: ToolId, args: dict[str, Any]) -> ToolResult:
        return ToolResult(data={"ok": True})


def test_that_an_unreplayable_tool_event_is_rendered_as_a_result_not_dropped() -> None:
    # A prior-turn tool event whose provider blob can't be natively replayed must
    # NOT be discarded — that would make the model forget the tool's data across
    # turns. Fall back to a result-only rendering so the data survives.
    loop = _make_blocking_loop()
    loop._react = cast(Any, _NoReplayReact())

    data: ToolEventData = {
        "tool_calls": [
            {
                "tool_id": "get_order_details",
                "rationale": "",
                "arguments": {"order_id": "#W2378156"},
                "result": cast(Any, {"data": {"status": "delivered"}, "metadata": {}}),
            }
        ]
    }
    metadata = {_PROVIDER_DATA_KEY: {"provider": "gemini", "model": "stale-model"}}

    messages = loop._build_tool_event_messages(data, metadata, "sess.compass")

    assert len(messages) == 1
    assert messages[0].role == Role.TOOL
    result_part = cast(ToolResultPart, messages[0].parts[0])
    # Rendered as a result (not dropped); the result data is carried through, whatever
    # the exact rendering (raw value or a descriptive string).
    assert "delivered" in str(result_part.content)


async def test_that_tool_call_step_is_committed_before_tool_results_are_appended() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(tool_ids_by_name={"test_tool": ToolId("local", "test_tool")})

    loop = _make_blocking_loop()
    loop._react = cast(Any, _NoopToolMessageReact())
    loop._tool_runner = cast(Any, _StubToolRunner())
    state = _LoopState()

    tool_call = ToolCallPart(
        id="call-1",
        name="test_tool",
        args={"value": 1},
    )
    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[tool_call]),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    committed = await loop._update_tool_calls(context, state, StepCompleted(result=result))

    assert committed is True
    assert [m.role for m in state.history] == [Role.ASSISTANT, Role.TOOL]
    assert state.history[0].tool_calls == [tool_call]
    assert state.history[1].tool_results[0].content == {"ok": True}
    assert state.steps == [result]


async def test_that_blocking_loop_emits_a_single_complete_message_event_without_chunks() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()

    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello there!")]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    # Block mode still receives the provider's text deltas; it just doesn't emit
    # them incrementally — the whole message is emitted once on step completion.
    events = [TextDelta(text="Hello "), TextDelta(text="there!"), StepCompleted(result=result)]
    for event in events:
        await loop._update_message(context, state, event)
        await loop._commit_new_event(state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    # Exactly one message event, carrying the full text and NO `chunks` key (so
    # consumers render it as a complete, non-streamed message).
    assert len(message_events) == 1
    data = cast(dict[str, Any], message_events[0].data)
    assert data["message"] == "Hello there!"
    assert "chunks" not in data

    # A completed message with no tool calls ends the loop.
    assert context.state.prepared_to_respond is True


async def test_that_text_before_and_after_a_tool_call_in_one_step_become_separate_messages() -> (
    None
):
    # TEXT, TOOL, TEXT within ONE step: the pre-tool and post-tool text must be TWO
    # separate message events. Block mode emits result.message.text once, and
    # TurnBuilder folds both segments into a single TextPart, so they arrive glued
    # ("flights.There are") with no break between them.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()

    # As TurnBuilder assembles it: both text segments fold into ONE TextPart, with
    # the tool call sitting after them in part order.
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[
                TextPart(text="Let me search for direct flights. There are no direct flights."),
                ToolCallPart(id="call-1", name="search_flights"),
            ],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="Let me search for direct flights. "),
        ToolCallStarted(id="call-1", name="search_flights"),
        TextDelta(text="There are no direct flights."),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._update_message(context, state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_texts = [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]

    assert message_texts == [
        "Let me search for direct flights. ",
        "There are no direct flights.",
    ]
