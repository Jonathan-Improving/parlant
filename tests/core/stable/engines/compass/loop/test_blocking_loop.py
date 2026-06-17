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

from dataclasses import replace
from typing import Any, cast

from parlant.core.agents import Effort
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.loop.base_loop import _LoopState, _PROVIDER_DATA_KEY
from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop
from parlant.core.engines.compass.loop.loop import LoopJob
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

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_agent,
    create_engine_context,
)


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


class _EmptyThenMessageReact:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if len(self.calls) == 1:
            yield StepCompleted(
                result=StepResult(
                    message=Message(role=Role.ASSISTANT),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )
        else:
            yield StepCompleted(
                result=StepResult(
                    message=Message(
                        role=Role.ASSISTANT,
                        parts=[
                            TextPart(
                                text="I'm sorry, I'm not able to help with that right now."
                            )
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )


class _RejectingReviewer:
    async def review_tool_calls(
        self,
        context: Any,
        reasoning: str,
        tool_calls: list[ToolCallPart],
    ) -> Any:
        return type(
            "ReviewResult",
            (),
            {
                "todo": "",
                "adjusted_reasoning": "Ask the user for the missing confirmation instead.",
            },
        )()


class _RejectedToolsThenMessageReact:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def stream_step(
        self,
        *,
        history: list[Message],
        tools: list[Any],
        tool_choice: str,
        reasoning: Any,
        hints: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "history": list(history),
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )

        if tool_choice == "none":
            yield StepCompleted(
                result=StepResult(
                    message=Message(
                        role=Role.ASSISTANT,
                        parts=[
                            TextPart(
                                text="I'm sorry, I'm not able to help with that right now."
                            )
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                    usage=Usage(),
                )
            )
            return

        tool_call = ToolCallPart(id="call-1", name="charge_card", args={})
        yield ToolCallStarted(id=tool_call.id, name=tool_call.name)
        yield StepCompleted(
            result=StepResult(
                message=Message(role=Role.ASSISTANT, parts=[tool_call]),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=Usage(),
            )
        )


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


async def test_that_max_engine_iterations_forces_a_final_message_with_tools_disabled() -> None:
    agent = replace(create_agent(), max_engine_iterations=1)
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "please help")],
        agent=agent,
    )
    context.state = ResponseState()

    react = _EmptyThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(react.calls) == 2
    assert react.calls[0]["tool_choice"] == "auto"
    assert react.calls[1]["tool_choice"] == "none"
    assert react.calls[1]["tools"] == []
    assert any(
        "You must now explain to the user" in message.text
        for message in react.calls[1]["history"]
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == (
        "I'm sorry, I'm not able to help with that right now."
    )


async def test_that_max_semantic_failures_force_a_final_message_with_tools_disabled() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "please charge it")])
    context.state = ResponseState(agent_effort=Effort.HIGH)

    react = _RejectedToolsThenMessageReact()
    loop = _make_blocking_loop()
    loop._react = cast(Any, react)
    loop._reviewer = cast(Any, _RejectingReviewer())

    await loop.run(LoopJob(context=context, system_instructions="SYSTEM"))

    assert len(react.calls) == 6
    assert [call["tool_choice"] for call in react.calls[:-1]] == ["auto"] * 5
    assert react.calls[-1]["tool_choice"] == "none"
    assert react.calls[-1]["tools"] == []
    assert any(
        "Tool use is disabled for this step" in message.text
        for message in react.calls[-1]["history"]
    )

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]
    assert cast(dict[str, Any], message_events[-1].data)["message"] == (
        "I'm sorry, I'm not able to help with that right now."
    )


async def test_that_text_after_a_tool_call_in_one_step_is_suppressed_after_the_preamble() -> None:
    # TEXT, TOOL, TEXT within ONE step: once the pre-tool preamble is surfaced,
    # further text in that same tool-call step should be suppressed so the user
    # doesn't receive a chain of progress updates before tools actually run.
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

    assert message_texts == ["Let me search for direct flights. "]


async def test_that_subsequent_blocking_tool_preambles_are_suppressed_and_not_committed() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()
    loop._mark_user_visible_message_emitted(state)

    tool_call = ToolCallPart(id="call-1", name="search_flights")
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[TextPart(text="I'll check another airport."), tool_call],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="I'll check another airport."),
        ToolCallStarted(id="call-1", name="search_flights"),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._update_message(context, state, event)
        await loop._commit_new_event(state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert message_events == []
    assert state.history[-1].text == ""
    assert state.history[-1].tool_calls == [tool_call]


async def test_that_blocking_tool_preambles_are_allowed_after_ten_seconds() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()
    loop._mark_user_visible_message_emitted(state)
    assert state.last_user_visible_message_at is not None
    state.last_user_visible_message_at -= loop._TOOL_PREAMBLE_INTERVAL_SECONDS + 1

    tool_call = ToolCallPart(id="call-1", name="search_flights")
    result = StepResult(
        message=Message(
            role=Role.ASSISTANT,
            parts=[TextPart(text="I'll check another airport."), tool_call],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(),
    )

    events = [
        TextDelta(text="I'll check another airport."),
        ToolCallStarted(id="call-1", name="search_flights"),
        StepCompleted(result=result),
    ]
    for event in events:
        await loop._update_message(context, state, event)
        await loop._commit_new_event(state, event)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    assert len(message_events) == 1
    assert cast(dict[str, Any], message_events[0].data)["message"] == (
        "I'll check another airport."
    )
    assert state.history[-1].text == "I'll check another airport."
    assert state.history[-1].tool_calls == [tool_call]
