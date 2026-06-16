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
from parlant.core.engines.compass.loop.base_loop import _LoopState
from parlant.core.engines.compass.loop.loop import LoopJob
from parlant.core.engines.compass.loop.streaming_loop import StreamingLoop
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
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
    Usage,
)
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import create_engine_context


def _make_streaming_loop() -> StreamingLoop:
    # _build_history only reads the LoopJob, so the heavier collaborators
    # (meter/optimization_policy/react/tool_runner) aren't exercised here.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    return StreamingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        optimization_policy=cast(Any, None),
        react=cast(Any, None),
        tool_runner=cast(Any, None),
        reviewer=cast(Any, None),
        hooks=EngineHooks(),
    )


async def test_that_turn_instructions_are_placed_before_the_last_customer_message() -> None:
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "buying a house for the first time, what do I need to know?"),
        ],
    )
    context.state = ResponseState()

    marker = "TURN_INSTRUCTIONS_MARKER_12345"

    async def turn_instructions(_: EngineContext) -> str:
        return marker

    job = LoopJob(
        context=context,
        system_instructions="SYSTEM_INSTRUCTIONS",
        step_instructions=turn_instructions,
    )

    history, instructions_index = await _make_streaming_loop()._build_history(job)

    # The model's most recent turn must be the customer's message — not the
    # imperative instructions note, which it otherwise tends to answer / echo.
    assert history[-1].role == Role.USER
    assert "buying a house" in history[-1].text

    # The turn instructions appear exactly once, immediately before that last
    # customer message, and _build_history reports their index (so the loop can
    # replace them in place when reevaluating).
    instruction_indices = [i for i, m in enumerate(history) if marker in m.text]
    assert instruction_indices == [len(history) - 2]
    assert instructions_index == len(history) - 2


async def test_that_reviewer_adjusted_reasoning_is_injected_even_without_step_instructions() -> (
    None
):
    # When step_instructions is None (e.g. low-effort agents) there is no turn-
    # instructions message in history. The reviewer's adjusted reasoning (step_notes)
    # must STILL reach the retried prompt — otherwise the retry re-streams the same
    # output and the review loop never converges.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState(step_notes="Do not call charge_card; ask for confirmation first.")

    loop = _make_streaming_loop()
    job = LoopJob(context=context, system_instructions="SYSTEM", step_instructions=None)

    history, instructions_index = await loop._build_history(job)
    assert instructions_index is None  # no step-instructions message exists for this config

    state = _LoopState(history=history, instructions_index=instructions_index)
    await loop._update_step_instructions(job, state)

    # The adjusted reasoning must be present in the (to-be-re-streamed) history.
    assert any("ask for confirmation first" in m.text for m in state.history)


async def test_that_tool_preamble_note_is_only_injected_until_a_message_is_visible() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "find flights")])
    context.state = ResponseState()

    loop = _make_streaming_loop()
    job = LoopJob(context=context, system_instructions="SYSTEM", step_instructions=None)

    history, instructions_index = await loop._build_history(job)
    state = _LoopState(history=history, instructions_index=instructions_index)

    await loop._update_step_instructions(job, state)
    assert any("Tool communication before tool use" in message.text for message in state.history)

    state.user_visible_message_emitted = True
    await loop._update_step_instructions(job, state)
    assert not any(
        "Tool communication before tool use" in message.text for message in state.history
    )


async def test_that_tool_preamble_note_is_injected_again_after_adjusted_reasoning() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "find flights")])
    context.state = ResponseState(step_notes="Ask for the missing airport before searching.")

    loop = _make_streaming_loop()
    job = LoopJob(context=context, system_instructions="SYSTEM", step_instructions=None)

    history, instructions_index = await loop._build_history(job)
    state = _LoopState(
        history=history,
        instructions_index=instructions_index,
        user_visible_message_emitted=True,
    )

    await loop._update_step_instructions(job, state)

    instructions_text = "\n".join(message.text for message in state.history)
    assert "Ask for the missing airport before searching." in instructions_text
    assert "Tool communication before tool use" in instructions_text


async def test_that_a_restarted_step_finalizes_and_resets_the_streamed_message() -> None:
    # When a step is restarted (reviewer rejection), the already-streamed preamble must
    # be finalized as its own message and the streaming state reset, so the retry begins
    # a fresh message instead of concatenating onto the rejected one.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    await loop._update_message(context, state, TextDelta(text="Let me check that for you."))
    assert state.message_buffer is not None

    await loop._reset_message_after_restart(context, state)
    assert state.message_handle is None
    assert state.message_buffer is None
    assert state.message_chunks == []
    assert state.in_the_middle_of_running_tools is False

    context.state.step_notes = "Ask for confirmation instead of calling the tool."

    # The retry's text starts a NEW message, not appended to the rejected preamble.
    await loop._update_message(context, state, TextDelta(text="Actually, here's the answer."))
    assert state.message_buffer is not None
    assert state.message_buffer.getvalue() == "Actually, here's the answer."

    # Two separate message events were emitted; the latest carries no preamble.
    emitter = cast(EventBuffer, context.session_event_emitter)
    message_texts = [
        cast(dict[str, Any], e.data)["message"]
        for e in emitter.events
        if e.kind == EventKind.MESSAGE
    ]
    assert "Let me check that for you." not in message_texts[-1]


async def test_that_text_after_a_tool_call_in_one_step_is_suppressed_after_the_preamble() -> None:
    # Within ONE step the model can emit text, call a tool, then emit more text
    # (TEXT, TOOL, TEXT). Once the pre-tool preamble is surfaced, further text in
    # that same tool-call step should be suppressed so the user doesn't receive a
    # chain of progress updates before tools actually run.
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState()

    events = [
        TextDelta(text="Let me search for direct flights. "),
        ToolCallStarted(id="call-1", name="search_flights"),
        TextDelta(text="There are no direct flights."),
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


async def test_that_subsequent_streamed_tool_preambles_are_suppressed_and_not_committed() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()
    loop = _make_streaming_loop()
    state = _LoopState(user_visible_message_emitted=True)

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
