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

import asyncio
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
import json
import re
import time
from typing import Any, Optional, cast

from parlant.core.agents import Effort
from parlant.core.async_utils import safe_gather
from parlant.core.common import JSONSerializable
from parlant.core.emissions import MessageEventHandle, StatusEventHandle
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.compass.response_state import EngineContext, IterationState
from parlant.core.engines.compass.loop.loop import Loop, LoopJob, LoopResult
from parlant.core.engines.compass.tool_runner import ToolRunner
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import (
    Message,
    Part,
    ReactError,
    ReactGenerator,
    ReasoningDelta,
    Role,
    StepCompleted,
    StepResult,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolMessageDeserializer,
    ToolMessageSerializer,
    ToolResultPart,
    ToolSpec,
    Usage,
    tool_specs_from_tools,
)
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    StatusEventData,
    ToolCall,
    ToolEventData,
)
from parlant.core.tools import ToolResult
from parlant.core.tracer import Tracer
from parlant.core.engines.compass.reviewer import Reviewer


# Key under which the provider's opaque tool-call replay blob is stored in a tool
# event's metadata. Read back at history-build time so the originating provider
# can faithfully replay the call (ids, signatures, …) on a later turn.
_PROVIDER_DATA_KEY = "__provider_data__"


class SessionToolMessageSerializer(ToolMessageSerializer):
    """Session-backed sink. The neutral record (tool_id/args/result) is built by
    the engine from execution and stored in the event's ``data``, so the call/
    result writes are no-ops here; we only capture the provider blob for metadata."""

    def __init__(self) -> None:
        self.provider_data: Mapping[str, JSONSerializable] = {}

    def write_calls(self, calls: Sequence[ToolCallPart]) -> None:
        return None

    def write_results(self, results: Sequence[ToolResultPart]) -> None:
        return None

    def write_provider_data(self, data: Mapping[str, Any]) -> None:
        self.provider_data = cast(Mapping[str, JSONSerializable], dict(data))


class SessionToolMessageDeserializer(ToolMessageDeserializer):
    """Session-backed source. Serves calls/results from the neutral ``ToolEventData``
    and the provider blob from the event's metadata."""

    def __init__(self, data: ToolEventData, provider_data: Mapping[str, Any]) -> None:
        self._data = data
        self._provider_data = provider_data

    def read_calls(self) -> Sequence[ToolCallPart]:
        return [
            ToolCallPart(name=call["tool_id"], args=call["arguments"])
            for call in self._data["tool_calls"]
        ]

    def read_results(self) -> Sequence[ToolResultPart]:
        results: list[ToolResultPart] = []
        for call in self._data["tool_calls"]:
            result = call.get("result", {}) or {}
            results.append(
                ToolResultPart(
                    name=call["tool_id"],
                    content=result.get("data", {}),
                    is_error="error_details" in (result.get("metadata", {}) or {}),
                )
            )
        return results

    def read_provider_data(self) -> Mapping[str, Any]:
        return self._provider_data


class _ToolPreambleDecision(Enum):
    ALLOW_INITIAL = auto()
    ALLOW_INTERVAL_UPDATE = auto()
    SUPPRESS = auto()


@dataclass
class _ToolPreamblePolicy:
    user_visible_message_emitted: bool = False
    last_user_visible_message_at: float | None = None

    def decide(self, now: float, interval_seconds: float) -> _ToolPreambleDecision:
        if not self.user_visible_message_emitted or self.last_user_visible_message_at is None:
            return _ToolPreambleDecision.ALLOW_INITIAL

        if now - self.last_user_visible_message_at > interval_seconds:
            return _ToolPreambleDecision.ALLOW_INTERVAL_UPDATE

        return _ToolPreambleDecision.SUPPRESS

    def mark_user_visible_message_emitted(self, now: float) -> None:
        self.user_visible_message_emitted = True
        self.last_user_visible_message_at = now


class _MessageCommitMode(Enum):
    KEEP_FULL_MESSAGE = auto()
    KEEP_NO_TEXT = auto()
    KEEP_TEXT_PREFIX = auto()


@dataclass
class _MessageCommitPolicy:
    mode: _MessageCommitMode = _MessageCommitMode.KEEP_FULL_MESSAGE
    prefix_len: int = 0

    @classmethod
    def keep_full_message(cls) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_FULL_MESSAGE)

    @classmethod
    def keep_no_text(cls) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_NO_TEXT)

    @classmethod
    def keep_text_prefix(cls, prefix_len: int) -> "_MessageCommitPolicy":
        return cls(_MessageCommitMode.KEEP_TEXT_PREFIX, prefix_len)


@dataclass
class _MessageEmissionState:
    handle: MessageEventHandle | None = None
    buffer: StringIO | None = None
    chunks: list[str | None] = field(default_factory=list)
    # Chars of THIS step's message already emitted as their own (complete) bubbles
    # via interrupt-splits (text -> tool -> text). The step-completion emit covers
    # `result.message.text[emitted_len:]` — the authoritative remainder.
    emitted_len: int = 0
    commit_policy: _MessageCommitPolicy = field(
        default_factory=_MessageCommitPolicy.keep_full_message
    )

    def clear_transient_output(self) -> None:
        self.handle = None
        self.buffer = None
        self.chunks = []

    def reset_step_output(self) -> None:
        self.clear_transient_output()
        self.emitted_len = 0

    def suppress_tool_text(self) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_no_text()

    def keep_tool_text_prefix(self, prefix_len: int) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_text_prefix(prefix_len)

    def reset_commit_policy(self) -> None:
        self.commit_policy = _MessageCommitPolicy.keep_full_message()

    def has_custom_commit_policy(self) -> bool:
        return self.commit_policy.mode != _MessageCommitMode.KEEP_FULL_MESSAGE


@dataclass
class _LoopState:
    history: list[Message] = field(default_factory=list)
    # Index of the turn-instructions message in `history`, so it can be replaced
    # in place when guidelines are reevaluated between steps. Stable because the
    # instructions sit before the last customer message and all later events are
    # appended after them.
    instructions_index: int | None = None
    in_the_middle_of_running_tools: bool = False

    reasoning_handle: StatusEventHandle | None = None
    reasoning_buffer: StringIO | None = None
    reasoning_chunks: list[str | None] = field(default_factory=list)

    message: _MessageEmissionState = field(default_factory=_MessageEmissionState)
    preamble: _ToolPreamblePolicy = field(default_factory=_ToolPreamblePolicy)

    disable_tools: bool = False
    force_message_note: str = ""

    steps: list[StepResult] = field(default_factory=list)


class _SemanticFailure(Exception):
    pass


class _GiveUp(Exception):
    pass


def _instructions_message(turn_instructions: str, cache_key: str) -> Message:
    return Message(
        role=Role.SYSTEM,
        cache_key=cache_key,
        parts=[
            TextPart(
                text=f"""\
The following is notes and context about the current state of the conversation — the guidelines, glossary, and tools relevant to it. Treat it as background that informs your next reply; it is NOT itself a message addressed to you, so never respond to it, acknowledge it, or refer to it.:
{turn_instructions}"""
            )
        ],
    )


class _InteractionHistoryBuilder:
    def __init__(self, logger: Logger, react: Callable[[], ReactGenerator]) -> None:
        self._logger = logger
        self._react = react

    async def build(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        cache_key = job.context.session.id

        system_message = Message(
            role=Role.SYSTEM,
            cache_key=cache_key,
            parts=[TextPart(text=job.system_instructions)],
        )

        history = [system_message]

        if job.context.state.session_summary:
            history.append(
                Message(
                    role=Role.SYSTEM,
                    cache_key=cache_key,
                    parts=[
                        TextPart(
                            text=f"""\
The earlier part of this session was compacted into the following summary.
Treat it as factual background context for the current interaction. It is not a new
message from the user and should not be acknowledged directly:

### Summary

{job.context.state.session_summary.strip()}
"""
                        )
                    ],
                )
            )

        for event in job.context.interaction.events:
            if event.kind == EventKind.MESSAGE and event.source == EventSource.CUSTOMER:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.CUSTOMER_UI:
                history.append(
                    Message(
                        role=Role.USER,
                        cache_key=cache_key,
                        parts=[TextPart(text=f"[Customer UI Event]: {event.data}")],
                    )
                )
            elif event.kind == EventKind.MESSAGE and event.source in (
                EventSource.AI_AGENT,
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
            ):
                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[TextPart(text=cast(MessageEventData, event.data)["message"])],
                    )
                )
            elif event.source == EventSource.HUMAN_AGENT:
                message_data = cast(MessageEventData, event.data)

                history.append(
                    Message(
                        role=Role.ASSISTANT,
                        cache_key=cache_key,
                        parts=[
                            TextPart(
                                text=f"[Intervention by human agent. Name: {message_data['participant']['display_name']}]: {message_data['message']}"
                            )
                        ],
                    )
                )
            elif event.kind == EventKind.TOOL:
                # Reconstruct every tool event regardless of source (matching the
                # alpha engine). Persisted tool events can carry source=AI_AGENT,
                # so gating on SYSTEM silently dropped prior-turn tool results.
                history.extend(
                    self.build_tool_event_messages(
                        cast(ToolEventData, event.data), event.metadata, cache_key
                    )
                )

        # Providers (e.g. Gemini, Anthropic) require at least one non-system
        # turn. When the agent speaks first — a greeting before any customer
        # message — the interaction is empty and the history holds only the
        # system message; give the model a user turn to respond to.
        if len(history) == 1:
            history.append(
                Message(
                    role=Role.USER,
                    cache_key=cache_key,
                    parts=[TextPart(text="[The conversation has not started yet.]")],
                )
            )

        # Tool events staged for this turn (e.g. retriever results emitted during
        # the on_generating_messages hook) aren't in the interaction history yet,
        # so fold them in here so the model sees the retrieved context.
        for tool_event in job.context.state.tool_events:
            history.extend(
                self.build_tool_event_messages(
                    cast(ToolEventData, tool_event.data), tool_event.metadata, cache_key
                )
            )

        instructions_index: int | None = None

        if include_turn_instructions and job.step_instructions:
            turn_instructions = await job.step_instructions(job.context)

            # Place the instructions immediately BEFORE the last customer message
            # rather than at the very end. Ending the prompt on an imperative note
            # makes the model treat it as the turn to answer — it paraphrases or
            # echoes the instructions back instead of replying. Keeping the
            # customer's message last keeps the model answering the customer.
            instructions_index = self._last_user_message_index(history)
            history.insert(instructions_index, _instructions_message(turn_instructions, cache_key))

        return history, instructions_index

    def build_tool_event_messages(
        self,
        data: ToolEventData,
        metadata: Optional[Mapping[str, JSONSerializable]],
        cache_key: str,
    ) -> list[Message]:
        provider_data = (metadata or {}).get(_PROVIDER_DATA_KEY)
        if isinstance(provider_data, Mapping) and provider_data:
            # A model-issued tool call carrying its provider's replay blob: let the
            # originating provider rebuild a faithful, native tool_use/tool_result
            # pair (consistent ids, signatures, …).
            messages = self._react().deserialize_tool_messages(
                SessionToolMessageDeserializer(data, provider_data)
            )
            if messages is None:
                # The current generator can't natively replay this blob (e.g. a
                # Gemini tool call whose stored thought_signature is unusable).
                # Degrade — don't drop: render the result so the model still sees
                # the tool's data across turns, rather than forgetting it entirely.
                self._logger.debug(
                    "Can't natively replay a tool event while building history "
                    f"({provider_data.get('provider')}/{provider_data.get('model')}); "
                    "falling back to a result-only rendering."
                )
                return self.build_result_only_tool_event_messages(data, cache_key)
            for message in messages:
                message.cache_key = cache_key
            return list(messages)

        # No provider blob (e.g. retriever-staged results, or events from before
        # this was introduced): fall back to the prior tool-result-only rendering.
        return self.build_result_only_tool_event_messages(data, cache_key)

    def build_result_only_tool_event_messages(
        self, data: ToolEventData, cache_key: str
    ) -> list[Message]:
        def build_content_with_args(call: ToolCall) -> str:
            return f"{call['tool_id']}({', '.join(f'{k}={v}' for k, v in call['arguments'].items())}) returned: {call.get('result', {}).get('data', {})}"

        messages: list[Message] = []

        call_id = 0
        for call in data["tool_calls"]:
            call_id += 1
            is_error = "error_details" in call.get("result", {}).get("metadata", {})

            messages.append(
                Message(
                    role=Role.TOOL,
                    cache_key=cache_key,
                    parts=[
                        ToolResultPart(
                            call_id=str(call_id),
                            name=call["tool_id"],
                            content=build_content_with_args(call),
                            is_error=is_error,
                        )
                    ],
                )
            )

        return messages

    def _last_user_message_index(self, history: Sequence[Message]) -> int:
        return next(
            (i for i in range(len(history) - 1, -1, -1) if history[i].role == Role.USER),
            len(history),
        )


class _TurnInstructionBuilder:
    def __init__(self, logger: Logger, preamble_interval_seconds: float) -> None:
        self._logger = logger
        self._preamble_interval_seconds = preamble_interval_seconds

    async def refresh(
        self,
        job: LoopJob,
        state: _LoopState,
        preamble_decision: _ToolPreambleDecision,
        loop_name: str,
    ) -> None:
        if job.step_instructions is not None:
            instructions = await job.step_instructions(job.context)
        else:
            instructions = ""

        reviewer_notes = self._build_reviewer_notes(job, state, preamble_decision)

        if reviewer_notes:
            instructions += (
                "\n\n### IMPORTANT: Please mind the following notes for your next step:\n\n"
                + "\n\n".join(reviewer_notes)
            )

        refreshed_instructions = _instructions_message(
            instructions,
            job.context.session.id,
        )

        if state.instructions_index is not None:
            if state.history[state.instructions_index].text != refreshed_instructions.text:
                self._logger.debug(
                    f"{loop_name} updated turn instructions:\n{refreshed_instructions.text}"
                )

                state.history[state.instructions_index] = refreshed_instructions
        elif instructions:
            # No instructions message exists (e.g. a config with no per-turn
            # step_instructions, like low-effort agents), but there's content to inject
            # — notably the reviewer's adjusted reasoning / TODO on a retry. Insert one,
            # positioned before the last customer message like history building, so it
            # actually reaches the model. Without this the review loop would re-stream
            # the same output and never converge.
            index = next(
                (
                    i
                    for i in range(len(state.history) - 1, -1, -1)
                    if state.history[i].role == Role.USER
                ),
                len(state.history),
            )
            state.history.insert(index, refreshed_instructions)
            state.instructions_index = index

            self._logger.debug(
                f"{loop_name} inserted turn instructions:\n{refreshed_instructions.text}"
            )

    def _build_reviewer_notes(
        self,
        job: LoopJob,
        state: _LoopState,
        preamble_decision: _ToolPreambleDecision,
    ) -> list[str]:
        reviewer_notes: list[str] = []

        if job.context.state.todo:
            reviewer_notes.append(
                "#### TODO LIST: Remaining tasks before responding to the user\n\n"
                + job.context.state.todo
            )

        if job.context.state.step_notes:
            reviewer_notes.append(
                "#### Suggested reasoning for the next step\n\n" + job.context.state.step_notes
            )

        reviewer_notes.append(self._tool_communication_note(preamble_decision))

        if state.force_message_note:
            reviewer_notes.append("#### Required final response\n\n" + state.force_message_note)

        return reviewer_notes

    def _tool_communication_note(self, preamble_decision: _ToolPreambleDecision) -> str:
        match preamble_decision:
            case _ToolPreambleDecision.ALLOW_INITIAL:
                return (
                    "#### Tool communication before tool use\n\n"
                    "If you need to use *a new tool* for this step, you should first send "
                    "exactly one short, natural sentence to the user about what you are "
                    "checking (Checking X; Let me do Y; Just a moment while I Z...). Do not "
                    'add a second "I\'m checking" or "let me check" sentence in the same '
                    "message. Keep it specific to the current action, and avoid repeating "
                    "wording from earlier messages."
                )
            case _ToolPreambleDecision.ALLOW_INTERVAL_UPDATE:
                return (
                    "#### Tool communication before tool use\n\n"
                    f"More than {self._preamble_interval_seconds:g} seconds have passed since your last message to the user. "
                    "If you need to use *a new tool* for this step, please update them on "
                    "what you're currently doing before running the next tool. Send exactly "
                    "one short sentence, and do not add another tool-progress sentence in "
                    "the same message."
                )
            case _ToolPreambleDecision.SUPPRESS:
                return (
                    "#### Tool communication before tool use\n\n"
                    f"Less than {self._preamble_interval_seconds:g} seconds have passed since your last message to the user. "
                    "If you need to use another tool now, do not send another progress "
                    "update or preamble before the tool call. Run the next tool silently."
                )


class _ReasoningEventProcessor:
    async def process(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case ReasoningDelta(text=text):
                if state.reasoning_handle is None:  # First reasoning chunk
                    state.reasoning_buffer = StringIO()
                    state.reasoning_buffer.write(text)
                    state.reasoning_chunks = [text]

                    state.reasoning_handle = await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        ),
                    )
                else:  # Subsequent reasoning chunk
                    assert state.reasoning_buffer is not None

                    state.reasoning_buffer.write(text)
                    state.reasoning_chunks.append(text)

                    state.reasoning_handle = await state.reasoning_handle.update(
                        StatusEventData(
                            status="processing",
                            message=state.reasoning_buffer.getvalue(),
                            chunks=state.reasoning_chunks,
                        )
                    )
            case StepCompleted(result=result):
                if result.message.reasoning:
                    context.state.reasoning_steps.append(result.message.reasoning)

                if state.reasoning_handle is not None:
                    await state.reasoning_handle.update(
                        StatusEventData(
                            status="processing",
                            message=result.message.reasoning,
                            chunks=[*state.reasoning_chunks, None],
                        )
                    )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None
            case _ if (
                state.reasoning_handle is not None
            ):  # In case reasoning is followed by events other than StepCompletion
                assert state.reasoning_buffer is not None

                if state.reasoning_buffer.getvalue().strip():
                    context.state.reasoning_steps.append(state.reasoning_buffer.getvalue())

                await state.reasoning_handle.update(
                    StatusEventData(
                        status="processing",
                        message=state.reasoning_buffer.getvalue(),
                        chunks=[*state.reasoning_chunks, None],
                    )
                )

                state.reasoning_buffer = None
                state.reasoning_chunks = []
                state.reasoning_handle = None


class _ToolStepController:
    def __init__(
        self,
        logger: Logger,
        react: Callable[[], ReactGenerator],
        tool_runner: Callable[[], ToolRunner],
        reviewer: Callable[[], Reviewer],
    ) -> None:
        self._logger = logger
        self._react = react
        self._tool_runner = tool_runner
        self._reviewer = reviewer

    async def process(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
        commit_step: Callable[[_LoopState, StreamEvent], Awaitable[None]],
    ) -> bool:
        match event:
            case ToolCallStarted():
                if not state.in_the_middle_of_running_tools:
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="processing", message="Evaluating tools"),
                    )

                state.in_the_middle_of_running_tools = True
                return False
            case StepCompleted(result=result) if result.needs_tools:
                adjusted_reasoning = await self.review_tool_calls(
                    context,
                    result.message.reasoning,
                    result.tool_calls,
                )

                if adjusted_reasoning:
                    context.state.step_notes = adjusted_reasoning

                    # Was reasoning emitted by the model for this step?
                    if result.message.reasoning:
                        # We need to replace this step's reasoning in the state
                        assert len(context.state.reasoning_steps) > 0
                        context.state.reasoning_steps[-1] = adjusted_reasoning

                    raise _SemanticFailure()  # Restart the step with the adjusted reasoning in place
                else:
                    context.state.step_notes = ""

                # Approved tool calls must be committed before their tool results
                # are appended, preserving the provider-required assistant-tool order.
                await commit_step(state, event)

                await context.session_event_emitter.emit_status_event(
                    trace_id=context.tracer.trace_id,
                    data=StatusEventData(status="processing", message="Running tools"),
                )

                await self.run_tool_calls(context, state, result.tool_calls)

                return True
            case _:
                state.in_the_middle_of_running_tools = False
                return False

    async def review_tool_calls(
        self,
        context: EngineContext,
        reasoning: str,
        tool_calls: Sequence[ToolCallPart],
    ) -> str | None:
        effort = context.state.dynamic_effort_level

        if effort == Effort.MIN:
            # Skip the review for minimal-effort agents
            return None

        if (effort in (Effort.LOW, Effort.MEDIUM)) and (
            not context.state.has_matched_high_criticality_guidelines
        ):
            # For non-high-effort agents, skip the review
            # if no high-criticality guidelines were matched
            return None

        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=StatusEventData(status="processing", message="Reviewing tool use"),
        )

        review_result = await self._reviewer().review_tool_calls(
            context,
            reasoning,
            tool_calls,
        )

        if todo := review_result.todo:
            context.state.todo = todo

        if adjusted_reasoning := review_result.adjusted_reasoning:
            return adjusted_reasoning

        return None

    async def run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        # Run all of the step's tool calls concurrently; results keep call order.
        results: tuple[ToolResult | None] = await safe_gather(
            *(self.run_tool_call(context, tool_call) for tool_call in tool_calls)
        )

        calls_and_results = list(zip(tool_calls, results))

        step_parts: list[ToolResultPart] = []
        transient_call_ids: list[str] = []
        persisted_call_ids: list[str] = []

        for tool_call, result in calls_and_results:
            if result is not None:
                if result.control.get("lifespan", "session") == "session":
                    persisted_call_ids.append(tool_call.id)
                else:
                    transient_call_ids.append(tool_call.id)

                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=result.data,
                        is_error="error_details" in result.metadata,
                    )
                )
            else:
                step_parts.append(
                    ToolResultPart(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
                )

        # Emit transient results into the transient response context
        transient_tool_event = await context.response_event_emitter.emit_tool_event(
            trace_id=context.tracer.trace_id,
            data=ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=tool_call.name,
                        arguments=tool_call.args,
                        result={
                            "data": result.data,
                            "metadata": result.metadata,
                            "control": result.control,
                            "guidelines": result.guidelines,
                            "canned_responses": result.canned_responses,
                            "canned_response_fields": result.canned_response_fields,
                        },
                        rationale=state.reasoning_buffer.getvalue()
                        if state.reasoning_buffer
                        else "Not provided.",
                    )
                    for tool_call, result in calls_and_results
                    if result and tool_call.id in transient_call_ids
                ]
            ),
        )

        # Capture the provider's replay blob for the persisted (session-lifespan)
        # calls, so a later turn can rebuild a faithful native tool turn from the
        # stored event. Index-aligned with the event's tool_calls below (same
        # filter/order). The original ToolCallParts carry the provider artifacts
        # (e.g. Gemini's thought_signature) the serializer needs.
        persisted_calls = [
            tool_call
            for tool_call, result in calls_and_results
            if result and tool_call.id in persisted_call_ids
        ]
        persisted_results = [part for part in step_parts if part.call_id in persisted_call_ids]
        tool_message_serializer = SessionToolMessageSerializer()
        self._react().serialize_tool_messages(
            [
                Message(role=Role.ASSISTANT, parts=persisted_calls),
                Message(role=Role.TOOL, parts=persisted_results),
            ],
            tool_message_serializer,
        )

        # Emit persisted results into the session response context
        persisted_tool_event = await context.session_event_emitter.emit_tool_event(
            trace_id=context.tracer.trace_id,
            data=ToolEventData(
                tool_calls=[
                    ToolCall(
                        tool_id=tool_call.name,
                        arguments=tool_call.args,
                        result={
                            "data": result.data,
                            "metadata": result.metadata,
                            "control": result.control,
                            "guidelines": result.guidelines,
                            "canned_responses": result.canned_responses,
                            "canned_response_fields": result.canned_response_fields,
                        },
                        rationale=state.reasoning_buffer.getvalue()
                        if state.reasoning_buffer
                        else "Not provided.",
                    )
                    for tool_call, result in calls_and_results
                    if result and tool_call.id in persisted_call_ids
                ]
            ),
            metadata={_PROVIDER_DATA_KEY: tool_message_serializer.provider_data},
        )

        context.state.tool_events.append(transient_tool_event)
        context.state.tool_events.append(persisted_tool_event)

        # Finally, append all the react step parts
        state.history.append(
            Message(
                role=Role.TOOL,
                cache_key=context.session.id,
                parts=list(step_parts),
            )
        )

    async def run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        tool_id = context.state.tool_ids_by_name.get(tool_call.name)

        if tool_id is None:
            self._logger.warning(f"Model requested an unknown tool: {tool_call.name}")
            return None

        return await self._tool_runner().run_tool(context, tool_id, tool_call.args)


class BaseLoop(Loop):
    """Shared agentic generation loop. Streaming vs blocking output differs only in
    how the assistant message is surfaced to the session, so that single step —
    :meth:`_surface_message_event` — is left abstract for concrete loops to implement;
    everything else (history building, reasoning/tool handling, the react loop) is
    output-mode-agnostic and lives here."""

    # Retry a step on a transient ReactError, but only before any event has been
    # emitted (a stream can't be replayed mid-flight). Waits between attempts.
    _RETRY_TIMEOUT_PER_ATTTEMPT = (2.0, 8.0, 32.0)
    _TOOL_PREAMBLE_INTERVAL_SECONDS = 15.0
    _SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])(?:\s+|$)")

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        react: ReactGenerator,
        tool_runner: ToolRunner,
        reviewer: Reviewer,
        hooks: EngineHooks,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._optimization_policy = optimization_policy
        self._react = react
        self._tool_runner = tool_runner
        self._reviewer = reviewer
        self._hooks = hooks
        self._history_builder = _InteractionHistoryBuilder(logger, lambda: self._react)
        self._turn_instruction_builder = _TurnInstructionBuilder(
            logger,
            self._TOOL_PREAMBLE_INTERVAL_SECONDS,
        )
        self._reasoning_event_processor = _ReasoningEventProcessor()
        self._tool_step_controller = _ToolStepController(
            logger,
            lambda: self._react,
            lambda: self._tool_runner,
            lambda: self._reviewer,
        )

    async def prefill(self, job: LoopJob) -> Usage:
        self._logger.debug(f"Prefilling job for session {job.context.session.id}")

        # Warm the cache for the stable prefix only — the system instructions and
        # the conversation so far. The per-turn instructions are dynamic and sit
        # past the cache breakpoint, so we leave them out here.
        history, _ = await self._history_builder.build(job, include_turn_instructions=False)

        usage = await self._react.prefill(
            history=history,
            tools=await self._get_tools(job.context),
            tool_choice="auto",
            reasoning=job.reasoning_config,
            hints={"model_size": job.model_size},
        )

        if usage.input_tokens > 0:
            self._logger.debug(f"{self.__class__.__name__} prefill usage:\n {usage}")

        return usage

    async def run(self, job: LoopJob) -> LoopResult:
        # Give hooks (e.g. retrievers) a chance to stage tool events before we
        # build the history — history building folds context.state.tool_events in.
        if not await self._hooks.call_on_generating_messages(job.context):
            # A hook requested that we not proceed with generating a response.
            return LoopResult(job=job, steps=[])

        history, instructions_index = await self._history_builder.build(job)
        state = _LoopState(history=history, instructions_index=instructions_index)

        while not job.context.state.prepared_to_respond:
            try:
                max_semantic_failures = self._max_semantic_failures(job)
                semantic_failure_count = 0

                while semantic_failure_count < max_semantic_failures:
                    try:
                        await self._run_step(job, state)
                    except _SemanticFailure:
                        semantic_failure_count += 1
                        await self._reset_message_after_restart(job.context, state)
                        continue
                    else:
                        break

                if semantic_failure_count == max_semantic_failures:
                    raise _GiveUp(
                        "The agent repeatedly proposed tool calls that were rejected by "
                        "policy review."
                    )

                job.context.state.iterations.append(
                    IterationState(
                        matched_guidelines=[],
                        ruled_out=[],
                        resolved_guidelines=[],
                        tool_insights=ToolInsights(evaluations={}, missing_data={}),
                        executed_tools=[],
                    )
                )

                if len(job.context.state.iterations) >= 30:
                    self._logger.warning(
                        f"Large number of engine iterations on session {job.context.session.id} ({job.context.session.title or 'Untitled'}):\n{state.steps}"
                    )

                if len(job.context.state.iterations) == job.context.agent.max_engine_iterations:
                    raise _GiveUp(
                        "The agent reached the maximum number of engine iterations "
                        "without preparing a response."
                    )
            except _GiveUp as exc:
                await self._give_up(job, state, str(exc))

        await job.context.session_event_emitter.emit_status_event(
            trace_id=job.context.tracer.trace_id,
            data=StatusEventData(status="ready", data={"stage": "completed"}),
        )

        await self._hooks.call_on_messages_emitted(job.context)

        return LoopResult(job=job, steps=state.steps)

    def _max_semantic_failures(self, job: LoopJob) -> int:
        """The number of times a step can be restarted due to a reviewer-provided
        policy-adjusted reasoning before we give up and propagate the failure."""
        match job.context.state.dynamic_effort_level:
            case Effort.MIN:
                return 1
            case Effort.LOW:
                return 2
            case Effort.MEDIUM:
                return 3
            case Effort.HIGH:
                return 5
            case Effort.MAX:
                return 10

    async def _run_step(self, job: LoopJob, state: _LoopState) -> None:
        """Run one react step, processing each event. A transient ReactError is
        retried — but only while NO event has been produced yet: once events have
        been emitted, the stream can't be replayed (it would re-emit chunks and
        re-run side effects), so the error propagates. The transient errors we
        retry are raised when the stream is opened, before any event."""

        await self._update_step_instructions(job, state)

        for attempt in range(len(self._RETRY_TIMEOUT_PER_ATTTEMPT) + 1):
            produced = False
            try:
                async for event in self._react.stream_step(
                    history=state.history,
                    tools=[] if state.disable_tools else await self._get_tools(job.context),
                    tool_choice="none" if state.disable_tools else "auto",
                    reasoning=job.reasoning_config,
                    hints={"model_size": job.model_size, "hedge_timeout": 10.0},
                ):
                    produced = True
                    await self._process_reasoning_event(job.context, state, event)
                    # Surface/close the message BEFORE the tool status: a text->tool
                    # transition finalizes the preamble as its own bubble first, so the
                    # tool status follows the message (no flicker), and post-tool text
                    # starts a fresh message instead of gluing onto the preamble.
                    await self._surface_message_event(job.context, state, event)
                    committed = await self._process_tool_event(job.context, state, event)
                    if not committed:
                        await self._commit_react_event(state, event)

                return
            except ReactError as exc:
                if (
                    not exc.retryable
                    or produced
                    or attempt == len(self._RETRY_TIMEOUT_PER_ATTTEMPT)
                ):
                    raise
                wait = self._RETRY_TIMEOUT_PER_ATTTEMPT[attempt]
                self._logger.warning(
                    f"{self.__class__.__name__} retrying step after a transient error "
                    f"({exc}); retrying in {wait}s (attempt {attempt + 2})."
                )
                await asyncio.sleep(wait)

    async def _give_up(self, job: LoopJob, state: _LoopState, reason: str) -> None:
        self._logger.warning(
            f"{reason} Forcing a final message on session {job.context.session.id} "
            f"({job.context.session.title or 'Untitled'}). Reasoning: \n"
            f"{json.dumps(job.context.state.reasoning_steps, indent=2)}"
        )

        await self._reset_message_after_restart(job.context, state)
        self._drop_trailing_empty_assistant_messages(state)

        state.disable_tools = True
        state.force_message_note = (
            "You've failed to address the current request after repeated attempts. "
            "You must now explain to the user why you were not able to help currently. "
            "Do not claim that the request was completed. Tool use is disabled for this "
            "step, so you must send a concise message to the user now."
        )

        try:
            await self._run_step(job, state)
        except _SemanticFailure:
            self._logger.error(
                f"{self.__class__.__name__} could not force a final message because the "
                "model still attempted tool use with tools disabled."
            )
        finally:
            state.disable_tools = False
            state.force_message_note = ""

        job.context.state.prepared_to_respond = True

    def _drop_trailing_empty_assistant_messages(self, state: _LoopState) -> None:
        while state.history:
            last_message = state.history[-1]
            if (
                last_message.role == Role.ASSISTANT
                and not last_message.text
                and not last_message.tool_calls
            ):
                state.history.pop()
                continue

            break

    async def _get_tools(self, context: EngineContext) -> list[ToolSpec]:
        return [*tool_specs_from_tools(context.state.available_tools)]

    def _tool_preamble_decision(self, state: _LoopState) -> _ToolPreambleDecision:
        return state.preamble.decide(
            time.monotonic(),
            self._TOOL_PREAMBLE_INTERVAL_SECONDS,
        )

    def _tool_preamble_is_allowed(self, state: _LoopState) -> bool:
        return self._tool_preamble_decision(state) in (
            _ToolPreambleDecision.ALLOW_INITIAL,
            _ToolPreambleDecision.ALLOW_INTERVAL_UPDATE,
        )

    def _mark_user_visible_message_emitted(self, state: _LoopState) -> None:
        state.preamble.mark_user_visible_message_emitted(time.monotonic())

    def _tool_preamble_text(self, text: str) -> str:
        match = self._SENTENCE_END_PATTERN.search(text)
        if match is None:
            return text

        return text[: match.end()].strip()

    def _message_with_text_prefix(self, message: Message, prefix_len: int) -> Message:
        remaining = prefix_len
        parts: list[Part] = []

        for part in message.parts:
            if isinstance(part, TextPart):
                if remaining <= 0:
                    continue

                text = part.text[:remaining]
                remaining -= len(text)
                if text:
                    parts.append(TextPart(text=text, provider_data=part.provider_data))
            else:
                parts.append(part)

        return Message(
            role=message.role,
            parts=parts,
            provider_data=message.provider_data,
            cache_key=message.cache_key,
        )

    def _apply_message_commit_policy(
        self,
        message: Message,
        policy: _MessageCommitPolicy,
    ) -> Message:
        match policy.mode:
            case _MessageCommitMode.KEEP_FULL_MESSAGE:
                return message
            case _MessageCommitMode.KEEP_NO_TEXT:
                return self._message_with_text_prefix(message, 0)
            case _MessageCommitMode.KEEP_TEXT_PREFIX:
                return self._message_with_text_prefix(message, policy.prefix_len)

    async def _commit_react_event(self, state: _LoopState, event: StreamEvent) -> None:
        if isinstance(event, StepCompleted):
            message = event.result.message
            if event.result.needs_tools:
                message = self._apply_message_commit_policy(
                    message,
                    state.message.commit_policy,
                )

            state.history.append(message)
            state.steps.append(event.result)
            state.message.reset_commit_policy()

            if event.result.message.reasoning:
                self._logger.trace(
                    f"{self.__class__.__name__} step reasoning:\n {event.result.message.reasoning}"
                )

            self._logger.debug(f"{self.__class__.__name__} step usage:\n {event.result.usage}")

    async def _process_reasoning_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        await self._reasoning_event_processor.process(context, state, event)

    async def _process_tool_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> bool:
        return await self._tool_step_controller.process(
            context,
            state,
            event,
            self._commit_react_event,
        )

    async def _review_tool_calls(
        self,
        context: EngineContext,
        reasoning: str,
        tool_calls: Sequence[ToolCallPart],
    ) -> str | None:
        return await self._tool_step_controller.review_tool_calls(
            context,
            reasoning,
            tool_calls,
        )

    async def _run_tool_calls(
        self,
        context: EngineContext,
        state: _LoopState,
        tool_calls: Sequence[ToolCallPart],
    ) -> None:
        await self._tool_step_controller.run_tool_calls(context, state, tool_calls)

    async def _run_tool_call(
        self, context: EngineContext, tool_call: ToolCallPart
    ) -> ToolResult | None:
        return await self._tool_step_controller.run_tool_call(context, tool_call)

    @abstractmethod
    async def _surface_message_event(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        """Surface the assistant's message for the current stream event.

        This is the sole output-mode-specific step: a streaming loop emits the
        message incrementally (chunked) as deltas arrive, while a blocking loop
        emits it once, complete, on step completion. Concrete loops should call
        :meth:`_complete_message_step` when a step that produced a message
        completes, to share the loop's termination + hook logic.
        """
        ...

    async def _complete_message_step(self, context: EngineContext, result: StepResult) -> None:
        """Shared step-completion tail for a step that produced a message: a message
        with no pending tool calls ends the loop, and the message-generated hook
        fires regardless of output mode."""
        if not result.needs_tools:
            context.state.prepared_to_respond = True

        await self._hooks.call_on_message_generated(context, result.message.text)

    async def _reset_message_after_restart(self, context: EngineContext, state: _LoopState) -> None:
        """A reviewer-rejected step is being restarted. Finalize the message already
        streamed during the rejected attempt as its own (complete) bubble, then reset
        the streaming state so the retry begins a fresh message instead of appending to
        it. A blocking loop doesn't stream a partial message (its handle is None here),
        so this only clears the per-attempt flags there."""
        if state.message.handle is not None:
            await state.message.handle.update(
                MessageEventData(
                    message=state.message.buffer.getvalue() if state.message.buffer else "",
                    participant=Participant(id=context.agent.id, display_name=context.agent.name),
                    chunks=[*state.message.chunks, None],
                )
            )

        state.message.reset_step_output()
        state.in_the_middle_of_running_tools = False

    def _get_model_size(self, context: EngineContext, state: _LoopState) -> ModelSize:
        return ModelSize.MEDIUM

    async def _build_history(
        self,
        job: LoopJob,
        *,
        include_turn_instructions: bool = True,
    ) -> tuple[list[Message], int | None]:
        return await self._history_builder.build(
            job,
            include_turn_instructions=include_turn_instructions,
        )

    def _instructions_message(self, turn_instructions: str, cache_key: str) -> Message:
        return _instructions_message(turn_instructions, cache_key)

    async def _update_step_instructions(
        self,
        job: LoopJob,
        state: _LoopState,
    ) -> None:
        await self._turn_instruction_builder.refresh(
            job,
            state,
            self._tool_preamble_decision(state),
            self.__class__.__name__,
        )

    def _build_tool_event_messages(
        self,
        data: ToolEventData,
        metadata: Optional[Mapping[str, JSONSerializable]],
        cache_key: str,
    ) -> list[Message]:
        return self._history_builder.build_tool_event_messages(data, metadata, cache_key)

    def _build_result_only_tool_event_messages(
        self, data: ToolEventData, cache_key: str
    ) -> list[Message]:
        return self._history_builder.build_result_only_tool_event_messages(data, cache_key)
