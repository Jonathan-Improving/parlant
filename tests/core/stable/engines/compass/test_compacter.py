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

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from parlant.core.agents import Effort
from parlant.core.common import JSONSerializable, generate_id
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.compass.compacter import (
    DEFAULT_COMPACTION_POLICIES,
    Compacter,
    CompactionDetail,
    CompactionPolicy,
    CompactionSchema,
)
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.nlp.generation import SchematicGenerationResult, SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    SessionId,
    ToolEventData,
)
from parlant.core.tracer import LocalTracer
from parlant.core.loggers import StdoutLogger

from tests.core.common.utils import create_event_message
from tests.core.stable.engines.compass.matching.utils import (
    create_agent,
    create_engine_context,
    create_guideline,
)


class _FakeTokenizer(EstimatingTokenizer):
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.prompts: list[str] = []

    async def estimate_token_count(self, prompt: str) -> int:
        self.prompts.append(prompt)
        return self.token_count


class _FakeCompactionGenerator(SchematicGenerator[CompactionSchema]):
    def __init__(self, token_count: int = 0, summary: str = "Compacted summary") -> None:
        self._tokenizer = _FakeTokenizer(token_count)
        self.summary = summary
        self.prompts: list[str] = []
        self.hints: list[Mapping[str, Any]] = []

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[CompactionSchema]:
        prompt_text = prompt.build() if isinstance(prompt, PromptBuilder) else prompt
        self.prompts.append(prompt_text)
        self.hints.append(hints)

        return SchematicGenerationResult(
            content=CompactionSchema(summary=self.summary),
            info=GenerationInfo(
                schema_name="CompactionSchema",
                model="fake",
                duration=0.0,
                usage=UsageInfo(input_tokens=1, output_tokens=1),
            ),
        )

    @property
    def id(self) -> str:
        return "fake-compaction-generator"

    @property
    def max_tokens(self) -> int:
        return 100_000

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer


class _FakeEntityQueries:
    def __init__(self, events: list[Event]) -> None:
        self.events = events

    async def find_events(self, session_id: SessionId) -> list[Event]:
        return self.events


class _RecordingLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.error_messages: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)


def _compacter(generator: _FakeCompactionGenerator) -> Compacter:
    tracer = LocalTracer()
    return Compacter(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        schematic_generator=generator,
    )


def _policy(
    threshold: int,
    detail: CompactionDetail = CompactionDetail.LOW,
) -> dict[Effort, CompactionPolicy]:
    return {
        effort: CompactionPolicy(token_threshold=threshold, detail_level=detail)
        for effort in DEFAULT_COMPACTION_POLICIES
    }


def _tool_event(offset: int) -> Event:
    creation_utc = datetime.now(timezone.utc)

    return Event(
        id=EventId(generate_id()),
        source=EventSource.SYSTEM,
        kind=EventKind.TOOL,
        creation_utc=creation_utc,
        modified_utc=creation_utc,
        offset=offset,
        trace_id="<main>",
        data=cast(
            JSONSerializable,
            ToolEventData(
                tool_calls=[
                    {
                        "tool_id": "lookup_account",
                        "arguments": {"account_id": "acct-123"},
                        "rationale": "Need account state",
                        "result": {
                            "data": {"balance": 1200},
                            "metadata": {},
                            "control": {},
                            "canned_responses": [],
                            "canned_response_fields": {},
                        },
                    }
                ]
            ),
        ),
        metadata={},
        deleted=False,
    )


@pytest.mark.asyncio
async def test_needs_compaction_returns_false_below_threshold() -> None:
    generator = _FakeCompactionGenerator(token_count=9)
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=10))

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    assert not await compacter.needs_compaction(context)


@pytest.mark.asyncio
async def test_needs_compaction_returns_true_at_threshold() -> None:
    generator = _FakeCompactionGenerator(token_count=10)
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=10))

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    assert await compacter.needs_compaction(context)


@pytest.mark.asyncio
async def test_token_counting_input_includes_tool_events() -> None:
    generator = _FakeCompactionGenerator(token_count=10)
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=10))

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "what is my balance?")])
    context.interaction = replace(
        context.interaction,
        events=[*context.interaction.events, _tool_event(offset=1)],
    )
    context.state = ResponseState(agent_effort=context.agent.effort)

    assert await compacter.needs_compaction(context)
    assert "lookup_account" in generator._tokenizer.prompts[-1]
    assert "balance" in generator._tokenizer.prompts[-1]


@pytest.mark.asyncio
async def test_policy_lookup_uses_dynamic_effort_level() -> None:
    generator = _FakeCompactionGenerator(token_count=7)
    compacter = _compacter(generator)
    policies = _policy(threshold=10)
    policies[Effort.HIGH] = CompactionPolicy(
        token_threshold=5,
        detail_level=CompactionDetail.HIGH,
    )
    compacter.set_policy(policies)

    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "I need help with a regulated action")],
        agent=replace(create_agent(), effort=Effort.LOW),
    )
    high_effort_guideline = replace(
        create_guideline("the user requests a regulated action"),
        effort=Effort.HIGH,
    )
    context.state = ResponseState(
        agent_effort=context.agent.effort,
        ordinary_guideline_matches=[
            GuidelineMatch(guideline=high_effort_guideline, rationale="relevant"),
        ],
    )

    assert await compacter.needs_compaction(context)


@pytest.mark.asyncio
async def test_compact_returns_summary_and_includes_prompt_context() -> None:
    generator = _FakeCompactionGenerator(summary="Remember that the user needs a refund.")
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=1, detail=CompactionDetail.HIGH))

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "I need a refund")])
    context.state = ResponseState(
        agent_effort=context.agent.effort,
        session_summary="Earlier: the user authenticated successfully.",
    )

    result = await compacter.compact(context)

    assert result.summary == "Remember that the user needs a refund."
    prompt = generator.prompts[-1]
    assert "# Session Summary" in prompt
    assert "Earlier: the user authenticated successfully." in prompt
    assert "# SYSTEM INSTRUCTIONS AND POLICIES" in prompt
    assert "# Interaction History" in prompt
    assert prompt.index("# SYSTEM INSTRUCTIONS AND POLICIES") < prompt.index("# Session Summary")
    assert prompt.index("# Session Summary") < prompt.index("# Interaction History")
    assert "I need a refund" in prompt
    assert "Detail level: high" in prompt
    assert generator.hints[-1]["reasoning_effort"] == "medium"
    assert generator.hints[-1]["cache"] == {
        "key": f"{context.session.id}.compacter",
        "breakpoint": PromptBuilder.INTERACTION_HISTORY_HEADER,
    }


def test_prompt_builder_add_session_summary_ignores_empty_summary() -> None:
    prompt = PromptBuilder().add_session_summary("").build()

    assert "Session Summary" not in prompt


def test_prompt_builder_add_session_summary_adds_background_section() -> None:
    prompt = PromptBuilder().add_session_summary("The user already confirmed identity.").build()

    assert "# Session Summary" in prompt
    assert "The user already confirmed identity." in prompt
    assert "not a new" in prompt


@pytest.mark.asyncio
async def test_engine_loads_only_events_after_last_compaction_marker() -> None:
    events = [
        create_event_message(0, EventSource.CUSTOMER, "old user message"),
        create_event_message(
            1,
            EventSource.SYSTEM,
            "first summary",
            metadata={"source": "compacter"},
        ),
        create_event_message(2, EventSource.CUSTOMER, "middle user message"),
        create_event_message(
            3,
            EventSource.SYSTEM,
            "latest summary",
            metadata={"source": "compacter"},
        ),
        create_event_message(4, EventSource.CUSTOMER, "new user message"),
    ]

    tracer = LocalTracer()
    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Any, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, _FakeEntityQueries(events)),
        hooks=cast(Any, object()),
    )
    state = ResponseState()

    interaction = await engine._load_interaction_state(  # pyright: ignore[reportPrivateUsage]
        create_engine_context(conversation=[]).info,
        state,
    )

    assert state.session_summary == "latest summary"
    assert [cast(dict[str, Any], event.data)["message"] for event in interaction.events] == [
        "new user message"
    ]


@pytest.mark.asyncio
async def test_engine_does_not_treat_plain_system_message_as_compaction_marker() -> None:
    events = [
        create_event_message(0, EventSource.CUSTOMER, "old user message"),
        create_event_message(1, EventSource.SYSTEM, "plain system message"),
        create_event_message(2, EventSource.CUSTOMER, "new user message"),
    ]

    tracer = LocalTracer()
    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Any, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, _FakeEntityQueries(events)),
        hooks=cast(Any, object()),
    )
    state = ResponseState(session_summary="stale summary")

    interaction = await engine._load_interaction_state(  # pyright: ignore[reportPrivateUsage]
        create_engine_context(conversation=[]).info,
        state,
    )

    assert state.session_summary == ""
    assert len(interaction.events) == 3


@pytest.mark.asyncio
async def test_compact_if_needed_reloads_history_before_generating_summary() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "old message")])
    context.state = ResponseState(agent_effort=context.agent.effort)

    persisted_events = [
        *context.interaction.events,
        create_event_message(1, EventSource.AI_AGENT, "current response"),
    ]
    generator = _FakeCompactionGenerator(token_count=10, summary="fresh compacted summary")
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=1))

    tracer = LocalTracer()
    logger = _RecordingLogger()
    engine = CompassEngine(
        logger=cast(Any, logger),
        tracer=tracer,
        meter=cast(Any, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=compacter,
        entity_queries=cast(Any, _FakeEntityQueries(persisted_events)),
        hooks=cast(Any, object()),
    )

    await engine._compact_if_needed(context)  # pyright: ignore[reportPrivateUsage]

    assert generator.prompts
    assert "old message" in generator.prompts[-1]
    assert "current response" in generator.prompts[-1]
    assert context.state.session_summary == "fresh compacted summary"

    emitted_events = context.session_event_emitter.events  # type: ignore[attr-defined]
    assert emitted_events[-2].kind == EventKind.STATUS
    assert cast(dict[str, Any], emitted_events[-2].data)["message"] == "Compacting session"
    assert emitted_events[-1].source == EventSource.SYSTEM
    assert emitted_events[-1].kind == EventKind.MESSAGE
    assert emitted_events[-1].metadata == {"source": "compacter"}
    assert cast(dict[str, Any], emitted_events[-1].data)["message"] == "fresh compacted summary"
    assert any("Summary:\nfresh compacted summary" in message for message in logger.debug_messages)


@pytest.mark.asyncio
async def test_compact_if_needed_does_not_emit_when_below_threshold() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "old message")])
    context.state = ResponseState(agent_effort=context.agent.effort)
    generator = _FakeCompactionGenerator(token_count=0)
    compacter = _compacter(generator)
    compacter.set_policy(_policy(threshold=1))

    tracer = LocalTracer()
    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Any, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=compacter,
        entity_queries=cast(Any, _FakeEntityQueries([])),
        hooks=cast(Any, object()),
    )

    await engine._compact_if_needed(context)  # pyright: ignore[reportPrivateUsage]

    assert generator.prompts == []
    assert context.session_event_emitter.events == []  # type: ignore[attr-defined]
