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
from unittest.mock import AsyncMock
import pytest

from parlant.core.agents import Effort
from parlant.core.common import Criticality
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.compass.matcher import (
    Matcher,
    _ContextUsage,
    _SESSION_GUIDELINE_IDS_METADATA_KEY,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
    create_term,
)


class _FakeEntityQueries:
    async def find_guideline_tool_associations(self):
        return []


class _FakeEntityCommands:
    def __init__(self) -> None:
        self.update_session = AsyncMock()


class _FakeRelationshipStore:
    async def list_relationships(self, *args, **kwargs):
        return []


class _FakeMatcherRegistry:
    def get(self, guideline_id):
        return None


def _make_warm_up_matcher() -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._guideline_ranker = AsyncMock()
    matcher._guideline_distiller = AsyncMock()
    matcher._matcher_registry = _FakeMatcherRegistry()
    matcher._relationship_store = _FakeRelationshipStore()
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = _FakeEntityCommands()
    return matcher


def _make_session_guidelines_matcher(
    entity_commands: _FakeEntityCommands | None = None,
) -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = entity_commands or _FakeEntityCommands()
    return matcher


def _context_with_guidelines(*guidelines, effort: Effort) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(
        agent_effort=effort,
        usable_guidelines=list(guidelines),
        glossary_terms={create_term("known term", "already loaded")},
    )
    return context


def test_that_distilled_actions_are_wrapped_as_policy_notes() -> None:
    matcher = object.__new__(Matcher)
    guideline = replace(
        create_guideline(condition="customer wants a refund", action="explain refund rules"),
        title="Refund eligibility",
    )

    note = matcher._format_distilled_policy_note(guideline, "ask for the order ID")

    assert note == (
        'According to policy "Refund eligibility", ask for the order ID\n'
        "Apply this only insofar as it remains compatible with the other active policies "
        "and system instructions."
    )


def test_that_distilled_policy_notes_have_a_fallback_title() -> None:
    matcher = object.__new__(Matcher)
    guideline = create_guideline(condition="customer wants a refund", action="explain refund rules")

    note = matcher._format_distilled_policy_note(guideline, "ask for the order ID")

    assert note.startswith('According to policy "Untitled policy", ask for the order ID')


def test_that_description_only_distilled_matches_are_rendered_as_instruction_reminders() -> None:
    guideline = replace(
        create_guideline(
            condition="booking a flight",
            action=None,
            description="Collect booking details in order.",
        ),
        criticality=Criticality.HIGH,
        title="Book flight",
    )
    match = GuidelineMatch(
        guideline=guideline,
        rationale="Relevant.",
        metadata={"distilled_action": "Ask the user for the trip type."},
    )

    prompt = PromptBuilder().add_matched_guidelines([match], {}, {guideline.id: guideline}).build()

    assert '### Review the instructions under "Book flight"' in prompt
    assert "Ask the user for the trip type." in prompt
    assert "IMPORTANT: Please go back and reason" in prompt


def test_that_matcher_queries_start_with_session_summary_when_available() -> None:
    matcher = _make_session_guidelines_matcher()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "current request")])
    context.state = ResponseState(session_summary="Earlier booking details were collected.")

    lines = matcher._build_interaction_query_lines(context)

    assert lines == [
        "Session summary: Earlier booking details were collected.",
        "EventSource.CUSTOMER: current request",
    ]
    assert matcher._build_tool_query(context).endswith(str(lines))


@pytest.mark.asyncio
async def test_that_session_guidelines_are_loaded_from_session_metadata() -> None:
    guideline_1 = create_guideline(condition="customer asks for help", action="ask what they need")
    guideline_2 = create_guideline(condition="customer asks for refund", action="explain refunds")
    context = _context_with_guidelines(guideline_1, guideline_2, effort=Effort.MEDIUM)
    context.session = replace(
        context.session,
        metadata={
            _SESSION_GUIDELINE_IDS_METADATA_KEY: [
                str(guideline_2.id),
                "missing-guideline",
                str(guideline_2.id),
                str(guideline_1.id),
                17,
            ]
        },
    )
    matcher = _make_session_guidelines_matcher()

    await matcher._load_session_guidelines(context)

    assert context.state.session_guidelines == {guideline_2, guideline_1}


@pytest.mark.asyncio
async def test_that_matched_guidelines_are_stored_in_session_metadata() -> None:
    guideline_1 = create_guideline(condition="customer asks for help", action="ask what they need")
    guideline_2 = create_guideline(condition="customer asks for refund", action="explain refunds")
    guideline_3 = create_guideline(condition="customer asks for billing", action="explain billing")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_guidelines_matcher(entity_commands)
    context = _context_with_guidelines(guideline_1, guideline_2, guideline_3, effort=Effort.MEDIUM)
    context.state.session_guidelines = [guideline_1]
    context.state.ordinary_guideline_matches = [
        GuidelineMatch(guideline=guideline_2, rationale="relevant"),
        GuidelineMatch(guideline=guideline_1, rationale="still relevant"),
    ]
    matches = [
        (GuidelineMatch(guideline=guideline_2, rationale="relevant"), _ContextUsage.MATCH_CURRENT_TURN),
        (
            GuidelineMatch(guideline=guideline_1, rationale="still relevant"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
    ]

    await matcher._store_session_guidelines(context, matches)

    assert context.state.session_guidelines == {guideline_1, guideline_2}
    entity_commands.update_session.assert_awaited_once()
    updated_session_id, params = entity_commands.update_session.await_args.args
    assert updated_session_id == context.session.id
    assert set(params["metadata"][_SESSION_GUIDELINE_IDS_METADATA_KEY]) == {
        str(guideline_1.id),
        str(guideline_2.id),
    }
    assert set(context.session.metadata[_SESSION_GUIDELINE_IDS_METADATA_KEY]) == {
        str(guideline_1.id),
        str(guideline_2.id),
    }


@pytest.mark.asyncio
async def test_that_session_only_matches_do_not_enter_turn_matched_guidelines() -> None:
    recalled_guideline = create_guideline(
        condition="customer asks for help",
        action="ask what they need",
    )
    turn_guideline = create_guideline(
        condition="customer asks for refund",
        action="explain refunds",
    )
    matcher = _make_session_guidelines_matcher()
    context = _context_with_guidelines(recalled_guideline, turn_guideline, effort=Effort.MEDIUM)
    matches = [
        (
            GuidelineMatch(guideline=recalled_guideline, rationale="recalled"),
            _ContextUsage.INCLUDE_IN_SESSION,
        ),
        (
            GuidelineMatch(guideline=turn_guideline, rationale="matched"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
    ]

    await matcher._record(context, matches, append=False)

    assert context.state.ordinary_guideline_matches == [
        GuidelineMatch(guideline=turn_guideline, rationale="matched")
    ]
    assert context.state.tool_enabled_guideline_matches == {}
    assert context.state.session_guidelines == {recalled_guideline, turn_guideline}


@pytest.mark.asyncio
async def test_that_session_guidelines_are_not_stored_when_metadata_is_unchanged() -> None:
    guideline = create_guideline(condition="customer asks for help", action="ask what they need")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_guidelines_matcher(entity_commands)
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)
    context.state.session_guidelines = [guideline]
    context.state.ordinary_guideline_matches = [GuidelineMatch(guideline=guideline, rationale="")]
    context.session = replace(
        context.session,
        metadata={_SESSION_GUIDELINE_IDS_METADATA_KEY: [str(guideline.id)]},
    )
    matches = [(GuidelineMatch(guideline=guideline, rationale=""), _ContextUsage.MATCH_CURRENT_TURN)]

    await matcher._store_session_guidelines(context, matches)

    entity_commands.update_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_warm_up_skips_distiller_when_no_guidelines_need_distillation() -> None:
    matcher = _make_warm_up_matcher()
    guideline = create_guideline(condition="customer asks for help", action="ask what they need")
    context = _context_with_guidelines(guideline, effort=Effort.HIGH)

    await matcher.warm_up(context)

    matcher._guideline_ranker.warm_up.assert_awaited_once_with(context)
    matcher._guideline_distiller.warm_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_warm_up_skips_ranker_when_only_distiller_is_needed() -> None:
    matcher = _make_warm_up_matcher()
    guideline = replace(
        create_guideline(
            condition="customer asks for help",
            action="ask what they need",
            description="Follow this detailed process. " * 20,
        ),
        criticality=Criticality.MEDIUM,
    )
    context = _context_with_guidelines(guideline, effort=Effort.HIGH)

    await matcher.warm_up(context)

    matcher._guideline_ranker.warm_up.assert_not_awaited()
    matcher._guideline_distiller.warm_up.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_warm_up_skips_both_components_when_strategy_needs_neither() -> None:
    matcher = _make_warm_up_matcher()
    guideline = create_guideline(condition="customer asks for help", action="ask what they need")
    context = _context_with_guidelines(guideline, effort=Effort.LOW)

    await matcher.warm_up(context)

    matcher._guideline_ranker.warm_up.assert_not_awaited()
    matcher._guideline_distiller.warm_up.assert_not_awaited()
