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
from parlant.core.engines.compass.matcher import Matcher
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
