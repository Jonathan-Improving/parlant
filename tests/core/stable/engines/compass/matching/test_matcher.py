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
from parlant.core.engines.compass.matching.guideline_distiller import (
    DistilledGuideline,
    GuidelineDistillationResult,
)
from parlant.core.engines.compass.matching.guideline_ranker import (
    GuidelineRankingResult,
    RankedGuideline,
)
from parlant.core.engines.compass.matching.guideline_recaller import (
    GuidelineRecallResult,
    RecalledGuideline,
)
from parlant.core.engines.compass.matcher import (
    Matcher,
    _ContextUsage,
    _SESSION_GUIDELINE_IDS_METADATA_KEY,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.matching.utils import (
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


class _FakeLogger:
    def debug(self, *args, **kwargs):
        pass


def _make_warm_up_matcher() -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._guideline_ranker = AsyncMock()
    matcher._guideline_distiller = AsyncMock()
    matcher._matcher_registry = _FakeMatcherRegistry()
    matcher._relationship_store = _FakeRelationshipStore()
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = _FakeEntityCommands()
    return matcher


def _make_batch_matcher() -> Matcher:
    matcher = _make_warm_up_matcher()
    matcher._logger = _FakeLogger()
    matcher._guideline_function_matcher = AsyncMock()
    matcher._guideline_function_matcher.match = AsyncMock(return_value=[])
    matcher._guideline_recaller = AsyncMock()
    matcher._guideline_recaller.recall = AsyncMock(return_value=GuidelineRecallResult([], 0.0))
    matcher._guideline_distiller = AsyncMock()
    matcher._guideline_distiller.distill = AsyncMock(
        return_value=GuidelineDistillationResult([], None)
    )
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
async def test_that_storing_session_guidelines_preserves_existing_guidelines() -> None:
    guideline_1 = create_guideline(condition="customer asks for help", action="ask what they need")
    guideline_2 = create_guideline(condition="customer asks for refund", action="explain refunds")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_guidelines_matcher(entity_commands)
    context = _context_with_guidelines(guideline_1, guideline_2, effort=Effort.MEDIUM)
    context.state.session_guidelines = {guideline_1}
    context.session = replace(
        context.session,
        metadata={_SESSION_GUIDELINE_IDS_METADATA_KEY: [str(guideline_1.id)]},
    )
    matches = [
        (
            GuidelineMatch(guideline=guideline_2, rationale="newly relevant"),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]

    await matcher._store_session_guidelines(context, matches)

    assert context.state.session_guidelines == {guideline_1, guideline_2}
    entity_commands.update_session.assert_awaited_once()
    _, params = entity_commands.update_session.await_args.args
    assert set(params["metadata"][_SESSION_GUIDELINE_IDS_METADATA_KEY]) == {
        str(guideline_1.id),
        str(guideline_2.id),
    }


@pytest.mark.asyncio
async def test_that_storing_no_new_session_guidelines_does_not_clear_existing_guidelines() -> None:
    guideline = create_guideline(condition="customer asks for help", action="ask what they need")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_guidelines_matcher(entity_commands)
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)
    context.state.session_guidelines = {guideline}
    context.session = replace(
        context.session,
        metadata={_SESSION_GUIDELINE_IDS_METADATA_KEY: [str(guideline.id)]},
    )

    await matcher._store_session_guidelines(context, [])

    assert context.state.session_guidelines == {guideline}
    entity_commands.update_session.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_that_warm_up_ranks_guideline_that_can_raise_effort() -> None:
    matcher = _make_warm_up_matcher()
    guideline = replace(
        create_guideline(condition="customer asks for regulated help", action="follow the policy"),
        effort=Effort.HIGH,
    )
    context = _context_with_guidelines(guideline, effort=Effort.LOW)

    await matcher.warm_up(context)

    matcher._guideline_ranker.warm_up.assert_awaited_once_with(context)
    matcher._guideline_distiller.warm_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_low_criticality_guideline_that_can_raise_effort_is_matched_this_turn() -> None:
    matcher = _make_batch_matcher()
    guideline = replace(
        create_guideline(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Criticality.LOW,
        effort=Effort.HIGH,
    )
    matcher._guideline_ranker.rank = AsyncMock(
        return_value=GuidelineRankingResult(
            [
                RankedGuideline(
                    guideline=guideline,
                    reasoning="Relevant.",
                    is_relevant=True,
                    score=1.0,
                )
            ],
            None,
        )
    )
    context = _context_with_guidelines(guideline, effort=Effort.LOW)

    matches = await matcher._run_batches(context, [guideline])

    matcher._guideline_recaller.recall.assert_awaited_once_with(context, [guideline])
    matcher._guideline_ranker.rank.assert_awaited_once_with(context, [guideline])
    assert matches == [
        (
            GuidelineMatch(guideline=guideline, rationale="Relevant."),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]


@pytest.mark.asyncio
async def test_that_ranked_guideline_can_still_be_discovered_into_session_by_recall() -> None:
    matcher = _make_batch_matcher()
    guideline = replace(
        create_guideline(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Criticality.HIGH,
    )
    matcher._guideline_recaller.recall = AsyncMock(
        return_value=GuidelineRecallResult(
            [RecalledGuideline(guideline=guideline, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._guideline_ranker.rank = AsyncMock(
        return_value=GuidelineRankingResult(
            [
                RankedGuideline(
                    guideline=guideline,
                    reasoning="Not currently relevant.",
                    is_relevant=False,
                    score=0.2,
                )
            ],
            None,
        )
    )
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [guideline])

    matcher._guideline_recaller.recall.assert_awaited_once_with(context, [guideline])
    matcher._guideline_ranker.rank.assert_awaited_once_with(context, [guideline])
    assert matches == [
        (
            GuidelineMatch(
                guideline=guideline,
                rationale="This may or may not be relevant right now - use your judgment.",
            ),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]


@pytest.mark.asyncio
async def test_that_distilled_guideline_can_still_be_discovered_into_session_by_recall() -> None:
    matcher = _make_batch_matcher()
    guideline = replace(
        create_guideline(
            condition="customer asks for a regulated workflow",
            action="follow the detailed workflow exactly",
            description="This workflow has many details. " * 20,
        ),
        criticality=Criticality.HIGH,
    )
    matcher._guideline_recaller.recall = AsyncMock(
        return_value=GuidelineRecallResult(
            [RecalledGuideline(guideline=guideline, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._guideline_distiller.distill = AsyncMock(
        return_value=GuidelineDistillationResult(
            [
                DistilledGuideline(
                    guideline=guideline,
                    reasoning="No next-step action remains.",
                    is_relevant=False,
                    distilled_action=None,
                )
            ],
            None,
        )
    )
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [guideline])

    matcher._guideline_recaller.recall.assert_awaited_once_with(context, [guideline])
    matcher._guideline_distiller.distill.assert_awaited_once_with(context, [guideline])
    assert matches == [
        (
            GuidelineMatch(
                guideline=guideline,
                rationale="This may or may not be relevant right now - use your judgment.",
            ),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]


@pytest.mark.asyncio
async def test_that_turn_match_takes_precedence_over_session_recall_discovery() -> None:
    matcher = _make_batch_matcher()
    guideline = replace(
        create_guideline(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Criticality.HIGH,
    )
    matcher._guideline_recaller.recall = AsyncMock(
        return_value=GuidelineRecallResult(
            [RecalledGuideline(guideline=guideline, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._guideline_ranker.rank = AsyncMock(
        return_value=GuidelineRankingResult(
            [
                RankedGuideline(
                    guideline=guideline,
                    reasoning="Relevant.",
                    is_relevant=True,
                    score=1.0,
                )
            ],
            None,
        )
    )
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [guideline])

    assert matches == [
        (
            GuidelineMatch(guideline=guideline, rationale="Relevant."),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]


@pytest.mark.asyncio
async def test_that_distilled_turn_match_takes_precedence_over_session_recall_discovery() -> None:
    matcher = _make_batch_matcher()
    guideline = replace(
        create_guideline(
            condition="customer asks for a regulated workflow",
            action="follow the detailed workflow exactly",
            description="This workflow has many details. " * 20,
        ),
        criticality=Criticality.HIGH,
    )
    matcher._guideline_recaller.recall = AsyncMock(
        return_value=GuidelineRecallResult(
            [RecalledGuideline(guideline=guideline, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._guideline_distiller.distill = AsyncMock(
        return_value=GuidelineDistillationResult(
            [
                DistilledGuideline(
                    guideline=guideline,
                    reasoning="Relevant.",
                    is_relevant=True,
                    distilled_action="Follow the detailed workflow exactly.",
                )
            ],
            None,
        )
    )
    context = _context_with_guidelines(guideline, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [guideline])

    matcher._guideline_recaller.recall.assert_awaited_once_with(context, [guideline])
    matcher._guideline_distiller.distill.assert_awaited_once_with(context, [guideline])
    assert matches == [
        (
            GuidelineMatch(
                guideline=guideline,
                rationale="Relevant.",
                metadata={
                    "distilled_action": (
                        f'According to policy "{guideline.title or "Untitled policy"}", '
                        "Follow the detailed workflow exactly.\n"
                        "Apply this only insofar as it remains compatible with the other active "
                        "policies and system instructions."
                    )
                },
            ),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]
