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

from typing import Any, Sequence, cast

from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.guidelines import Guideline
from parlant.core.loggers import StdoutLogger
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import (
    base_test_that_guidelines_are_ranked_correctly,
    create_engine_context,
    create_guideline,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


def _make_ranker() -> GuidelineRanker:
    # The prompt-building helpers (`_build_prompt`/`_build_shared_prompt`/
    # `_cache_breakpoint`) only read the context, so the schematic generator isn't
    # exercised — and this avoids the (separately broken) container fixture.
    tracer = LocalTracer()
    return GuidelineRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        schematic_generator=cast(Any, None),
    )


def _cached_prefix(ranker: GuidelineRanker, context: EngineContext, guideline: Guideline) -> str:
    # The portion Gemini would cache: everything before the breakpoint marker (the
    # marker itself and the rest are the live, per-turn suffix).
    prompt = ranker._build_prompt(context, guideline, shots=[]).build()  # type: ignore[arg-type]
    breakpoint_marker = ranker._cache_breakpoint(context)
    index = prompt.find(breakpoint_marker)
    assert index != -1, f"cache breakpoint {breakpoint_marker!r} not found in prompt"
    return prompt[:index]


def test_that_the_ranker_prompt_includes_the_agent_reasoning_but_keeps_it_out_of_the_cached_prefix(
    ranker: GuidelineRanker,
) -> None:
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")]
    )
    context.state = ResponseState(
        reasoning_steps=[
            "The customer asked which toppings are available.",
            "I should list the available toppings from current stock.",
        ],
    )
    shots: Sequence[object] = []  # shots are irrelevant to the reasoning section

    prompt = ranker._build_prompt(context, guideline, shots).build()  # type: ignore[arg-type]
    assert "list the available toppings from current stock" in prompt

    # Caching invariant: per-step reasoning must NOT enter the cached shared prefix.
    shared = ranker._build_shared_prompt(context, shots).build()  # type: ignore[arg-type]
    assert "list the available toppings from current stock" not in shared


def test_that_the_latest_customer_message_is_kept_out_of_the_cached_prefix() -> None:
    # The latest customer message changes every turn, so it must live in the live
    # suffix — not the cached prefix — or the prefix hash shifts each turn and the
    # warmed cache is never reused.
    ranker = _make_ranker()
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "ZZQ_LATEST what toppings do you have?"),
        ]
    )
    context.state = ResponseState()

    prompt = ranker._build_prompt(context, guideline, shots=[]).build()  # type: ignore[arg-type]
    prefix = _cached_prefix(ranker, context, guideline)

    assert "ZZQ_LATEST" in prompt  # present in the full prompt (the live suffix)
    assert "ZZQ_LATEST" not in prefix  # but NOT in the cached prefix


def test_that_the_cached_prefix_is_identical_with_and_without_a_trailing_customer_message() -> None:
    # The crux: the prefix `prefill` warms at the end of a turn (history ending in
    # the agent's reply) must be byte-identical to the prefix the next turn's
    # matching builds (that same history plus the new customer message).
    ranker = _make_ranker()
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    base = [
        (EventSource.CUSTOMER, "hi"),
        (EventSource.AI_AGENT, "hello, how can I help?"),
    ]

    prefill_context = create_engine_context(conversation=base)
    prefill_context.state = ResponseState()

    matching_context = create_engine_context(
        conversation=[*base, (EventSource.CUSTOMER, "what toppings do you have?")]
    )
    matching_context.state = ResponseState()

    assert _cached_prefix(ranker, prefill_context, guideline) == _cached_prefix(
        ranker, matching_context, guideline
    )


def test_that_all_trailing_customer_messages_are_excluded_from_the_cached_prefix() -> None:
    # Truncation happens at the last AI-agent message, so multiple customer messages
    # sent back-to-back before the agent replies all land in the suffix.
    ranker = _make_ranker()
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )
    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "hi"),
            (EventSource.AI_AGENT, "hello, how can I help?"),
            (EventSource.CUSTOMER, "ZZQ_FIRST a question"),
            (EventSource.CUSTOMER, "ZZQ_SECOND and another"),
        ]
    )
    context.state = ResponseState()

    prefix = _cached_prefix(ranker, context, guideline)

    assert "ZZQ_FIRST" not in prefix
    assert "ZZQ_SECOND" not in prefix


GUIDELINES_DICT: dict[str, dict[str, str]] = {
    "ask_toppings": {
        "condition": "the customer asks about toppings",
        "action": "list the available toppings",
    },
}


def test_that_a_guideline_ranker_can_be_created(ranker: GuidelineRanker) -> None:
    assert ranker is not None


async def test_that_a_relevant_guideline_is_ranked_as_relevant(ranker: GuidelineRanker) -> None:
    await base_test_that_guidelines_are_ranked_correctly(
        ranker,
        GUIDELINES_DICT,
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
        conversation_guideline_names=["ask_toppings"],
        relevant_guideline_names=["ask_toppings"],
        irrelevant_guideline_names=[],
    )
