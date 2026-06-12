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

from typing import Sequence

from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    base_test_that_guidelines_are_ranked_correctly,
    create_engine_context,
    create_guideline,
)


@fixture
def ranker(container: Container) -> GuidelineRanker:
    return container[GuidelineRanker]


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
