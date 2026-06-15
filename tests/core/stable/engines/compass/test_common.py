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

from parlant.core.agents import Effort
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.compass.common import get_dynamic_effort_level
from parlant.core.engines.compass.response_state import ResponseState

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_agent,
    create_engine_context,
    create_guideline,
)


def test_that_dynamic_effort_is_agent_effort_when_no_matched_guideline_has_effort() -> None:
    context = create_engine_context(
        conversation=[],
        agent=create_agent(),
    )
    context.state = ResponseState()

    assert get_dynamic_effort_level(context) == Effort.MEDIUM


def test_that_dynamic_effort_uses_maximum_matched_guideline_effort() -> None:
    context = create_engine_context(
        conversation=[],
        agent=replace(create_agent(), effort=Effort.LOW),
    )
    high_effort_guideline = replace(
        create_guideline("the user requests a regulated action"), effort=Effort.HIGH
    )
    low_effort_guideline = replace(create_guideline("the user greets the agent"), effort=Effort.MIN)
    context.state = ResponseState(
        ordinary_guideline_matches=[
            GuidelineMatch(guideline=high_effort_guideline, rationale="relevant"),
            GuidelineMatch(guideline=low_effort_guideline, rationale="also relevant"),
        ],
    )

    assert get_dynamic_effort_level(context) == Effort.HIGH
