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

from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_signal_proposer import GuidelineSignalProposer


def test_that_guideline_signal_proposer_formats_full_guideline() -> None:
    proposer = object.__new__(GuidelineSignalProposer)

    assert proposer._format_guideline(
        "Refunds",
        GuidelineContent(
            condition="the customer asks for a refund",
            action="review eligibility",
            description="Use the current refund guideline.",
        ),
    ) == (
        "# Refunds\n\n"
        "## When the customer asks for a refund then review eligibility\n\n"
        "Use the current refund guideline."
    )


def test_that_guideline_signal_proposer_formats_partial_guidelines() -> None:
    proposer = object.__new__(GuidelineSignalProposer)

    assert proposer._format_guideline(
        "Observation",
        GuidelineContent(condition="the customer reports a lost card", action=None),
    ) == "# Observation\n\n## Condition: the customer reports a lost card"

    assert proposer._format_guideline(
        "Action Only",
        GuidelineContent(condition="", action="speak concisely"),
    ) == "# Action Only\n\n## Action: speak concisely"

    assert proposer._format_guideline(
        "Title Only",
        GuidelineContent(condition="", action=None),
    ) == "# Title Only"
