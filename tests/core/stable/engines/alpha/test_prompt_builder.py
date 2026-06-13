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

from datetime import datetime, timezone

from parlant.core.common import Criticality, generate_id
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.tools import Tool, ToolId, ToolOverlap


def _tool(name: str, description: str, *, consequential: bool = False) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters={},
        required=[],
        consequential=consequential,
        overlap=ToolOverlap.NONE,
    )


def _guideline(
    condition: str,
    action: str,
    *,
    description: str | None = None,
    criticality: Criticality = Criticality.MEDIUM,
) -> Guideline:
    now = datetime.now(timezone.utc)
    return Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        content=GuidelineContent(condition=condition, action=action, description=description),
        enabled=True,
        tags=[],
        metadata={},
        criticality=criticality,
    )


def _match(
    condition: str, action: str, criticality: Criticality = Criticality.MEDIUM
) -> GuidelineMatch:
    return GuidelineMatch(
        guideline=_guideline(condition, action, criticality=criticality), rationale="because"
    )


# ───────────────────── guideline instructions vs. list ──────────────────────


def test_that_guideline_instructions_explain_how_to_follow_without_listing_guidelines() -> None:
    prompt = PromptBuilder().add_guideline_instructions().build()

    assert "RELEVANT DOMAIN PROTOCOL INSTRUCTIONS" in prompt
    assert "You may choose not to follow an instruction only" in prompt
    # The explanation must not contain any of the actual matched guidelines.
    assert "Instruction #" not in prompt


def test_that_matched_guidelines_list_the_guidelines_without_the_explanation() -> None:
    match = _match("the customer asks about toppings", "list the available toppings")
    guidelines = {match.guideline.id: match.guideline}

    prompt = PromptBuilder().add_matched_guidelines([match], {}, guidelines).build()

    assert "Instruction #1)" in prompt
    assert "list the available toppings" in prompt
    # The how/when explanation belongs to add_guideline_instructions, not here.
    assert "You may choose not to follow an instruction only" not in prompt


def test_that_matched_guidelines_lead_with_a_skip_if_already_satisfied_rule() -> None:
    # Co-located with the list (turn-level), so the anti-repetition rule has the
    # same recency as the guidelines themselves rather than living far up in the
    # cached system block.
    match = _match("the customer asks about toppings", "list the available toppings")
    guidelines = {match.guideline.id: match.guideline}

    prompt = PromptBuilder().add_matched_guidelines([match], {}, guidelines).build()

    assert "ALREADY satisfied" in prompt
    assert "skip it silently" in prompt
    # The assessment must be internal — no narrated "let me check the guidelines" preamble.
    assert "This whole assessment is INTERNAL" in prompt


def test_that_matched_guidelines_renders_an_empty_state_when_there_are_no_matches() -> None:
    prompt = PromptBuilder().add_matched_guidelines([], {}, {}).build()

    assert "Instruction #" not in prompt
    assert "No special behavioral instructions" in prompt


def test_that_matched_guidelines_list_their_associated_tools() -> None:
    match = _match("the customer asks about the weather", "tell them the forecast")
    guidelines = {match.guideline.id: match.guideline}
    tool_enabled = {match: [ToolId(service_name="local", tool_name="get_weather")]}

    prompt = PromptBuilder().add_matched_guidelines([], tool_enabled, guidelines).build()

    assert "tell them the forecast" in prompt
    assert "get_weather" in prompt
    assert "consider using" in prompt.lower()


def test_that_matched_low_criticality_guidelines_list_their_associated_tools() -> None:
    match = _match("the customer greets you", "greet back", criticality=Criticality.LOW)
    guidelines = {match.guideline.id: match.guideline}
    tool_enabled = {match: [ToolId(service_name="local", tool_name="say_hello")]}

    prompt = (
        PromptBuilder().add_matched_low_criticality_guidelines([], tool_enabled, guidelines).build()
    )

    assert "greet back" in prompt
    assert "say_hello" in prompt
    assert "consider using" in prompt.lower()


# ──────────────── low-criticality instructions vs. list ─────────────────────


def test_that_low_criticality_instructions_explain_without_listing() -> None:
    prompt = PromptBuilder().add_low_criticality_guideline_instructions().build()

    assert "general principles" in prompt
    assert "you may ignore" in prompt.lower()
    assert "When always, then" not in prompt


def test_that_matched_low_criticality_guidelines_list_the_principles() -> None:
    match = _match(
        "the customer is chatty",
        "keep it brief",
        criticality=Criticality.LOW,
    )
    guidelines = {match.guideline.id: match.guideline}

    prompt = PromptBuilder().add_matched_low_criticality_guidelines([match], {}, guidelines).build()

    assert "keep it brief" in prompt


# ─────────────────────── system-wide instructions ───────────────────────────


def test_that_system_wide_guidelines_list_all_instructions_with_their_details() -> None:
    g1 = _guideline("the customer asks about toppings", "list the available toppings")
    g2 = _guideline(
        "the customer wants a refund",
        "follow the refund policy",
        description="Refunds are issued within 30 days of purchase.",
    )

    prompt = PromptBuilder().add_system_wide_guidelines([g1, g2]).build()

    # Every instruction is listed (not just a matched subset), with its condition,
    # action, and details.
    assert "Instruction #1)" in prompt
    assert "Instruction #2)" in prompt
    assert "list the available toppings" in prompt
    assert "follow the refund policy" in prompt
    assert "Refunds are issued within 30 days of purchase." in prompt
    # Referred to as "instructions", never "guidelines".
    assert "instruction" in prompt.lower()
    assert "guideline" not in prompt.lower()


def test_that_system_wide_guidelines_render_nothing_when_there_are_no_instructions() -> None:
    prompt = PromptBuilder().add_system_wide_guidelines([]).build()

    assert "Instruction #" not in prompt


# ─────────────────────────── tool descriptions ──────────────────────────────


def test_that_tool_descriptions_list_relevant_tools_framed_as_optional() -> None:
    prompt = (
        PromptBuilder()
        .add_tool_descriptions([_tool("get_weather", "Get the current weather for a city.")])
        .build()
    )

    assert "RELEVANT TOOLS" in prompt
    # Non-consequential tools are listed by name only (no description line).
    assert "- get_weather" in prompt
    # Framed as optional — the agent should consider them but doesn't have to.
    assert "positively consider using the following tools" in prompt
    assert "not required to use any of them" in prompt


def test_that_consequential_tools_carry_a_caution_note() -> None:
    prompt = (
        PromptBuilder()
        .add_tool_descriptions(
            [
                _tool("get_weather", "Get the weather."),
                _tool("charge_card", "Charge the customer's card.", consequential=True),
            ]
        )
        .build()
    )

    # The consequential note attaches only to the consequential tool.
    assert "CONSEQUENTIAL" in prompt
    assert "confirm with the user" in prompt
    charge_line = next(line for line in prompt.splitlines() if "charge_card" in line)
    weather_line = next(line for line in prompt.splitlines() if "get_weather" in line)
    assert "CONSEQUENTIAL" in charge_line
    assert "CONSEQUENTIAL" not in weather_line


def test_that_tool_descriptions_render_an_empty_state_when_there_are_no_tools() -> None:
    prompt = PromptBuilder().add_tool_descriptions([]).build()

    assert "No tools have been specifically highlighted" in prompt
    assert "RELEVANT TOOLS" not in prompt
