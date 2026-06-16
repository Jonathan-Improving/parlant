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

from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from typing import Any, Optional, TypeAlias

from parlant.core.agents import Effort
from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext as _EngineContext
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.journeys import Journey, JourneyId
from parlant.core.common import Criticality
from parlant.core.tools import Tool, ToolId


_EFFORT_ORDER: dict[Effort, int] = {
    Effort.MIN: 0,
    Effort.LOW: 1,
    Effort.MEDIUM: 2,
    Effort.HIGH: 3,
    Effort.MAX: 4,
}


@dataclass(frozen=True)
class IterationState:
    """State of a single iteration in the response process"""

    matched_guidelines: list[GuidelineMatch]
    ruled_out: list[GuidelineMatch]
    resolved_guidelines: list[GuidelineMatch]
    tool_insights: ToolInsights
    executed_tools: list[ToolId]


@dataclass
class ResponseState:
    agent_effort: Effort = Effort.MEDIUM
    ordinary_guideline_matches: list[GuidelineMatch] = field(default_factory=list)
    tool_enabled_guideline_matches: dict[GuidelineMatch, list[ToolId]] = field(default_factory=dict)
    # tools the matched guidelines enabled this turn (described in the prompt)
    matched_tools: list[Tool] = field(default_factory=list)
    # all the agent's candidate tools, ranked by relevance to the conversation
    agent_tool_pool: list[Tool] = field(default_factory=list)
    # final catalog offered to the model (matched_tools ∪ top of the pool, capped, by name)
    available_tools: list[Tool] = field(default_factory=list)
    tool_ids_by_name: dict[str, ToolId] = field(default_factory=dict)  # to run a tool by its name

    # The agent's reasoning from each step of the response loop so far this turn,
    # in order. The loop appends to it after every step; the matching components
    # (ranker, distiller) feed it into their per-guideline prompts so each step's
    # evaluation is aware of what the agent has already concluded. Empty on the
    # initial match (no steps have run yet).
    reasoning_steps: list[str] = field(default_factory=list)

    # Durable summary of session events before the latest compaction marker.
    # Empty means no compaction summary is active for this loaded interaction.
    session_summary: str = ""

    # Reviewer-provided replacement reasoning when pending tool calls would breach
    # policy. Empty means no breach was found, or the reviewer has not run yet.
    step_notes: str = ""

    # Reviewer-provided summary of what the agent still needs to do before
    # responding to the user. Empty means the reviewer has not run yet.
    todo: str = ""

    # Per-turn signals the matcher precomputes (once) so its per-guideline strategy
    # selection can stay synchronous: guidelines that carry tools, and guidelines
    # that participate in a dependency relationship.
    guideline_ids_with_tools: set[GuidelineId] = field(default_factory=set)
    guideline_ids_with_dependencies: set[GuidelineId] = field(default_factory=set)

    # Tools attached to a guideline's action (e.g. a journey distilled into a
    # guideline carries the tools of its tool-using steps). Read by the distiller
    # to surface what each tool does; empty for guidelines without attached tools.
    tools_by_guideline: dict[GuidelineId, set[tuple[ToolId, Tool]]] = field(default_factory=dict)

    # TODO: Remove what isn't needed
    context_variables: list[tuple[ContextVariable, ContextVariableValue]] = field(
        default_factory=list
    )
    glossary_terms: set[Term] = field(default_factory=set)
    capabilities: list[Capability] = field(default_factory=list)
    journeys: list[Journey] = field(default_factory=list)
    journey_paths: dict[JourneyId, list[Optional[str]]] = field(default_factory=dict)
    tool_events: list[EmittedEvent] = field(default_factory=list)
    tool_insights: ToolInsights = field(default_factory=ToolInsights)
    prepared_to_respond: bool = False
    message_events: list[EmittedEvent] = field(default_factory=list)
    usable_guidelines: list[Guideline] = field(default_factory=list)
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)
    iterations: list[IterationState] = field(default_factory=list)

    @property
    def ordinary_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.ordinary_guideline_matches]

    @property
    def tool_enabled_guidelines(self) -> list[Guideline]:
        return [gp.guideline for gp in self.tool_enabled_guideline_matches.keys()]

    @property
    def guidelines(self) -> list[Guideline]:
        return self.ordinary_guidelines + self.tool_enabled_guidelines

    @cached_property
    def dynamic_effort_level(self) -> Effort:
        """Resolve effective effort from the agent default and matched guideline effort levels."""
        efforts = [
            self.agent_effort,
            *(
                match.guideline.effort
                for match in chain(
                    self.ordinary_guideline_matches,
                    self.tool_enabled_guideline_matches.keys(),
                )
                if match.guideline.effort is not None
            ),
        ]

        return max(efforts, key=lambda effort: _EFFORT_ORDER[effort])

    @cached_property
    def has_matched_high_criticality_guidelines(self) -> bool:
        return any(
            match.guideline.criticality == Criticality.HIGH
            for match in chain(
                self.ordinary_guideline_matches,
                self.tool_enabled_guideline_matches.keys(),
            )
        )

    def invalidate_cached_properties(self) -> None:
        self.__dict__.pop("dynamic_effort_level", None)
        self.__dict__.pop("has_matched_high_criticality_guidelines", None)


# The compass engine sees its own ResponseState typed through context.state.
EngineContext: TypeAlias = _EngineContext[ResponseState]
