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

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from parlant.core.agents import Effort
from parlant.core.common import Criticality, DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.prompt_builder import (
    BuiltInSection,
    EventAdaptationFormat,
    PromptBuilder,
    SectionStatus,
)
from parlant.core.engines.compass.matching.common import (
    add_agent_reasoning,
    aggregate_generation_info,
)
from parlant.core.engines.compass.common import get_dynamic_reasoning_effort_for_matching
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class RankedGuideline:
    guideline: Guideline
    reasoning: str
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class GuidelineRankingResult:
    ranked_guidelines: Sequence[RankedGuideline]
    # Aggregated usage across every per-guideline ranking request sent this call,
    # or None when no requests were sent.
    generation_info: GenerationInfo | None


class GuidelineRankSchema(DefaultBaseModel):
    tldr: str | None = None
    s: int


@dataclass
class GuidelineRankingShot(Shot):
    interaction_events: Sequence[Event]
    # The ranker evaluates a single guideline per prompt, so each shot carries one.
    guideline: GuidelineContent
    expected_result: GuidelineRankSchema


class GuidelineRanker:
    """First-pass guideline filter.

    Before the full guideline-matching stages, a smaller language model is used to
    cheaply filter out guidelines that do not apply for sure. Each guideline is
    evaluated in its own prompt; guidelines that pass this filter go on to the
    later stages of message generation.
    """

    RELEVANCE_SCORE_THRESHOLD = 4
    GUIDELINE_CACHE_BREAKPOINT = "- Guideline: ###"
    REASONING_CACHE_BREAKPOINT = "AGENT'S REASONING SO FAR THIS TURN"
    STAGED_EVENTS_CACHE_BREAKPOINT = "STAGED EVENTS"
    CURRENT_TURN_CACHE_BREAKPOINT = "# CURRENT TURN"

    # Message sources counted as "the agent's reply" — the boundary the cached
    # interaction history is truncated at (everything after is this turn's tail).
    _AGENT_MESSAGE_SOURCES = (
        EventSource.AI_AGENT,
        EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT,
        EventSource.HUMAN_AGENT,
    )

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        schematic_generator: SchematicGenerator[GuidelineRankSchema],
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._schematic_generator = schematic_generator

    async def rank(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> GuidelineRankingResult:
        if not guidelines:
            return GuidelineRankingResult([], None)

        with self._tracer.span("guideline.rank"):
            t_start = asyncio.get_event_loop().time()

            results = await asyncio.gather(
                *(self._rank_guideline(context, guideline) for guideline in guidelines)
            )

            t_end = asyncio.get_event_loop().time()

            return GuidelineRankingResult(
                ranked_guidelines=[ranked for ranked, _ in results],
                generation_info=aggregate_generation_info(
                    [info for _, info in results],
                    total_duration=t_end - t_start,
                ),
            )

    def _should_include_tldr(self, context: EngineContext) -> bool:
        match context.state.dynamic_effort_level:
            case Effort.MIN:
                return False
            case Effort.LOW:
                return False
            case Effort.MEDIUM:
                return True
            case Effort.HIGH:
                return True
            case Effort.MAX:
                return True

    async def _rank_guideline(
        self,
        context: EngineContext,
        guideline: Guideline,
    ) -> tuple[RankedGuideline, GenerationInfo]:
        prompt = self._build_prompt(context, guideline, shots=await self.shots())

        inference = await self._schematic_generator.generate(
            prompt=prompt,
            hints={
                "reasoning_effort": get_dynamic_reasoning_effort_for_matching(context),
                "cache": {
                    "key": self._cache_key(context),
                    "breakpoint": self._cache_breakpoint(context),
                },
                "hedge_timeout": 5.0,
            },
        )

        score = inference.content.s

        # A score of 1-2 means "filtered out for sure"; 3+ ("maybe" and up) passes
        # the first-pass filter on to the next stages.
        return (
            RankedGuideline(
                guideline=guideline,
                reasoning=inference.content.tldr
                or f"This guideline ranked {score * 2} out of 10 in relevance to your next response.",
                is_relevant=score >= self.RELEVANCE_SCORE_THRESHOLD,
                score=float(score) / 5.0,
            ),
            inference.info,
        )

    def _build_reasoning(self, output: GuidelineRankSchema) -> str:
        if output.tldr:
            match output.s:
                case s if s < 1:
                    return f"NOT relevant. {output.tldr}"
                case 1 | 2:
                    return f"NOT relevant. {output.tldr}"
                case 3:
                    return f"Possibly relevant. {output.tldr}"
                case 4:
                    return f"Quite relevant. {output.tldr}"
                case 5:
                    return f"Very relevant. {output.tldr}"
                case _:
                    return f"Relevance score overflow: {output.s * 2} out of 10. {output.tldr}"
        else:
            return f"This guideline ranked {output.s * 2} out of 10 in relevance to your next response."

    def _cache_key(self, context: EngineContext) -> str:
        # Namespace the provider cache per session AND component, so components
        # that cache concurrently (e.g. within a matching batch) never clobber a
        # shared entry.
        return f"{context.session.id}.guideline-ranker"

    def _cache_breakpoint(self, context: EngineContext) -> str:
        # The shared prefix always ends right before the per-turn "# CURRENT TURN"
        # section, so the cache boundary is fixed — independent of which per-turn
        # sections (latest message, staged events, reasoning) happen to be present.
        return self.CURRENT_TURN_CACHE_BREAKPOINT

    async def warm_up(self, context: EngineContext) -> GenerationInfo | None:
        """Warm the generator's cache for the ranker's shared prompt prefix.

        `rank` fans out one request per guideline concurrently, each repeating the
        shared prefix. The cache hint lets providers that support explicit caching
        (Gemini) cache everything before the first dynamic tail section, so the
        per-guideline fan-out sends only the live suffix. Providers without explicit
        caching ignore the hint. The same reasoning_effort is used so the cached
        variant matches. Best-effort: warming failures must not break preparation."""
        with self._tracer.span("guideline.prefill"):
            try:
                prompt = self._build_prompt(
                    context,
                    guideline=self._cache_prefill_guideline(),
                    shots=await self.shots(),
                )
                inference = await self._schematic_generator.generate(
                    prompt=prompt,
                    hints={
                        "reasoning_effort": get_dynamic_reasoning_effort_for_matching(context),
                        "cache": {
                            "key": self._cache_key(context),
                            "breakpoint": self._cache_breakpoint(context),
                        },
                    },
                )
                return inference.info
            except Exception as exc:
                self._logger.warning(f"Guideline ranker prefill failed (continuing): {exc}")
                return None

    async def shots(self) -> Sequence[GuidelineRankingShot]:
        return await shot_collection.list()

    def _cache_prefill_guideline(self) -> Guideline:
        return Guideline(
            id=GuidelineId("cache-prefill-guideline"),
            creation_utc=datetime.now(timezone.utc),
            modified_utc=datetime.now(timezone.utc),
            content=GuidelineContent(
                condition="the cache is being warmed before guideline ranking",
                action="return a low relevance score",
            ),
            enabled=True,
            tags=[],
            metadata={},
            criticality=Criticality.LOW,
        )

    def _format_shots(self, context: EngineContext, shots: Sequence[GuidelineRankingShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(context, shot)}"
            for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, context: EngineContext, shot: GuidelineRankingShot) -> str:
        def adapt_event(e: Event) -> JSONSerializable:
            source_map: dict[EventSource, str] = {
                EventSource.CUSTOMER: "user",
                EventSource.CUSTOMER_UI: "frontend_application",
                EventSource.HUMAN_AGENT: "human_service_agent",
                EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT: "ai_agent",
                EventSource.AI_AGENT: "ai_agent",
                EventSource.SYSTEM: "system-provided",
            }

            return {
                "event_kind": e.kind.value,
                "event_source": source_map[e.source],
                "data": e.data,
            }

        formatted_shot = ""
        if shot.interaction_events:
            formatted_shot += f"""
- **Interaction Events**:
{json.dumps([adapt_event(e) for e in shot.interaction_events], indent=2)}

"""

        formatted_shot += f"""
- **Guideline**:
{_format_guideline(None, shot.guideline.condition, shot.guideline.action, shot.guideline.description)}

"""

        expected_result = shot.expected_result.model_dump(mode="json")
        if not self._should_include_tldr(context) or expected_result.get("tldr") is None:
            expected_result.pop("tldr", None)

        formatted_shot += f"""
- **Expected Result**:
```json
{json.dumps(expected_result, indent=2)}
```
"""

        return formatted_shot

    def _build_prompt(
        self,
        context: EngineContext,
        guideline: Guideline,
        shots: Sequence[GuidelineRankingShot],
    ) -> PromptBuilder:
        # Start from the cross-turn-stable shared prefix, then append the
        # turn-varying tail. The tail opens with a fixed "# CURRENT TURN" marker (the
        # cache breakpoint); everything before it is the cacheable prefix.
        builder = self._build_shared_prompt(context, shots)

        # The customer's latest message(s) were split out of the cached history (see
        # `_build_shared_prompt`); render them here, in the live suffix, so the cached
        # prefix stays identical from one turn's `prefill` to the next turn's matching.
        _, trailing_events = self._split_interaction_at_last_agent_reply(context.interaction.events)
        builder.add_section(
            name="guideline-ranker-current-turn",
            template="""
# CURRENT TURN
{current_turn_text}
""",
            props={"current_turn_text": self._format_current_turn(builder, trailing_events)},
            status=SectionStatus.ACTIVE,
        )

        # Per-step reasoning goes in the tail (not the cached shared prefix) so the
        # cache stays valid while the matching tracks the agent's evolving reasoning.
        add_agent_reasoning(builder, context.state.reasoning_steps)

        # TODO: It's problematic that tool events aren't on a shared timeline
        # with the reasoning steps. Fix this at some point.
        builder.add_staged_tool_events(context.state.tool_events)

        builder.add_section(
            name=BuiltInSection.GUIDELINES,
            template="""
- Guideline: ###
{guideline_text}
###
""",
            props={
                "guideline_text": _format_guideline(
                    guideline.title,
                    guideline.content.condition,
                    guideline.content.action,
                    guideline.content.description,
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        return builder

    def _build_shared_prompt(
        self,
        context: EngineContext,
        shots: Sequence[GuidelineRankingShot],
    ) -> PromptBuilder:
        """The cross-turn-stable head of the ranker prompt (instructions, shots,
        agent + customer identity). Excludes turn-varying context, so it stays
        byte-identical across turns — the prefix `prefill` warms."""
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-ranker-general-instructions",
            template="""
# GENERAL INSTRUCTIONS

In our system, a conversational multi-turn AI agent's behavior is controlled by a set of guidelines.
Some guidelines are condition/action instructions: when the condition applies, the agent is directed to take its associated action.
Other guidelines are policy-style instructions: they may have no condition or action, and their title and policy text describe the rule, constraint, or information that may govern the conversation.
""",
            props={},
        )
        builder.add_section(
            name="guideline-ranker-task-description",
            template="""
# Task Description

Act as a first-pass filter to screen out clearly irrelevant guidelines. You are NOT making the final determination - a human judge will review every guideline that isn't clearly irrelevant and decide whether it actually applies.

Output an integer score from 1 to 5 indicating how relevant the guideline is to the most recent state of the conversation:
- 1 - not relevant at all, at no point in the conversation
- 2 - only very loosely related to the subject of the conversation
- 3 - partially relevant: was previously relevant but not clearly right now, or only part of the guideline matches
- 4 - requires further checking by the human: a partial match, but one that clearly justifies further inspection
- 5 - clearly matches

Scores of 1 and 2 mean the guideline will be filtered out for sure; 3 means maybe; 4 and 5 mean it should continue on to the next stages. When in doubt, lean towards a higher score - the human will make the final call.

You will be given the context of the customer, the interaction so far (including tool calls and results), contextual information (variables) about the customer and the current interaction session, and other information. You shall use all of that information to evaluate the relevance of the guideline to the current state of the interaction, but you should focus more on the most recent state of the interaction.

Important considerations:
1. Focus on recency: Evaluate relevance based on the latest part of the conversation, particularly the most recent customer message.
2. Semantic evaluation: Assess the actual meaning of the guideline, not just keyword matching.
3. Context matters: Consider the full context and intent behind the user's message, including tool results, context variables, capabilities etc'.
4. For condition/action guidelines, do not rank solely because the action sounds useful; rank by whether the condition applies. The action and details may be used to interpret the condition.
5. For policy-style guidelines without a condition, rank by whether the title and policy/details text govern the current situation, the customer's request, the agent's next response, or a tool/result currently in play.
6. Match based on entire context: a guideline may be matched based on the entire context of the interaction. It may be relevant even if it has nothing to do with the latest customer message.
""",
            props={},
        )
        builder.add_section(
            name="guideline-ranker-examples",
            template="""
#Examples of Guideline Ranking Evaluations:

{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(context, shots),
                "shots": shots,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)

        # Without a tldr field to capture its reasoning, the model tends to append
        # an unsolicited explanation after the JSON; tell it not to so we don't pay
        # for output tokens we discard. Kept as the final note (recency).
        # (Providers that enforce structured output - OpenAI json_object, Gemini
        # function-calling - ignore this harmlessly.)
        json_only_note = (
            ""
            if self._should_include_tldr(context)
            else "\nRespond with ONLY the JSON object above - no explanation, reasoning, or other text."
        )

        builder.add_section(
            name="guideline-ranker-output-format",
            template="""
# OUTPUT FORMAT

- Evaluate the guideline by filling in the details in the following structure:
```json
{result_structure_text}
```
{json_only_note}
""",
            props={
                "result_structure_text": self._format_output(context),
                "json_only_note": json_only_note,
            },
        )

        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        builder.add_capabilities_for_guideline_matching(context.state.capabilities)
        if context.state.session_summary:
            builder.add_session_summary(context.state.session_summary)
        # Cache only the history through the agent's last reply; the trailing customer
        # message(s) are rendered in the per-turn tail (see `_build_prompt`), keeping
        # this prefix byte-identical from one turn's `prefill` to the next turn's
        # matching (which only appends the new customer message).
        cached_events, _ = self._split_interaction_at_last_agent_reply(context.interaction.events)
        builder.add_interaction_history(
            cached_events,
            format=EventAdaptationFormat.ROLE_SCRIPT,
        )

        return builder

    def _split_interaction_at_last_agent_reply(
        self,
        events: Sequence[Event],
    ) -> tuple[Sequence[Event], Sequence[Event]]:
        """Split the interaction into the cross-turn-stable head (through the agent's
        last reply) and the per-turn tail (the customer message(s) that arrived after
        it). Caching only the head lets the prefix `prefill` warms at the end of a turn
        be reused by the next turn's matching, which merely appends the new message."""
        cutoff = 0
        for index, event in enumerate(events):
            if event.kind == EventKind.MESSAGE and event.source in self._AGENT_MESSAGE_SOURCES:
                cutoff = index + 1
        return events[:cutoff], events[cutoff:]

    def _format_current_turn(
        self,
        builder: PromptBuilder,
        events: Sequence[Event],
    ) -> str:
        rendered = [
            builder.adapt_event(event, format=EventAdaptationFormat.ROLE_SCRIPT)
            for event in events
            if event.kind != EventKind.STATUS
        ]
        if not rendered:
            return "(No new customer message has arrived yet.)"
        return (
            "The following continues the interaction history above — the customer's "
            "most recent message(s):\n" + "\n".join(rendered)
        )

    def _format_output(self, context: EngineContext) -> str:
        result: dict[str, JSONSerializable] = {}

        if self._should_include_tldr(context):
            result["tldr"] = (
                "<A brief, one-line summary of why the guideline is or "
                "isn't relevant to the most recent state of the interaction>"
            )

        result["s"] = (
            "<The integer score from 1 to 5 indicating how relevant the guideline is, per the scale described above>"
        )

        return json.dumps(result, indent=4)


def _format_guideline(
    title: str | None,
    condition: str,
    action: str | None,
    description: str | None = None,
) -> str:
    title = (title or "").strip()
    condition = (condition or "").strip()
    action = (action or "").strip()
    description = (description or "").strip()

    sections: list[str] = []

    if title:
        sections.append(f"Title: {title}")

    if condition and action:
        sections.append(f"When: {condition}\nThen: {action}")
    elif condition:
        sections.append(f"Condition: {condition}")
    elif action:
        sections.append(f"Action: {action}")

    if description:
        label = "Policy" if not condition and not action else "Details"
        sections.append(f"{label}: {description}")

    return "\n\n".join(sections)


def _make_event(e_id: str, source: EventSource, message: str) -> Event:
    return Event(
        id=EventId(e_id),
        source=source,
        kind=EventKind.MESSAGE,
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        offset=0,
        trace_id="",
        data={"message": message},
        metadata={},
        deleted=False,
    )


example_1_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "My throat is dry, do you have anything that can help me with that?",
    ),
]

example_1_guideline = GuidelineContent(
    condition="The customer might be thirsty",
    action=None,
)

example_1_expected = GuidelineRankSchema(
    tldr="Dry throat suggests potential thirst",
    s=4,
)


example_2_events = [
    _make_event(
        "21",
        EventSource.CUSTOMER,
        "Can delivery arrive here before 8 PM?",
    ),
]

example_2_guideline = GuidelineContent(
    condition="The customer asks for late night (10 PM or later) delivery",
    action=None,
)

example_2_expected = GuidelineRankSchema(
    tldr="8 PM is not late night (10 PM or later)",
    s=1,
)


example_3_events = [
    _make_event(
        "31",
        EventSource.CUSTOMER,
        "I'll consult with my partner about getting Platinum",
    ),
]

example_3_guideline = GuidelineContent(
    condition="Expressed unsureness about getting a product",
    action=None,
)

example_3_expected = GuidelineRankSchema(
    tldr="The customer needs consultation before getting Platinum, which might be a product, so it's best to forward this to the human evaluator",
    s=4,
)


_baseline_shots: Sequence[GuidelineRankingShot] = [
    GuidelineRankingShot(
        description="",
        interaction_events=example_1_events,
        guideline=example_1_guideline,
        expected_result=example_1_expected,
    ),
    GuidelineRankingShot(
        description="",
        interaction_events=example_2_events,
        guideline=example_2_guideline,
        expected_result=example_2_expected,
    ),
    GuidelineRankingShot(
        description="",
        interaction_events=example_3_events,
        guideline=example_3_guideline,
        expected_result=example_3_expected,
    ),
]

shot_collection = ShotCollection[GuidelineRankingShot](_baseline_shots)
