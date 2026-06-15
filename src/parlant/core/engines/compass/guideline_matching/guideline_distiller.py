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
from collections.abc import Set
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from parlant.core.engines.alpha.tool_calling.common import get_tool_spec
from parlant.core.engines.compass.guideline_matching.common import (
    add_agent_reasoning,
    aggregate_generation_info,
    reasoning_effort_for,
)
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import Event, EventId, EventKind, EventSource
from parlant.core.shots import Shot, ShotCollection
from parlant.core.tools import Tool, ToolId
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class DistilledGuideline:
    guideline: Guideline
    reasoning: str
    is_relevant: bool
    distilled_action: Optional[str]


@dataclass(frozen=True)
class GuidelineDistillationResult:
    distilled_guidelines: Sequence[DistilledGuideline]
    # Aggregated usage across every per-guideline distillation request this call,
    # or None when no requests were sent.
    generation_info: GenerationInfo | None


class GuidelineDistillSchema(DefaultBaseModel):
    reasoning: str
    is_relevant: bool
    distilled_action: Optional[str] = None


@dataclass
class GuidelineDistillationShot(Shot):
    interaction_events: Sequence[Event]
    # The distiller evaluates a single guideline per prompt, so each shot carries one.
    guideline: GuidelineContent
    expected_result: GuidelineDistillSchema


class GuidelineDistiller:
    """Distills a (potentially verbose) guideline down to the guidance relevant right now.

    A guideline's action and detailed instructions may describe many things across
    different situations - an ordered sequence of steps, or a set of rules and
    information. The distiller evaluates whether the guideline currently applies and, if
    so, extracts exactly the guidance relevant to the next agent response: the next step
    for a sequential action, or every applicable rule for a policy. Each guideline is
    evaluated in its own prompt.
    """

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        schematic_generator: SchematicGenerator[GuidelineDistillSchema],
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._schematic_generator = schematic_generator

    async def distill(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> GuidelineDistillationResult:
        if not guidelines:
            return GuidelineDistillationResult([], None)

        with self._tracer.span("guideline.distill"):
            if len(guidelines) > 1:
                # Warm-then-fan-out (see GuidelineRanker.rank): distill the first
                # guideline and AWAIT it so the shared prompt prefix is cached, then
                # fan out the rest concurrently against the warm cache.
                first = await self._distill_guideline(context, guidelines[0])
                rest = await asyncio.gather(
                    *(self._distill_guideline(context, guideline) for guideline in guidelines[1:])
                )
                results = [first, *rest]
            else:
                results = [await self._distill_guideline(context, guidelines[0])]

            return GuidelineDistillationResult(
                distilled_guidelines=[distilled for distilled, _ in results],
                generation_info=aggregate_generation_info([info for _, info in results]),
            )

    async def _distill_guideline(
        self,
        context: EngineContext,
        guideline: Guideline,
    ) -> tuple[DistilledGuideline, GenerationInfo]:
        prompt = self._build_prompt(context, guideline, shots=await self.shots())

        inference = await self._schematic_generator.generate(
            prompt=prompt,
            hints={
                "reasoning_effort": reasoning_effort_for(context),
                "cache": {"action": "load", "key": self._cache_key(context)},
            },
        )

        return (
            DistilledGuideline(
                guideline=guideline,
                reasoning=inference.content.reasoning,
                is_relevant=inference.content.is_relevant,
                distilled_action=inference.content.distilled_action,
            ),
            inference.info,
        )

    def _cache_key(self, context: EngineContext) -> str:
        # Namespace the provider cache per session+nonce AND component, so components
        # that cache concurrently never clobber a shared entry. The nonce (minted in
        # CompassEngine.initialize, shared across the session's turns) is stable across
        # the store (prefill) / load (distill) pair.
        return f"{context.session.id}.{context.state.cache_nonce}.guideline-distiller"

    async def prefill(self, context: EngineContext) -> GenerationInfo | None:
        """Warm the generator's cache for the distiller's shared prompt prefix, so
        the per-guideline fan-out's `cache: load` requests hit it. The throwaway
        generation triggers the `cache: store`. Best-effort: warming failures must
        not break preparation. See :meth:`GuidelineRanker.prefill`."""
        with self._tracer.span("guideline.distill-prefill"):
            try:
                prompt = self._build_shared_prompt(context, shots=await self.shots())
                inference = await self._schematic_generator.generate(
                    prompt=prompt,
                    hints={
                        "reasoning_effort": reasoning_effort_for(context),
                        "cache": {"action": "store", "key": self._cache_key(context)},
                    },
                )
                return inference.info
            except Exception as exc:
                self._logger.warning(f"Guideline distiller prefill failed (continuing): {exc}")
                return None

    async def shots(self) -> Sequence[GuidelineDistillationShot]:
        return await shot_collection.list()

    def _format_shots(self, shots: Sequence[GuidelineDistillationShot]) -> str:
        return "\n".join(
            f"Example #{i}: ###\n{self._format_shot(shot)}" for i, shot in enumerate(shots, start=1)
        )

    def _format_shot(self, shot: GuidelineDistillationShot) -> str:
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
{_format_guideline(shot.guideline.condition, shot.guideline.action, shot.guideline.description)}

"""

        expected_result = shot.expected_result.model_dump(mode="json")
        if expected_result.get("distilled_action") is None:
            expected_result.pop("distilled_action", None)

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
        shots: Sequence[GuidelineDistillationShot],
    ) -> PromptBuilder:
        # The cross-turn/within-turn-stable shared prefix, then the per-guideline
        # tail: staged tool events and the specific guideline (with its tools). The
        # guideline is what differs across the fan-out, so everything before it stays
        # byte-identical within a turn — the prefix `prefill` warms.
        builder = self._build_shared_prompt(context, shots)

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
                    guideline.content.condition,
                    guideline.content.action,
                    guideline.content.description,
                    context.state.tools_by_guideline.get(guideline.id, set()),
                ),
            },
            status=SectionStatus.ACTIVE,
        )

        return builder

    def _build_shared_prompt(
        self,
        context: EngineContext,
        shots: Sequence[GuidelineDistillationShot],
    ) -> PromptBuilder:
        """The shared head of the distiller prompt (instructions, shots, identities,
        output format, and the per-turn context) — everything except the specific
        guideline. Stays byte-identical across the per-guideline fan-out, so it's the
        prefix `prefill` warms."""
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-distiller-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines". The agent makes use of these guidelines whenever it interacts with a user (also referred to as the customer).
Each guideline is composed of two parts:
- "condition": A natural-language condition that specifies when the guideline should apply.
          We examine the conversation in its current state and test this condition
          to determine whether the guideline should inform the next reply to the user.
- "action": A natural-language instruction that the agent should follow whenever the "condition"
          part of the guideline applies to the conversation in its particular state.
          Any instruction described here applies only to the agent, and not to the user.
""",
            props={},
        )
        builder.add_section(
            name="guideline-distiller-task-description",
            template="""
Task Description
----------------
Your task is twofold. First, evaluate whether the provided guideline applies to the most recent state of the interaction between yourself (an AI agent) and a user. Second, if it does apply, distill the guideline - its action together with any detailed instructions - down to exactly the guidance the agent needs for its very next response.

Determining applicability:
A guideline applies in either of these cases:
1. Its condition is relevant to the latest part of the conversation, and in particular to the most recent customer message; or
2. Its condition applied earlier and the agent is still in the middle of carrying out the action. Many actions span several steps, so a guideline remains applicable until its action has been fully carried out. For example, for the action "when the customer wants a drink, ask which drink and then which size", if the customer asked for a drink and the agent has only asked and received an answer for the first question, the guideline still applies - the agent has yet to ask about the size.

Evaluate the actual meaning of the condition, not just keyword matches, and take the full context into account - including context variables, glossary terms, capabilities, and tool results. Do not consider the guideline applicable based solely on earlier parts of the conversation if the topic has since shifted and its action is not still in progress, even if the previous topic remains unresolved. If the conversation moves from a broader issue to a related sub-issue, the guideline remains applicable as long as it is relevant to that sub-issue; once the discussion has clearly moved on to an entirely different topic, it no longer applies.
Record your applicability decision in the "is_relevant" field. "is_relevant" means this guideline contributes an actual instruction to the next response, so decide as follows:
- If the guideline's condition does not apply, set "is_relevant" to false.
- If the condition applies but there is genuinely nothing left to do right now - for example its action was already fully carried out earlier and has not arisen again for a new reason - also set "is_relevant" to false. There is no "relevant but nothing to do" state: if the guideline has nothing to contribute to the next response, it is not relevant.
- Only when the condition applies AND there is something concrete to do now, set "is_relevant" to true - and then you MUST provide a non-empty "distilled_action".
Before concluding that a step was already carried out, confirm it actually happened earlier in the conversation; if you are not sure it was completed, treat it as still needing doing. When "is_relevant" is false, omit "distilled_action" entirely.

Distilling the guideline:
If the guideline applies, work out which of its guidance is relevant right now. A guideline is often broad: its action and detailed instructions may lay out an ordered sequence of steps, a set of rules or constraints, plain information, or be phrased generally enough that it could be applied in more than one way - and may cover several different situations at once. Distill it down to only what bears on the current state of the conversation. Two shapes come up most:
1. An ordered sequence of steps (a multi-step process or checklist): only the next step that still needs doing is relevant - return just that step, not the steps that come after it.
2. A set of rules, constraints, or information that vary by situation: return every part that applies to the customer's current request, and leave out the parts that don't. Don't collapse several applicable rules down to just one of them.
When the action is more general, choose how best to apply it given the specific context of the conversation.

A rule only "applies to the customer's current request" when the customer is actually asking about, or attempting, the thing that rule governs - not merely because the rule's preconditions happen to hold.

Be especially careful with constraints, prohibitions, and limitations (rules of the form "X cannot be done" or "Y is not allowed"). Surface such a rule ONLY when the customer is actually trying to do the restricted thing, or has explicitly asked whether they can. If the customer's specific intent isn't known yet (e.g. they've only said they want to make "some change"), do NOT list or warn about restrictions that may or may not apply to it - just take the step that moves things forward (e.g. ask what change they want). Never append a restriction onto an otherwise-correct action "just in case" - that unsolicited warning derails the conversation. Constraints are applied reactively, when the customer actually hits them, not announced up front.

Internal verification instructions ("make sure the rules apply before doing X") tell YOU what to check before acting; they are not, by themselves, something to recite to the customer.

Some of the guidance may already have been carried out, in full or in part, earlier in the conversation. In that case, output only the part that still needs to be carried out now. If it was already fully carried out and its condition has not arisen again for a new reason, there is nothing left to do, so the guideline is not relevant (set "is_relevant" to false). If the condition has arisen again for a new reason (a new or subtly different context), the guidance should be applied again for that new occurrence. Be conservative about repeating guidance that delivers static, one-time information (e.g. "send our address"): only repeat it if the condition genuinely arose again.

If a flow has returned to an earlier stage (e.g. the customer corrects something they said earlier), don't just take the step that literally follows the changed point. Skip any later steps whose information you already have and that is still valid, and jump forward to the next step that genuinely still needs doing.

Be relevant-complete but selective: include everything that genuinely bears on the next response, and exclude everything that doesn't. Don't return a generic restatement of the action when the detailed instructions let you be specific; but equally, don't dump the whole guideline verbatim or bundle in steps and rules that aren't yet relevant, and never overwhelm the customer. When a single next step is all that applies, return just that step.

Make the result STANDALONE. What you output is the only thing the agent will have when it writes its next response - the original guideline, its action, and its detailed instructions will NOT be available to it. So spell out the actual substance: state the specific rule, value, fact, or step in full. Never point at content the agent can no longer see - don't say things like "follow the policy", "apply the rules above", "as described in the guideline", or "check the details". State the rule itself rather than referring to it: write out what specifically should be said or done, not "tell the customer the rule".

Carry through every concrete operative detail the guideline specifies for what you're surfacing - exact amounts, numbers, formulas, rates, thresholds, named tools, dates, and specific values. These are usually the whole point of the instruction, and the agent will have no way to recover them. When the guideline expresses a value as a formula (for example a fixed rate per unit) and the conversation supplies the inputs (for example the quantity), apply it and state the resulting figure, rather than collapsing it to a vague "an amount applies". Never paraphrase a specific value into a vague one, and never drop it. Record the result in the "distilled_action" field.

The exact format of your response will be provided later in this prompt.
""",
            props={},
        )
        builder.add_section(
            name="guideline-distiller-examples",
            template="""
Examples of Guideline Distillations:
-------------------
{formatted_shots}
""",
            props={
                "formatted_shots": self._format_shots(shots),
                "shots": shots,
            },
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)

        builder.add_section(
            name="guideline-distiller-output-format",
            template="""
OUTPUT FORMAT
-----------------
- Evaluate the guideline by filling in the details in the following structure:
```json
{result_structure_text}
```
""",
            props={
                "result_structure_text": self._format_output(),
            },
        )

        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        builder.add_capabilities_for_guideline_matching(context.state.capabilities)
        builder.add_interaction_history(context.interaction.events)

        return builder

    def _format_output(self) -> str:
        result: dict[str, JSONSerializable] = {
            "reasoning": (
                "<A brief explanation of whether the guideline currently applies and, "
                "if so, which of its guidance is relevant to the next agent response>"
            ),
            "is_relevant": (
                "<BOOL: true only if the guideline applies AND has a concrete instruction "
                "to contribute to the next response; false otherwise (including when it "
                "applies but its action was already fully carried out)>"
            ),
            "distilled_action": (
                "<REQUIRED whenever is_relevant=True; omit ONLY when is_relevant=False. "
                "A standalone, self-contained statement of the guidance relevant to the "
                "next response, distilled from the guideline's action and detailed "
                "instructions: the next step for a sequential action, or every applicable "
                "rule/piece of information for a policy. Spell out the actual substance - "
                "never refer to 'the policy', 'the rules', or anything the agent can no "
                "longer see, and carry through every concrete operative detail (exact "
                "amounts, numbers, formulas, rates, named tools, dates), applying any "
                "formula to inputs the conversation supplies>"
            ),
        }

        return json.dumps(result, indent=4)


def _readable_tool_spec(tool_id: ToolId, tool: Tool) -> dict[str, JSONSerializable]:
    # ``get_tool_spec`` renders each parameter as a JSON *string*; left as-is, the
    # surrounding ``json.dumps`` would re-escape it into an unreadable nested string.
    # Parse the parameter specs back into objects so they render as clean nested JSON.
    spec = get_tool_spec(tool_id, tool)
    for key in ("optional_arguments", "required_parameters"):
        params = spec.get(key)
        if isinstance(params, dict):
            spec[key] = {name: json.loads(value) for name, value in params.items()}
    return spec


def _format_guideline(
    condition: str,
    action: Optional[str],
    description: Optional[str],
    tools: Set[tuple[ToolId, Tool]] = set(),
) -> str:
    # The action is optional and only present to contextualize the condition; omit it
    # entirely when absent rather than rendering "Action: None".
    text = f"Condition: {condition}."
    if action:
        text += f" Action: {action}"
    if description:
        text += f" Details: {description}"
    if tools:
        # Surface the tools attached to the action (description + arguments), so the
        # distiller knows what each tool does and can name it as the next step.
        tools_text = json.dumps(
            [_readable_tool_spec(tool_id, tool) for tool_id, tool in tools], indent=2
        )
        text += (
            "\nThe action may be carried out (in full or in part) using the following tools. "
            "When the next step is to run one of these tools, the distilled action should "
            f"say so explicitly:\n{tools_text}"
        )
    return text


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


# Shot 1: a journey expressed as a single guideline. The action describes the entire
# multi-step journey, and the distiller must collapse it into just the next step given
# that the journey is already in progress.
example_1_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I'd like to book a flight please."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "I'd be happy to help you book a flight! Could you please tell me your source and destination airports?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "I want to fly from JFK in New York to LAX in Los Angeles.",
    ),
]

example_1_guideline = GuidelineContent(
    condition="the customer wants to book a flight",
    action=(
        "Ask for the source and destination airports, then for the dates of the departure "
        "and return flight, then whether they want economy or business class, then for the "
        "name of the traveler, and finally book the flight using the book_flight tool."
    ),
)

example_1_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer wants to book a flight and has already provided the source and "
        "destination airports, so the journey is in progress. The next step in the action "
        "is to ask for the departure and return dates."
    ),
    is_relevant=True,
    distilled_action="Ask for the dates of the departure and return flight.",
)


# Shot 2: the guideline's condition does not apply to the current state of the
# conversation at all.
example_2_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi, I'm planning a trip to Italy next month. What can I do there?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! Do you prefer exploring cities or enjoying scenic landscapes?",
    ),
    _make_event(
        "34",
        EventSource.CUSTOMER,
        "Actually I'm also wondering — do I need any special visas or documents as an American citizen?",
    ),
]

example_2_guideline = GuidelineContent(
    condition="The customer is looking for flight or accommodation booking assistance",
    action="Provide links or suggestions for flight aggregators and hotel booking platforms.",
)

example_2_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is asking about visas and travel documents, not about booking flights "
        "or accommodation, so the condition does not apply to the current state of the "
        "conversation."
    ),
    is_relevant=False,
)


# Shot 3: the guideline applies and its action is a single, concrete instruction that
# should be taken as is.
example_3_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "Hi there, what is the S&P 500 trading at right now?",
    ),
]

example_3_guideline = GuidelineContent(
    condition="the customer asks about the value of a stock",
    action="provide the price using the 'check_stock_price' tool",
)

example_3_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is asking about the value of the S&P 500, so the guideline applies. "
        "Its action is a single concrete instruction, so it should be taken as is."
    ),
    is_relevant=True,
    distilled_action="Provide the price of the S&P 500 using the 'check_stock_price' tool.",
)


# Shot 4: the guideline previously applied and its (static) action was already taken;
# there is no new reason to retake it. Nothing remains to be done, so the guideline is
# NOT relevant (rather than relevant with an empty action).
example_4_events = [
    _make_event("11", EventSource.CUSTOMER, "Hi, I need help changing the email on my account."),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "Sure! Could you please provide your account ID so I can verify your identity?",
    ),
    _make_event("34", EventSource.CUSTOMER, "It's ACC12345."),
    _make_event("56", EventSource.AI_AGENT, "Thanks! I've updated your email."),
    _make_event(
        "88",
        EventSource.CUSTOMER,
        "Also, can you check the last payment on my account?",
    ),
]

example_4_guideline = GuidelineContent(
    condition="The customer is asking for account-related help",
    action="Ask for their account ID to verify their identity",
)

example_4_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer is still asking for account-related help, but they already provided "
        "their account ID earlier and it remains valid for this request, so the action has "
        "already been carried out and there is no new reason to ask for it again. Nothing "
        "remains to be done, so the guideline is not relevant."
    ),
    is_relevant=False,
)


# Shot 5: the guideline previously applied and is now triggered again for a new reason;
# the action should be retaken in a way that is specific to the current context.
example_5_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I'm planning a trip next month. Any ideas on where to go?",
    ),
    _make_event(
        "23",
        EventSource.AI_AGENT,
        "That sounds exciting! What kind of activities do you enjoy — relaxing on the beach, hiking, museums, food tours?",
    ),
    _make_event("34", EventSource.CUSTOMER, "I love hiking and exploring local food scenes."),
    _make_event(
        "56",
        EventSource.AI_AGENT,
        "Great! You might enjoy a trip to the Pacific Northwest — plenty of trails and great food in Portland and Seattle.",
    ),
    _make_event("88", EventSource.CUSTOMER, "What about a winter trip in Europe?"),
]

example_5_guideline = GuidelineContent(
    condition="The customer wants recommendations for a trip",
    action="Ask for their preferred activities and recommend accordingly",
)

example_5_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer raised a new trip — a winter trip to Europe — so the condition arose "
        "again for a new reason and the action should be reapplied. Their preferred "
        "activities for this new trip aren't known yet, so the next step is to ask about "
        "them for the Europe trip specifically."
    ),
    is_relevant=True,
    distilled_action="Ask the customer what activities they'd enjoy on their winter trip to Europe.",
)


# Shot 6: the flow has returned to an earlier step (the customer changed a detail they
# gave earlier). The later detail they already provided is still valid, so the next step
# is not the one that literally follows the changed step - it's the next step that still
# needs doing.
example_6_events = [
    _make_event("11", EventSource.CUSTOMER, "I'd like to book a home cleaning."),
    _make_event("21", EventSource.AI_AGENT, "Sure! What date would you like?"),
    _make_event("31", EventSource.CUSTOMER, "Next Tuesday."),
    _make_event("41", EventSource.AI_AGENT, "Got it. What's the address?"),
    _make_event("51", EventSource.CUSTOMER, "42 Oak Street."),
    _make_event("61", EventSource.AI_AGENT, "And how many rooms need cleaning?"),
    _make_event(
        "71",
        EventSource.CUSTOMER,
        "Actually, can we make it next Wednesday instead of Tuesday?",
    ),
]

example_6_guideline = GuidelineContent(
    condition="the customer wants to book a home cleaning",
    action=(
        "Ask for the desired date, then for the home address, then for the number of rooms, "
        "and finally confirm the details and book the cleaning."
    ),
)

example_6_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer changed the date, returning to an earlier step of the action. The home "
        "address they already gave is still valid, so there's no need to ask for it again. The "
        "next step that still needs doing is asking how many rooms need cleaning."
    ),
    is_relevant=True,
    distilled_action="Ask the customer how many rooms need cleaning.",
)


# Shot 7: a policy expressed as a vague action plus detailed rules. The action only
# points at the policy; the substance lives in the details. This is not a sequence of
# steps, so the distiller must surface every rule that bears on the customer's request -
# here two of the three - rather than picking a single next step.
example_7_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I'd like to downgrade to the Basic plan. And if I change my mind, can I switch back later this month?",
    ),
]

example_7_guideline = GuidelineContent(
    condition="the customer wants to change their subscription plan",
    action="follow the plan change policy",
    description=(
        "Changing a subscription plan:\n"
        "- Upgrades take effect immediately.\n"
        "- Downgrades take effect at the end of the current billing cycle.\n"
        "- A plan can be changed at most once per billing cycle."
    ),
)

example_7_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer wants to downgrade and asks whether they can switch again later this "
        "month, so the guideline applies. The downgrade-timing rule and the once-per-cycle "
        "rule both bear on this request; the upgrade rule does not. Both relevant rules "
        "should be surfaced, not just one."
    ),
    is_relevant=True,
    distilled_action=(
        "Let the customer know the downgrade takes effect at the end of the current billing "
        "cycle, and that a plan can only be changed once per billing cycle (so they could not "
        "switch again this month)."
    ),
)


# Shot 8: the relevant rule carries a concrete operative detail - a per-guest rate. The
# distilled action must carry the rate through AND apply it to the guest count from the
# conversation, not abstract it into "a deposit is required".
example_8_events = [
    _make_event(
        "11",
        EventSource.CUSTOMER,
        "I'd like to book catering for 8 guests for next Friday's event.",
    ),
]

example_8_guideline = GuidelineContent(
    condition="the customer wants to book catering",
    action="follow the catering booking policy",
    description=(
        "Booking catering:\n"
        "- A refundable deposit of $25 per guest is required to confirm the booking.\n"
        "- Cancellations within 48 hours of the event forfeit the deposit."
    ),
)

example_8_expected = GuidelineDistillSchema(
    reasoning=(
        "The customer wants to book catering for 8 guests, so the deposit rule applies. The "
        "deposit is $25 per guest, which for 8 guests is $200 - that amount must be stated, "
        "not abstracted away. The cancellation rule isn't raised yet, so it's left out."
    ),
    is_relevant=True,
    distilled_action=(
        "Tell the customer a refundable deposit of $25 per guest is required to confirm the "
        "booking - $200 for the 8 guests."
    ),
)


_baseline_shots: Sequence[GuidelineDistillationShot] = [
    GuidelineDistillationShot(
        description="",
        interaction_events=example_1_events,
        guideline=example_1_guideline,
        expected_result=example_1_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_2_events,
        guideline=example_2_guideline,
        expected_result=example_2_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_3_events,
        guideline=example_3_guideline,
        expected_result=example_3_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_4_events,
        guideline=example_4_guideline,
        expected_result=example_4_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_5_events,
        guideline=example_5_guideline,
        expected_result=example_5_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_6_events,
        guideline=example_6_guideline,
        expected_result=example_6_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_7_events,
        guideline=example_7_guideline,
        expected_result=example_7_expected,
    ),
    GuidelineDistillationShot(
        description="",
        interaction_events=example_8_events,
        guideline=example_8_guideline,
        expected_result=example_8_expected,
    ),
]

shot_collection = ShotCollection[GuidelineDistillationShot](_baseline_shots)
