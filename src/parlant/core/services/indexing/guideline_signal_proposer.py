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

import json
import traceback
from dataclasses import dataclass
from typing import Optional, Sequence

from parlant.core.agents import Agent
from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.guideline_matching.generic.common import escape_json_string
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import Term
from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.indexing.common import EvaluationError, ProgressReport
from parlant.core.shots import Shot, ShotCollection


class GuidelineSignalProposition(DefaultBaseModel):
    signals: Sequence[str]
    rationale: str


class GuidelineSignalPropositionSchema(DefaultBaseModel):
    rationale: str
    signals: list[str]


@dataclass
class GuidelineSignalProposerShot(Shot):
    title: str
    guideline: GuidelineContent
    expected_result: GuidelineSignalPropositionSchema


class GuidelineSignalProposer:
    def __init__(
        self,
        logger: Logger,
        optimization_policy: OptimizationPolicy,
        schematic_generator: SchematicGenerator[GuidelineSignalPropositionSchema],
    ) -> None:
        self._logger = logger
        self._optimization_policy = optimization_policy
        self._schematic_generator = schematic_generator

    async def propose_signals(
        self,
        guideline: GuidelineContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        progress_report: Optional[ProgressReport] = None,
    ) -> GuidelineSignalProposition:
        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("GuidelineSignalProposer"):
            generation_attempt_temperatures = (
                self._optimization_policy.get_guideline_proposition_retry_temperatures(
                    hints={"type": self.__class__.__name__}
                )
            )

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    proposition = await self._generate_signals(
                        guideline,
                        title,
                        agent,
                        glossary_terms,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    )
                    signals = self._normalize_signals(proposition.signals)

                    if len(signals) != 5:
                        raise ValueError(
                            f"Expected exactly 5 guideline signals, but got {len(signals)}"
                        )

                    if progress_report:
                        await progress_report.increment(1)

                    return GuidelineSignalProposition(
                        signals=signals,
                        rationale=proposition.rationale,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"GuidelineSignalProposer attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )

                    last_generation_exception = exc

            raise EvaluationError() from last_generation_exception

    def _normalize_signals(self, signals: Sequence[str]) -> Sequence[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for signal in signals:
            if not (clean := signal.strip()):
                continue

            key = clean.casefold()
            if key in seen:
                continue

            normalized.append(clean)
            seen.add(key)

        return normalized

    async def _build_prompt(
        self,
        guideline: GuidelineContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        shots: Sequence[GuidelineSignalProposerShot],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-signal-proposer-general-instructions",
            template="""
GENERAL INSTRUCTIONS
-----------------
In our system, the behavior of a conversational AI agent is guided by "guidelines".
Each guideline has a condition describing when it should apply, and may also have an action and description.

The Compass engine uses guideline signals for semantic recall. A signal is a short example user message that should activate the guideline.
Signals are embedded separately from the guideline text and compared with the latest user message.
""",
        )

        builder.add_section(
            name="guideline-signal-proposer-task-description",
            template="""
TASK DESCRIPTION
-----------------
Your task is to suggest user messages that should activate the given guideline.

Generate exactly 5 signals.
Each signal must be phrased as something a user/customer might actually say in a conversation.
The 5 signals should be distinct and unique from each other while still being clearly relevant to the guideline.
Signals should cover common wording, paraphrases, and important edge cases implied by the guideline.
Prefer concrete natural messages over keywords.
Do not include messages that would activate a different guideline more specifically.
Do not mention that these are signals, embeddings, or guidelines inside the signal text.
""",
        )

        builder.add_section(
            name="guideline-signal-proposer-agent",
            template="""
AGENT
-----------
{agent_text}
""",
            props={"agent_text": self._format_agent(agent)},
        )

        builder.add_section(
            name="guideline-signal-proposer-glossary",
            template="""
GLOSSARY
-----------
{glossary_text}
""",
            props={"glossary_text": self._format_glossary(glossary_terms)},
        )

        builder.add_section(
            name="guideline-signal-proposer-shots",
            template="""
EXAMPLES
-----------
{shots_text}""",
            props={"shots_text": self._format_shots(shots)},
        )

        builder.add_section(
            name="guideline-signal-proposer-guideline",
            template="""
GUIDELINE
-----------
{guideline_text}
""",
            props={"guideline_text": self._format_guideline(title, guideline)},
        )

        builder.add_section(
            name="guideline-signal-proposer-output-format",
            template="""OUTPUT FORMAT
-----------
Use the following format:
Expected output (JSON):
```json
{{
  "rationale": "<str, short explanation of the activation surface you covered>",
  "signals": [
    "<str, exactly 5 different example user messages that should activate this guideline>"
  ]
}}
```
""",
        )

        return builder

    async def _generate_signals(
        self,
        guideline: GuidelineContent,
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        temperature: float,
    ) -> GuidelineSignalPropositionSchema:
        prompt = await self._build_prompt(
            guideline,
            title,
            agent,
            glossary_terms,
            await shot_collection.list(),
        )

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        self._logger.debug(
            f"GuidelineSignalProposer response: {response.content.model_dump_json(indent=2)}"
        )

        return response.content

    def _format_agent(self, agent: Agent | None) -> str:
        if agent is None:
            return "No agent context was provided."

        return f"{agent.name}\n\n{agent.description or ''}".strip()

    def _format_glossary(self, terms: Sequence[Term]) -> str:
        if not terms:
            return "No glossary terms were provided."

        def format_term(term: Term) -> str:
            synonyms = f"\nSynonyms: {', '.join(term.synonyms)}" if term.synonyms else ""
            return f"## {term.name}\n{term.description}{synonyms}"

        return "\n\n".join(format_term(term) for term in terms)

    def _format_guideline(self, title: str, guideline: GuidelineContent) -> str:
        result = f"# {escape_json_string(title)}"

        if guideline.condition and guideline.action:
            result += (
                f"\n\n## When {escape_json_string(guideline.condition)} "
                f"then {escape_json_string(guideline.action)}"
            )
        elif guideline.condition:
            result += f"\n\n## Condition: {escape_json_string(guideline.condition)}"
        elif guideline.action:
            result += f"\n\n## Action: {escape_json_string(guideline.action)}"

        if guideline.description:
            result += f"\n\n{escape_json_string(guideline.description)}"

        return result

    def _format_shots(self, shots: Sequence[GuidelineSignalProposerShot]) -> str:
        return "\n".join(
            [
                f"""Example {i}: {shot.description}
Guideline:
{self._format_guideline(shot.title, shot.guideline)}

Expected Response:
{json.dumps(shot.expected_result.model_dump(mode="json", exclude_unset=True), indent=2)}
###
"""
                for i, shot in enumerate(shots, start=1)
            ]
        )


example_1_guideline = GuidelineContent(
    condition="The customer wants to report a lost or stolen card",
    action="Help them secure the card and explain replacement options",
)
example_1_shot = GuidelineSignalProposerShot(
    description="Card-loss guideline with several natural phrasings",
    title="Lost or Stolen Card",
    guideline=example_1_guideline,
    expected_result=GuidelineSignalPropositionSchema(
        rationale="The signals cover lost cards, stolen cards, and urgent card-security language.",
        signals=[
            "I lost my debit card",
            "My card was stolen and I need help",
            "I can't find my credit card, can you block it?",
            "Someone took my card",
            "I need to report a missing card",
        ],
    ),
)

example_2_guideline = GuidelineContent(
    condition="The customer asks about refund eligibility for a delayed flight",
    action="Review the reservation details and explain eligible refund or compensation options",
)
example_2_shot = GuidelineSignalProposerShot(
    description="Travel compensation guideline where the user may not say refund directly",
    title="Delayed Flight Refund Eligibility",
    guideline=example_2_guideline,
    expected_result=GuidelineSignalPropositionSchema(
        rationale="The signals cover direct refund requests, compensation wording, and frustration about delay impact.",
        signals=[
            "My flight was delayed, can I get a refund?",
            "Do I qualify for compensation because my flight was late?",
            "The delay ruined my plans, what can you do for me?",
            "Am I entitled to anything for a delayed flight?",
            "My plane arrived late and I want to know my options",
        ],
    ),
)

example_3_guideline = GuidelineContent(
    condition="The customer needs to update the shipping address on an existing order",
    action="Collect the order identifier and new address before updating the order",
)
example_3_shot = GuidelineSignalProposerShot(
    description="Order-change guideline with address-update phrasing",
    title="Update Shipping Address",
    guideline=example_3_guideline,
    expected_result=GuidelineSignalPropositionSchema(
        rationale="The signals cover changing, correcting, and redirecting delivery addresses.",
        signals=[
            "I need to change the delivery address for my order",
            "Can you ship my package somewhere else?",
            "I entered the wrong address at checkout",
            "Please update where my order is being sent",
            "My order is going to the old address",
        ],
    ),
)

_baseline_shots: Sequence[GuidelineSignalProposerShot] = [
    example_1_shot,
    example_2_shot,
    example_3_shot,
]

shot_collection = ShotCollection[GuidelineSignalProposerShot](_baseline_shots)
