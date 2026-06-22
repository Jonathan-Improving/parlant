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

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum, IntEnum, auto
from io import StringIO
from itertools import chain
import traceback
from typing import cast

from parlant.core.agents import Effort
from parlant.core.async_utils import safe_gather
from parlant.core.common import Criticality, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.guideline_matcher_registry import GuidelineMatcherRegistry
from parlant.core.engines.compass.guideline_matching.guideline_function_matcher import (
    GuidelineFunctionMatcher,
)
from parlant.core.engines.compass.guideline_matching.guideline_distiller import GuidelineDistiller
from parlant.core.engines.compass.guideline_matching.guideline_ranker import GuidelineRanker
from parlant.core.engines.compass.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.engines.compass.variable_loader import VariableLoader
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.relationships import (
    RelationshipEntityKind,
    RelationshipKind,
    RelationshipStore,
)
from parlant.core.sessions import ToolEventData
from parlant.core.tags import TagId
from parlant.core.tools import Tool, ToolId, ToolRelevanceResult

_GUIDELINE_IS_COMPLEX: dict[GuidelineId, bool] = {}
_SESSION_GUIDELINE_IDS_METADATA_KEY = "compass.session_guideline_ids"


class _ContextUsage(Enum):
    """How to use a guideline within the engine's context"""

    INCLUDE_IN_SESSION = auto()
    MATCH_CURRENT_TURN = auto()


class _MatcherStrategy(IntEnum):
    """How much effort to spend deciding whether a guideline applies, cheapest to
    most thorough."""

    NONE = auto()
    RECALL = auto()
    RANK = auto()
    DISTILL = auto()


class Matcher:
    """Prepares the turn's response state: which guidelines apply and which tools
    are offered to the model.

    ``fill`` does the initial preparation; ``update`` refreshes it after a step
    (reevaluating guidelines gated on the tools that just ran). Both leave
    ``context.state`` carrying the matched guidelines, the matched guidelines'
    tools, and the offered tool catalog.
    """

    _MAX_AVAILABLE_TOOLS = 16
    _MAX_GLOSSARY_TERMS = 30

    def __init__(
        self,
        logger: Logger,
        guideline_recaller: GuidelineRecaller,
        guideline_ranker: GuidelineRanker,
        guideline_distiller: GuidelineDistiller,
        guideline_function_matcher: GuidelineFunctionMatcher,
        matcher_registry: GuidelineMatcherRegistry,
        relationship_store: RelationshipStore,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
        variable_loader: VariableLoader,
    ) -> None:
        self._logger = logger
        self._guideline_recaller = guideline_recaller
        self._guideline_ranker = guideline_ranker
        self._guideline_distiller = guideline_distiller
        self._guideline_function_matcher = guideline_function_matcher
        self._matcher_registry = matcher_registry
        self._relationship_store = relationship_store
        self._entity_queries = entity_queries
        self._entity_commands = entity_commands
        self._variable_loader = variable_loader

    async def preload(self, context: EngineContext) -> None:
        # Load the shared prompt/matching inputs that matcher-owned preparation
        # controls before matching and cache warm-up.
        context.state.context_variables, guidelines = await safe_gather(
            self._variable_loader.load(context),
            self._entity_queries.find_guidelines_for_context(context.agent.id, []),
        )

        context.state.usable_guidelines = list(guidelines)

        await self._load_session_guidelines(context)

    async def fill(self, context: EngineContext) -> None:
        """Initial preparation: match all usable guidelines, rank the agent's tool
        pool, and load the relevant glossary (all independent, so in parallel), then
        select the offered tools."""
        await self._load_tools_by_guideline(context)
        await safe_gather(
            self._match(context),
            self._rank_tool_pool(context),
            self._load_glossary(context),
        )
        await self._select_tools(context)

    async def update(self, context: EngineContext) -> None:
        """Refresh after a step: reevaluate guidelines gated on the tools that
        just ran, then re-select tools. The tool pool ranking depends on the
        (unchanged) conversation, so it isn't re-ranked."""
        await self._reevaluate(context)
        await self._select_tools(context)

    async def warm_up(self, context: EngineContext) -> None:
        """Warm only the matcher components that the current strategy can use."""
        should_prefill_ranker, should_prefill_distiller = await self._get_prefill_targets(context)

        if not should_prefill_ranker and not should_prefill_distiller:
            return

        # The glossary is part of the cached shared prefix, so it must be loaded before
        # warming — otherwise the warmed prefix omits it and the first real turn (which
        # has loaded it) misses. At end-of-turn it's already loaded by `fill`, so skip.
        if not context.state.glossary_terms:
            await self._load_glossary(context)

        prefill_tasks = []

        if should_prefill_ranker:
            prefill_tasks.append(self._guideline_ranker.warm_up(context))

        if should_prefill_distiller:
            prefill_tasks.append(self._guideline_distiller.warm_up(context))

        await safe_gather(*prefill_tasks)

    async def _get_prefill_targets(self, context: EngineContext) -> tuple[bool, bool]:
        guidelines = [
            g
            for g in context.state.usable_guidelines
            if g.criticality != Criticality.LOW and self._matcher_registry.get(g.id) is None
        ]

        if not guidelines:
            return False, False

        await self._load_strategy_choice_signals(context, guidelines)

        prefill_ranker = False
        prefill_distiller = False

        for guideline in guidelines:
            match self._get_strategy(context, guideline):
                case _MatcherStrategy.RANK:
                    prefill_ranker = True
                case _MatcherStrategy.DISTILL:
                    prefill_distiller = True

        return prefill_ranker, prefill_distiller

    # --- guideline matching ---

    async def _load_session_guidelines(self, context: EngineContext) -> None:
        guideline_ids = self._read_session_guideline_ids(context.session.metadata)

        guidelines_by_id = {g.id: g for g in context.state.usable_guidelines}

        context.state.session_guidelines = {
            guidelines_by_id[guideline_id]
            for guideline_id in guideline_ids
            if guideline_id in guidelines_by_id
        }

    async def _store_session_guidelines(
        self,
        context: EngineContext,
        matches: Sequence[tuple[GuidelineMatch, _ContextUsage]],
    ) -> None:
        guidelines_by_id = {g.id: g for g in context.state.usable_guidelines}
        guideline_ids = {m[0].guideline.id for m in matches}

        context.state.session_guidelines = {guidelines_by_id[gid] for gid in guideline_ids}

        current_guideline_ids = self._read_session_guideline_ids(context.session.metadata)

        if guideline_ids == current_guideline_ids:
            return

        metadata = dict(context.session.metadata)

        metadata[_SESSION_GUIDELINE_IDS_METADATA_KEY] = [
            str(guideline_id) for guideline_id in guideline_ids
        ]

        await self._entity_commands.update_session(context.session.id, {"metadata": metadata})
        context.session = replace(context.session, metadata=metadata)

    def _read_session_guideline_ids(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> set[GuidelineId]:
        last_known_state = set(
            cast(Iterable[str], metadata.get(_SESSION_GUIDELINE_IDS_METADATA_KEY, []))
        )

        return {GuidelineId(guideline_id) for guideline_id in last_known_state}

    async def _match(self, context: EngineContext) -> None:
        matches = await self._run_batches(context, context.state.usable_guidelines)
        await self._record(context, matches, append=False)

    async def _reevaluate(self, context: EngineContext) -> None:
        executed_tool_ids = self._executed_tool_ids(context)
        if not executed_tool_ids:
            return

        gated = await self._find_guidelines_gated_on_tools(context, executed_tool_ids)
        already_matched = self._matched_guideline_ids(context)
        candidates = [g for g in gated if g.id not in already_matched]
        if not candidates:
            return

        matches = await self._run_batches(context, candidates)

        if not matches:
            return

        await self._record(context, matches, append=True)

    def _get_strategy(self, context: EngineContext, guideline: Guideline) -> _MatcherStrategy:
        def needs_distillation(g: Guideline) -> bool:
            if (is_complex := _GUIDELINE_IS_COMPLEX.get(g.id)) is not None:
                return is_complex

            condition_len = len(g.content.condition) if g.content.condition else 0
            action_len = len(g.content.action) if g.content.action else 0
            description_len = len(g.content.description) if g.content.description else 0

            result = (condition_len + action_len + description_len) >= 300
            _GUIDELINE_IS_COMPLEX[g.id] = result
            return result

        strategy = _MatcherStrategy.RECALL

        match context.state.dynamic_effort_level:
            case Effort.MIN:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.HIGH:
                        strategy = _MatcherStrategy.RECALL
            case Effort.LOW:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.HIGH:
                        strategy = _MatcherStrategy.RECALL
            case Effort.MEDIUM:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.HIGH:
                        strategy = (
                            _MatcherStrategy.DISTILL
                            if needs_distillation(guideline)
                            else _MatcherStrategy.RANK
                        )
            case Effort.HIGH:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = (
                            _MatcherStrategy.DISTILL
                            if needs_distillation(guideline)
                            else _MatcherStrategy.RANK
                        )
                    case Criticality.HIGH:
                        strategy = (
                            _MatcherStrategy.DISTILL
                            if needs_distillation(guideline)
                            else _MatcherStrategy.RANK
                        )
            case Effort.MAX:
                match guideline.criticality:
                    case Criticality.LOW:
                        strategy = _MatcherStrategy.RECALL
                    case Criticality.MEDIUM:
                        strategy = _MatcherStrategy.RANK
                    case Criticality.HIGH:
                        strategy = _MatcherStrategy.RANK

        if strategy < _MatcherStrategy.RANK:
            # There are some special conditions under which we want
            # to ensure a baseline of matching effort, since their
            # matching carries important implications.

            if guideline.labels:
                # If the guideline has labels, we want to ensure high-quality analytics,
                # so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_has_dependencies(context, guideline):
                # If the guideline has dependencies, it may be gating important follow-up
                # guidelines, so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

            if self._check_if_has_tools(context, guideline):
                # If the guideline has tools, it may be gating important interactions
                # with data and/or actions, so we should at least rank - not use embeddings.
                return _MatcherStrategy.RANK

        return strategy

    async def _load_strategy_choice_signals(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> None:
        associations = await self._entity_queries.find_guideline_tool_associations()
        context.state.guideline_ids_with_tools = {a.guideline_id for a in associations}

        # A guideline "has dependencies" if it takes part in a dependency
        # relationship — either as the dependent (source) or as the guideline being
        # depended on (target, which gates its dependents). Either side means
        # getting its match right carries downstream consequences.
        #
        # Endpoints may be guidelines directly, or tags (TAG_ALL/TAG_ANY) standing
        # in for every guideline that carries the tag; we resolve both against the
        # candidate guidelines so the per-guideline check can stay a plain lookup.
        dependency_relationships = chain(
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY, indirect=False
            ),
            await self._relationship_store.list_relationships(
                kind=RelationshipKind.DEPENDENCY_ANY, indirect=False
            ),
        )

        dependency_guideline_ids: set[GuidelineId] = set()
        dependency_tags: set[TagId] = set()
        for relationship in dependency_relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint.kind == RelationshipEntityKind.GUIDELINE:
                    dependency_guideline_ids.add(cast(GuidelineId, endpoint.id))
                elif endpoint.kind.is_tag:
                    dependency_tags.add(cast(TagId, endpoint.id))

        context.state.guideline_ids_with_dependencies = {
            guideline.id
            for guideline in guidelines
            if guideline.id in dependency_guideline_ids
            or not dependency_tags.isdisjoint(guideline.tags)
        }

    def _check_if_has_tools(self, context: EngineContext, guideline: Guideline) -> bool:
        return guideline.id in context.state.guideline_ids_with_tools

    def _check_if_has_dependencies(self, context: EngineContext, guideline: Guideline) -> bool:
        return guideline.id in context.state.guideline_ids_with_dependencies

    async def _run_batches(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> Sequence[tuple[GuidelineMatch, _ContextUsage]]:
        """Decide which of `guidelines` apply by routing each to a strategy and
        running the resulting batches in parallel.

        A guideline with a code (Python) matcher always goes to the function
        matcher, regardless of strategy — an explicit matcher is authoritative. The
        rest are bucketed by `_get_strategy`.
        """
        # Low-criticality guidelines are always included in the system instructions
        guidelines = [g for g in guidelines if g.criticality != Criticality.LOW]

        if not guidelines:
            return []

        # _get_strategy runs per guideline and is synchronous, so precompute the
        # store-backed signals it consults (which guidelines have tools / are in a
        # dependency relationship) once, here.
        await self._load_strategy_choice_signals(context, guidelines)

        code_batch: list[Guideline] = []
        recall_batch: list[Guideline] = []
        rank_batch: list[Guideline] = []
        distill_batch: list[Guideline] = []

        for guideline in guidelines:
            if self._matcher_registry.get(guideline.id) is not None:
                code_batch.append(guideline)
                continue

            match self._get_strategy(context, guideline):
                case _MatcherStrategy.RECALL:
                    recall_batch.append(guideline)
                case _MatcherStrategy.RANK:
                    rank_batch.append(guideline)
                case _MatcherStrategy.DISTILL:
                    distill_batch.append(guideline)

        code_matches, recalled, ranked, distilled = await safe_gather(
            self._guideline_function_matcher.match(context, code_batch),
            self._guideline_recaller.recall(context, recall_batch),
            self._guideline_ranker.rank(context, rank_batch),
            self._guideline_distiller.distill(context, distill_batch),
        )

        ranking_results = StringIO()

        if ranked.generation_info:
            ranking_results.write(f"Usage: {ranked.generation_info}\n\n")

        if ranked.ranked_guidelines:
            for idx, rank_result in enumerate(ranked.ranked_guidelines, start=1):
                g = rank_result.guideline

                ranking_results.write(
                    f"### {idx} [Score: {rank_result.score:.2f} ({'Relevant' if rank_result.is_relevant else 'Not Relevant'})]\n\n"
                )
                if g.content.condition:
                    ranking_results.write(f"    Condition: {g.content.condition}\n")
                if g.content.action:
                    ranking_results.write(f"    Action: {g.content.action}\n")
                ranking_results.write(f"    Reasoning: {rank_result.reasoning}\n\n")

            self._logger.debug(
                f"{self.__class__.__name__} guideline ranking results:\n{ranking_results.getvalue()}"
            )

        distillation_results = StringIO()

        if distilled.generation_info:
            distillation_results.write(f"Usage: {distilled.generation_info}\n\n")

        if distilled.distilled_guidelines:
            for idx, distill_result in enumerate(distilled.distilled_guidelines, start=1):
                g = distill_result.guideline

                distillation_results.write(
                    f"### {idx} [{'Relevant' if distill_result.is_relevant else 'Not Relevant'}]\n\n"
                )
                if g.content.condition:
                    distillation_results.write(f"    Condition: {g.content.condition}\n")
                if g.content.action:
                    distillation_results.write(f"    Action: {g.content.action}\n")
                if g.content.description:
                    distillation_results.write(
                        f"    Description: {g.content.description.strip()}\n"
                    )
                if distill_result.distilled_action:
                    distillation_results.write(
                        f"    Distilled Action: {distill_result.distilled_action.strip()}\n"
                    )
                distillation_results.write(f"    Reasoning: {distill_result.reasoning.strip()}\n\n")

            self._logger.debug(
                f"{self.__class__.__name__} guideline distillation results:\n{distillation_results.getvalue()}"
            )

        matches: list[tuple[GuidelineMatch, _ContextUsage]] = [
            (m, _ContextUsage.MATCH_CURRENT_TURN) for m in code_matches
        ]

        matches += [
            (
                GuidelineMatch(
                    guideline=rc.guideline,
                    rationale="This may or may not be relevant right now - use your judgment.",
                ),
                _ContextUsage.INCLUDE_IN_SESSION,
            )
            for rc in recalled.recalled_guidelines
            if rc.is_relevant
        ]

        matches += [
            (
                GuidelineMatch(
                    guideline=rk.guideline,
                    rationale=rk.reasoning
                    or "This may or may not be relevant right now - use your judgment.",
                ),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
            for rk in ranked.ranked_guidelines
            if rk.is_relevant
        ]

        matches += [
            (
                GuidelineMatch(
                    guideline=dg.guideline,
                    rationale=dg.reasoning,
                    metadata={
                        "distilled_action": self._format_distilled_policy_note(
                            dg.guideline, dg.distilled_action
                        )
                    }
                    if dg.distilled_action
                    else {},
                ),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
            for dg in distilled.distilled_guidelines
            if dg.is_relevant
        ]

        return matches

    def _format_distilled_policy_note(self, guideline: Guideline, distilled_action: str) -> str:
        policy_title = guideline.title or "Untitled policy"
        return (
            f'According to policy "{policy_title}", {distilled_action.strip()}\n'
            "Apply this only insofar as it remains compatible with the other active policies "
            "and system instructions."
        )

    async def _record(
        self,
        context: EngineContext,
        matches: Sequence[tuple[GuidelineMatch, _ContextUsage]],
        *,
        append: bool,
    ) -> None:
        # Classify into ordinary vs tool-enabled (the latter carries the tool ids
        # the matched guidelines enable).
        tool_enabled = await self._find_tool_enabled_guideline_matches(matches)
        ordinary = [m for m in matches if m not in tool_enabled]

        if append:
            # Never reorder, so the already-rendered guidelines stay a
            # byte-identical prefix.
            context.state.ordinary_guideline_matches.extend([key[0] for key in ordinary])
            context.state.tool_enabled_guideline_matches.update(
                {key[0]: tool_enabled[key] for key in tool_enabled}
            )
        else:
            context.state.tool_enabled_guideline_matches = {
                key[0]: tool_enabled[key] for key in tool_enabled
            }
            context.state.ordinary_guideline_matches = [
                m[0] for m in set(matches).difference(set(tool_enabled.keys()))
            ]

        await self._store_session_guidelines(context, matches)

        context.state.invalidate_cached_properties()

    async def _find_tool_enabled_guideline_matches(
        self,
        guideline_matches: Sequence[tuple[GuidelineMatch, _ContextUsage]],
    ) -> dict[tuple[GuidelineMatch, _ContextUsage], list[ToolId]]:
        matches_by_id = {m[0].guideline.id: m for m in guideline_matches}

        tools_for_guidelines: dict[tuple[GuidelineMatch, _ContextUsage], list[ToolId]] = (
            defaultdict(list)
        )

        for association in await self._entity_queries.find_guideline_tool_associations():
            if association.guideline_id in matches_by_id:
                tools_for_guidelines[matches_by_id[association.guideline_id]].append(
                    association.tool_id
                )

        return dict(tools_for_guidelines)

    def _matched_guideline_ids(self, context: EngineContext) -> set[GuidelineId]:
        return {
            m.guideline.id
            for m in chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches,
            )
        }

    def _executed_tool_ids(self, context: EngineContext) -> set[ToolId]:
        # The react loop records executed tools (by name) in state.tool_events;
        # map them back to ToolIds via the per-turn name->id table.
        executed: set[ToolId] = set()
        for event in context.state.tool_events:
            for call in cast(ToolEventData, event.data)["tool_calls"]:
                if tool_id := context.state.tool_ids_by_name.get(call["tool_id"]):
                    executed.add(tool_id)
        return executed

    async def _find_guidelines_gated_on_tools(
        self,
        context: EngineContext,
        tool_ids: set[ToolId],
    ) -> list[Guideline]:
        usable_by_id = {g.id: g for g in context.state.usable_guidelines}
        gated: dict[GuidelineId, Guideline] = {}

        for tool_id in tool_ids:
            relationships = await self._relationship_store.list_relationships(
                kind=RelationshipKind.REEVALUATION,
                indirect=False,
                target_id=tool_id,
            )

            for relationship in relationships:
                # Source is a guideline (match by id prefix, as elsewhere) or a tag
                # (match by tag membership).
                matched = [
                    g for gid, g in usable_by_id.items() if gid.startswith(relationship.source.id)
                ]

                if not matched and relationship.source.kind.is_tag:
                    matched = [g for g in usable_by_id.values() if relationship.source.id in g.tags]

                for guideline in matched:
                    gated[guideline.id] = guideline

        return list(gated.values())

    # --- tool selection ---

    async def _select_tools(self, context: EngineContext) -> None:
        # Resolve the matched guidelines' tools, then fold them with the ranked
        # pool into the offered catalog. Idempotent when nothing changed (so the
        # rendered prompt stays byte-identical), and picks up any tools a
        # reevaluated tool-enabled guideline brought in.
        await self._resolve_matched_tools(context)
        self._select_available_tools(context)

    async def _resolve_matched_tools(self, context: EngineContext) -> None:
        tool_ids = list(
            dict.fromkeys(
                tool_id
                for tool_ids in context.state.tool_enabled_guideline_matches.values()
                for tool_id in tool_ids
            )
        )
        context.state.matched_tools = await self._resolve_tools_by_id(tool_ids)

    async def _rank_tool_pool(self, context: EngineContext) -> None:
        # Rank the agent's candidate tools against the agent description + the
        # conversation, scoped per service. Each service ranks only its own tools;
        # we merge the scored results across services.
        candidate_ids = await self._agent_candidate_tool_ids(context)
        # Map names back to ToolIds so a tool call (which carries only a name) can
        # be routed to its service when run.
        context.state.tool_ids_by_name = {tid.tool_name: tid for tid in candidate_ids}

        if not candidate_ids:
            context.state.agent_tool_pool = []
            return

        query = self._build_tool_query(context)

        names_by_service: dict[str, list[str]] = defaultdict(list)
        for tool_id in candidate_ids:
            names_by_service[tool_id.service_name].append(tool_id.tool_name)

        results: list[ToolRelevanceResult] = []
        for service_name, names in names_by_service.items():
            try:
                service = await self._entity_queries.read_tool_service(service_name)
                results.extend(
                    await service.find_relevant_tools(query, names, self._MAX_AVAILABLE_TOOLS)
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to rank tools for service {service_name}: {e!r}\n"
                    f"{traceback.format_exc()}"
                )

        results.sort(key=lambda r: r.score, reverse=True)
        context.state.agent_tool_pool = [r.tool for r in results]

    def _select_available_tools(self, context: EngineContext) -> None:
        # Matched-turn tools are always included; fill up to _MAX_AVAILABLE_TOOLS
        # with the most relevant general tools.
        chosen: list[Tool] = list(context.state.matched_tools)
        seen = {tool.name for tool in chosen}
        for tool in context.state.agent_tool_pool:
            if len(chosen) >= self._MAX_AVAILABLE_TOOLS:
                break
            if tool.name not in seen:
                seen.add(tool.name)
                chosen.append(tool)

        # Emit by name so an unchanged selection is byte-identical turn to turn,
        # keeping the cached tools prefix warm (selection uses scores; emission
        # order is stable).
        context.state.available_tools = sorted(chosen, key=lambda tool: tool.name)

    def _build_tool_query(self, context: EngineContext) -> str:
        return (
            f"{context.agent.description or ''}\n\n{self._build_interaction_query_lines(context)}"
        )

    # --- glossary ---

    async def _load_glossary(self, context: EngineContext) -> None:
        # Load the glossary terms most relevant to the conversation so far (capped at
        # _MAX_GLOSSARY_TERMS) into the state, so the responder can surface them in its
        # (cached) system instructions. Loaded once here, not per response step.
        terms = await self._entity_queries.find_glossary_terms_for_context(
            context.agent.id,
            query=str(self._build_interaction_query_lines(context)),
            max_terms=self._MAX_GLOSSARY_TERMS,
        )
        context.state.glossary_terms = set(terms)

    def _build_interaction_query_lines(self, context: EngineContext) -> list[str]:
        lines: list[str] = []

        if context.state.session_summary:
            lines.append(f"Session summary: {context.state.session_summary}")

        lines.extend(f"{m.source}: {m.content}" for m in context.interaction.messages)

        if not lines:
            # No conversation yet (the initialize-time warm-up). Rank against a neutral
            # greeting so the glossary still loads — letting the warmed prefix include it
            # and match the first real turn. With a glossary under the cap the full set
            # is returned regardless of query, so it matches that turn exactly.
            lines.append("User: Hello")

        return lines

    async def _agent_candidate_tool_ids(self, context: EngineContext) -> set[ToolId]:
        guideline_ids = {g.id for g in context.state.usable_guidelines}
        return {
            association.tool_id
            for association in await self._entity_queries.find_guideline_tool_associations()
            if association.guideline_id in guideline_ids
        }

    async def _load_tools_by_guideline(self, context: EngineContext) -> None:
        # Load the tools associated with each guideline into the state, so they can be
        # looked up when a guideline matches. Loaded once here, not per response step.
        context.state.tools_by_guideline = defaultdict(set)

        guideline_tool_associations = await self._entity_queries.find_guideline_tool_associations()

        for association in guideline_tool_associations:
            if tool := await self._resolve_tool_by_id(association.tool_id):
                context.state.tools_by_guideline[association.guideline_id].add(
                    (association.tool_id, tool)
                )

    async def _resolve_tools_by_id(self, tool_ids: Iterable[ToolId]) -> list[Tool]:
        tools: list[Tool] = []

        for tool_id in tool_ids:
            if tool := await self._resolve_tool_by_id(tool_id):
                tools.append(tool)

        return tools

    async def _resolve_tool_by_id(self, tool_id: ToolId) -> Tool | None:
        try:
            service = await self._entity_queries.read_tool_service(tool_id.service_name)
            return await service.read_tool(tool_id.tool_name)
        except Exception as e:
            self._logger.warning(f"Failed to resolve tool {tool_id.to_string()}: {e}")
            return None
