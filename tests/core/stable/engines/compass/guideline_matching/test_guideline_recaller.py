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

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.guideline_matching.guideline_recaller import (
    GuidelineRecaller,
    _LogisticModel,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.engine_context import EngineContext
from parlant.core.guidelines import Guideline, GuidelineContent
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.sessions import EventSource

from parlant.core.agents import Agent

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_agent,
    create_engine_context,
    create_guideline,
)


@fixture
def recaller(container: Container) -> GuidelineRecaller:
    return container[GuidelineRecaller]


def test_that_a_guideline_recaller_can_be_created(recaller: GuidelineRecaller) -> None:
    assert recaller is not None


def test_that_the_logistic_model_separates_a_linearly_separable_set() -> None:
    rng = np.random.default_rng(0)
    positives = rng.normal(loc=[2.0, 2.0], scale=0.3, size=(30, 2))
    negatives = rng.normal(loc=[-2.0, -2.0], scale=0.3, size=(30, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([1] * 30 + [0] * 30)

    model = _LogisticModel.fit(features, labels, C=0.5)

    # Held-out points on each side fall on the correct side of the boundary.
    assert model.decision(np.array([[2.0, 2.0]]))[0] > 0.0
    assert model.decision(np.array([[-2.0, -2.0]]))[0] < 0.0
    # The set is separable, so every positive outscores every negative.
    assert model.decision(positives).min() > model.decision(negatives).max()


def test_that_the_logistic_model_balances_class_weights() -> None:
    # One lone positive against many negatives: balanced weighting must keep it
    # from being drowned out (an unweighted fit would put it below the boundary).
    rng = np.random.default_rng(1)
    positives = np.array([[3.0, 0.0]])
    negatives = rng.normal(loc=[-1.0, 0.0], scale=0.5, size=(100, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([1] + [0] * 100)

    model = _LogisticModel.fit(features, labels, C=0.5, class_weight="balanced")

    assert model.decision(positives)[0] > 0.0


class _FakeTokenizer(EstimatingTokenizer):
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str], hints: Mapping[str, Any] = {}) -> EmbeddingResult:
        self.embed_calls.append(texts)
        return EmbeddingResult(vectors=[self._vector_for_text(text) for text in texts])

    @property
    def id(self) -> str:
        return "fake-radar-embedder"

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return _FakeTokenizer()

    @property
    def dimensions(self) -> int:
        return 2

    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.lower()

        if any(term in lowered for term in ("refund", "money back", "money")):
            return [1.0, 0.0]

        if any(term in lowered for term in ("hours", "open")):
            return [-1.0, 0.0]

        if any(term in lowered for term in ("package", "shipping", "delivery")):
            return [0.0, 1.0]

        return [0.0, -1.0]


class _FakeNLPService:
    def __init__(self, embedder: _FakeEmbedder) -> None:
        self._embedder = embedder

    async def get_embedder(self, hints: Mapping[str, Any] = {}) -> _FakeEmbedder:
        return self._embedder


class _FakeEmbeddingCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[type[Embedder], tuple[str, ...]], EmbeddingResult] = {}

    async def get(
        self, embedder_type: type[Embedder], texts: list[str], hints: Mapping[str, Any] = {}
    ) -> EmbeddingResult | None:
        return self._entries.get((embedder_type, tuple(texts)))

    async def set(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        vectors: list[list[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        self._entries[(embedder_type, tuple(texts))] = EmbeddingResult(vectors=vectors)


def _radar_recaller(
    embedder: _FakeEmbedder | None = None,
    embedding_cache: _FakeEmbeddingCache | None = None,
    **kwargs: Any,
) -> GuidelineRecaller:
    return GuidelineRecaller(
        nlp_service=_FakeNLPService(embedder or _FakeEmbedder()),  # type: ignore[arg-type]
        tracer=create_engine_context(conversation=[]).tracer,
        embedding_cache=embedding_cache or _FakeEmbeddingCache(),  # type: ignore[arg-type]
        **kwargs,
    )


def _context(
    conversation: list[tuple[EventSource, str]],
    agent: Agent | None = None,
) -> EngineContext[Any]:
    context = create_engine_context(conversation=conversation, agent=agent)
    context.state = ResponseState()
    return context


def _create_sample_guidelines() -> dict[str, Guideline]:
    refund = create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        tags=[],
    )
    hours = create_guideline(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        tags=[],
    )
    shipping = create_guideline(
        condition="the customer asks where their package is",
        action="share the shipping status",
        tags=[],
    )

    return {"refund": refund, "hours": hours, "shipping": shipping}


async def test_that_the_recaller_recalls_the_discriminating_policy() -> None:
    recaller = _radar_recaller()
    guidelines = _create_sample_guidelines()
    available = list(guidelines.values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    result = await recaller.recall(context, available)

    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}

    assert result.duration >= 0.0
    assert len(result.recalled_guidelines) == 3
    assert relevance_by_id[guidelines["refund"].id]
    assert not relevance_by_id[guidelines["hours"].id]
    assert not relevance_by_id[guidelines["shipping"].id]


async def test_that_the_recaller_stays_sticky_across_user_turns() -> None:
    # The refund-relevant turn is earlier in the conversation; max-over-turns must
    # keep the refund policy relevant even though the latest turn is off-topic.
    recaller = _radar_recaller()
    guidelines = _create_sample_guidelines()
    available = list(guidelines.values())
    context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "what are your opening hours?"),
        ]
    )

    result = await recaller.recall(context, available)

    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}

    assert relevance_by_id[guidelines["refund"].id]


async def test_that_the_recaller_embeds_policy_signals() -> None:
    recaller = _radar_recaller()
    refund = replace(
        create_guideline(
            condition="the customer asks for account support",
            action="start the account support flow",
            tags=[],
        ),
        signals=["I want my money back"],
    )
    hours = create_guideline(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        tags=[],
    )
    shipping = create_guideline(
        condition="the customer asks where their package is",
        action="share the shipping status",
        tags=[],
    )

    context = _context([(EventSource.CUSTOMER, "I want my money back please")])

    result = await recaller.recall(context, [refund, hours, shipping])

    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}
    assert relevance_by_id[refund.id]


def test_that_the_recaller_formats_description_only_policy_guideline() -> None:
    recaller = _radar_recaller()
    guideline = replace(
        create_guideline(condition="", action=None, tags=[]),
        title="Refund Policy",
        content=GuidelineContent(
            condition="",
            action=None,
            description="Refunds are allowed within 30 days.",
        ),
    )

    assert recaller._guideline_embedding_content(guideline) == (
        "# Refund Policy\n\nRefunds are allowed within 30 days."
    )


def test_that_the_recaller_embeds_policy_guideline_text_and_signals() -> None:
    recaller = _radar_recaller()
    guideline = replace(
        create_guideline(condition="", action=None, tags=[]),
        title="Refund Policy",
        content=GuidelineContent(
            condition="",
            action=None,
            description="Refunds are allowed within 30 days.",
        ),
        signals=["I want my money back"],
    )

    assert recaller._list_guideline_contents(guideline) == [
        "# Refund Policy\n\nRefunds are allowed within 30 days.",
        "I want my money back",
    ]


async def test_that_the_recaller_reuses_the_cached_policy_frame() -> None:
    embedder = _FakeEmbedder()
    recaller = _radar_recaller(embedder)
    guidelines = list(_create_sample_guidelines().values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    await recaller.recall(context, guidelines)
    await recaller.recall(context, guidelines)

    assert len(embedder.embed_calls) == 2
    assert len(embedder.embed_calls[0]) == 3
    assert len(embedder.embed_calls[1]) == 1


async def test_that_the_recaller_uses_the_persistent_embedding_cache() -> None:
    embedder = _FakeEmbedder()
    embedding_cache = _FakeEmbeddingCache()
    recaller_1 = _radar_recaller(embedder, embedding_cache)
    recaller_2 = _radar_recaller(embedder, embedding_cache)
    guidelines = list(_create_sample_guidelines().values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    await recaller_1.recall(context, guidelines)
    await recaller_2.recall(context, guidelines)

    assert len(embedder.embed_calls) == 2


async def test_that_the_recaller_returns_nothing_for_an_empty_interaction() -> None:
    recaller = _radar_recaller()
    guidelines = _create_sample_guidelines()

    context = _context([])

    result = await recaller.recall(context, list(guidelines.values()))

    assert result.recalled_guidelines == []


async def test_that_the_recaller_includes_a_single_candidate_guideline() -> None:
    recaller = _radar_recaller()
    guideline = create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        tags=[],
    )
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    result = await recaller.recall(context, [guideline])

    assert result.recalled_guidelines[0].guideline.id == guideline.id
    assert result.recalled_guidelines[0].is_relevant


async def test_that_retrain_reports_progress_and_warms_recall() -> None:
    recaller = _radar_recaller()
    guidelines = list(_create_sample_guidelines().values())
    agent = create_agent()

    seen: list[float] = []

    async def on_progress(percentage: float) -> None:
        seen.append(percentage)

    report = ProgressReport(on_progress)
    await recaller.retrain(agent.id, guidelines, report)

    assert report.percentage == 100.0
    assert seen and seen[-1] == 100.0

    # Recall for this agent now serves off the trained frame.
    context = _context([(EventSource.CUSTOMER, "I'd like my money back")], agent=agent)
    result = await recaller.recall(context, guidelines)
    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}
    assert relevance_by_id[guidelines[0].id] or any(relevance_by_id.values())


async def test_that_retrain_calibrates_thresholds_from_negatives() -> None:
    recaller = _radar_recaller()
    guidelines = list(_create_sample_guidelines().values())
    agent = create_agent()

    await recaller.retrain(agent.id, guidelines)

    frame = recaller._frames_by_agent[agent.id]
    for guideline in guidelines:
        policy = frame.by_guideline[guideline.id]
        # Every policy has negatives (the other policies), so its threshold is a real
        # negative-calibrated percentile — not the degenerate "never fire" infinity.
        assert math.isfinite(policy.threshold)


async def test_that_each_agent_gets_its_own_trained_frame() -> None:
    recaller = _radar_recaller()
    guidelines = list(_create_sample_guidelines().values())
    agent_a = create_agent()
    agent_b = create_agent()

    await recaller.retrain(agent_a.id, guidelines)

    # Agent A is trained; agent B is not — frames are strictly per-agent.
    assert agent_a.id in recaller._frames_by_agent
    assert agent_b.id not in recaller._frames_by_agent


async def test_that_a_pinned_signal_forces_recall() -> None:
    hours = create_guideline(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        tags=[],
    )
    refund = create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        tags=[],
    )
    shipping = create_guideline(
        condition="the customer asks where their package is",
        action="share the shipping status",
        tags=[],
    )
    context = _context([(EventSource.CUSTOMER, "I want my money back")])

    plain = await _radar_recaller().recall(context, [hours, refund, shipping])
    plain_hours = next(r for r in plain.recalled_guidelines if r.guideline.id == hours.id)
    assert not plain_hours.is_relevant

    pinned_hours = replace(hours, signals=["[__pin__]I want my money back"])
    pinned = await _radar_recaller(pin_match_epsilon=0.5).recall(
        context, [pinned_hours, refund, shipping]
    )
    pinned_result = next(r for r in pinned.recalled_guidelines if r.guideline.id == hours.id)
    assert pinned_result.is_relevant


async def test_that_pin_prefixed_signals_become_must_fire_exemplars() -> None:
    recaller = _radar_recaller(pin_match_epsilon=0.5)
    hours = replace(
        create_guideline(
            condition="the customer asks about opening hours",
            action="tell them the store hours",
            tags=[],
        ),
        signals=["[__pin__]I want my money back"],
    )
    refund = create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        tags=[],
    )

    agent = create_agent()
    await recaller.retrain(agent.id, [hours, refund])

    assert len(recaller._frames_by_agent[agent.id].by_guideline[hours.id].pin_exemplars) == 1
