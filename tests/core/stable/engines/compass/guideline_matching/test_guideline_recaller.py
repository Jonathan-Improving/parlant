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

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.engine_context import EngineContext
from parlant.core.guidelines import Guideline
from parlant.core.nlp.embedding import Embedder, EmbeddingCache, EmbeddingResult, NullEmbeddingCache
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
)


@fixture
def recaller(container: Container) -> GuidelineRecaller:
    container[EmbeddingCache] = NullEmbeddingCache()
    return container[GuidelineRecaller]


def test_that_a_guideline_recaller_can_be_created(recaller: GuidelineRecaller) -> None:
    assert recaller is not None


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


class _TopicShiftEmbedder(_FakeEmbedder):
    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.lower()

        if "old package issue" in lowered and "money back" in lowered:
            return [0.0, 1.0]

        return super()._vector_for_text(text)


class _CentralityBoostEmbedder(_FakeEmbedder):
    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.lower()

        if "central policy" in lowered:
            return [0.1, -0.005]

        if "east policy" in lowered:
            return [1.0, 0.0]

        if "west policy" in lowered:
            return [-1.0, 0.01]

        if "north request" in lowered:
            return [0.0, 1.0]

        return super()._vector_for_text(text)


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
    centrality_boost_beta: float = GuidelineRecaller.DEFAULT_CENTRALITY_BOOST_BETA,
) -> GuidelineRecaller:
    return GuidelineRecaller(
        nlp_service=_FakeNLPService(embedder or _FakeEmbedder()),  # type: ignore[arg-type]
        tracer=create_engine_context(conversation=[]).tracer,
        embedding_cache=embedding_cache or _FakeEmbeddingCache(),  # type: ignore[arg-type]
        centrality_boost_beta=centrality_boost_beta,
    )


def _context(conversation: list[tuple[EventSource, str]]) -> EngineContext[Any]:
    context = create_engine_context(conversation=conversation)
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


async def test_that_the_recaller_uses_centroid_relative_relevance() -> None:
    recaller = _radar_recaller()
    guidelines = _create_sample_guidelines()
    available = list(guidelines.values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    result = await recaller.recall(context, available)

    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}
    scores_by_id = {r.guideline.id: r.score for r in result.recalled_guidelines}

    assert result.duration >= 0.0
    assert len(result.recalled_guidelines) == 3
    assert relevance_by_id[guidelines["refund"].id]
    assert not relevance_by_id[guidelines["hours"].id]
    assert not relevance_by_id[guidelines["shipping"].id]
    assert scores_by_id[guidelines["refund"].id] > 0.0


async def test_that_the_recaller_unions_cumulative_and_latest_user_message_relevance() -> None:
    recaller = _radar_recaller(_TopicShiftEmbedder())
    guidelines = _create_sample_guidelines()
    available = list(guidelines.values())
    context = _context(
        [
            (EventSource.CUSTOMER, "I have an old package issue"),
            (EventSource.AI_AGENT, "I can check that for you."),
            (EventSource.CUSTOMER, "Actually, I want my money back"),
        ]
    )

    result = await recaller.recall(context, available)

    scores_by_id = {r.guideline.id: r.score for r in result.recalled_guidelines}
    relevance_by_id = {r.guideline.id: r.is_relevant for r in result.recalled_guidelines}

    assert relevance_by_id[guidelines["refund"].id]
    assert scores_by_id[guidelines["refund"].id] > 0.0


async def test_that_the_recaller_boosts_near_centroid_policy_entities() -> None:
    central = create_guideline(
        condition="central policy applies",
        action="follow the central policy",
        tags=[],
    )
    east = create_guideline(
        condition="east policy applies",
        action="follow the east policy",
        tags=[],
    )
    west = create_guideline(
        condition="west policy applies",
        action="follow the west policy",
        tags=[],
    )
    context = _context([(EventSource.CUSTOMER, "north request")])

    unboosted_result = await _radar_recaller(
        _CentralityBoostEmbedder(),
        centrality_boost_beta=0.0,
    ).recall(context, [central, east, west])
    boosted_result = await _radar_recaller(_CentralityBoostEmbedder()).recall(
        context,
        [central, east, west],
    )

    unboosted_central = next(
        r for r in unboosted_result.recalled_guidelines if r.guideline.id == central.id
    )
    boosted_central = next(
        r for r in boosted_result.recalled_guidelines if r.guideline.id == central.id
    )

    assert not unboosted_central.is_relevant
    assert boosted_central.is_relevant
    assert boosted_central.score > unboosted_central.score


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
