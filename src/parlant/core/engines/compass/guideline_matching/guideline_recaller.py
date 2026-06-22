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

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
import time

from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.common import xxh3_checksum
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.nlp.embedding import Embedder, EmbeddingCache
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import EventSource
from parlant.core.tracer import Tracer


_EPSILON = 1e-12


@dataclass(frozen=True)
class RecalledGuideline:
    guideline: Guideline
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class GuidelineRecallResult:
    recalled_guidelines: Sequence[RecalledGuideline]
    duration: float


@dataclass(frozen=True)
class _PolicyFrame:
    centroid: tuple[float, ...]
    directions: dict[GuidelineId, tuple[tuple[float, ...], ...]]


class GuidelineRecaller:
    DEFAULT_RECALL_MARGIN = 0.03
    _MAX_POLICY_FRAME_CACHE_SIZE = 32

    def __init__(
        self,
        nlp_service: NLPService,
        tracer: Tracer,
        embedding_cache: EmbeddingCache,
        recall_margin: float = DEFAULT_RECALL_MARGIN,
    ) -> None:
        self._nlp_service = nlp_service
        self._tracer = tracer
        self._embedding_cache = embedding_cache
        self._recall_margin = recall_margin
        self._policy_frame_cache: OrderedDict[
            tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], _PolicyFrame
        ] = OrderedDict()

    async def recall(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> GuidelineRecallResult:
        with self._tracer.span("guideline.recall"):
            started_at = time.time()
            recalled_guidelines = await self._do_recall(context, guidelines)
            return GuidelineRecallResult(
                recalled_guidelines=recalled_guidelines,
                duration=time.time() - started_at,
            )

    async def _do_recall(
        self,
        context: EngineContext,
        guidelines: Sequence[Guideline],
    ) -> Sequence[RecalledGuideline]:
        if not guidelines:
            return []

        queries = self._build_queries(context)

        if not queries:
            return []

        if len(guidelines) == 1:
            return [
                RecalledGuideline(
                    guideline=guidelines[0],
                    is_relevant=True,
                    score=0.0,
                )
            ]

        embedder = await self._nlp_service.get_embedder()
        frame = await self._get_policy_frame(embedder, guidelines)
        query_vectors = await self._embed_many(embedder, queries)
        query_directions = [
            direction
            for vector in query_vectors
            if (direction := self._normalize(self._subtract(vector, frame.centroid))) is not None
        ]

        if not query_directions:
            return []

        return [
            RecalledGuideline(
                guideline=guideline,
                is_relevant=score > -self._recall_margin,
                score=score,
            )
            for guideline in guidelines
            if (policy_directions := frame.directions.get(guideline.id))
            for score in [
                max(
                    self._dot(query_direction, policy_direction)
                    for query_direction in query_directions
                    for policy_direction in policy_directions
                )
            ]
        ]

    async def _get_policy_frame(
        self,
        embedder: Embedder,
        guidelines: Sequence[Guideline],
    ) -> _PolicyFrame:
        ordered_policy_specs = sorted(
            ((g, self._list_guideline_contents(g)) for g in guidelines),
            key=lambda spec: str(spec[0].id),
        )
        ordered_guidelines = [guideline for guideline, _ in ordered_policy_specs]
        ordered_policy_contents = [contents for _, contents in ordered_policy_specs]
        key = (
            embedder.id,
            tuple(
                (str(g.id), tuple(xxh3_checksum(content) for content in contents))
                for g, contents in zip(ordered_guidelines, ordered_policy_contents)
            ),
        )

        if frame := self._policy_frame_cache.get(key):
            self._policy_frame_cache.move_to_end(key)
            return frame

        vectors_by_guideline = await self._embed_guideline_entities(
            embedder,
            ordered_guidelines,
            ordered_policy_contents,
        )
        centroid = self._centroid(
            [
                vector
                for vectors in vectors_by_guideline.values()
                for vector in vectors
            ]
        )
        directions = {
            guideline_id: tuple(
                direction
                for vector in vectors
                if (direction := self._normalize(self._subtract(vector, centroid))) is not None
            )
            for guideline_id, vectors in vectors_by_guideline.items()
        }
        frame = _PolicyFrame(centroid=centroid, directions=directions)

        self._policy_frame_cache[key] = frame
        self._policy_frame_cache.move_to_end(key)

        while len(self._policy_frame_cache) > self._MAX_POLICY_FRAME_CACHE_SIZE:
            self._policy_frame_cache.popitem(last=False)

        return frame

    async def _embed_guideline_entities(
        self,
        embedder: Embedder,
        guidelines: Sequence[Guideline],
        guideline_contents: Sequence[Sequence[str]],
    ) -> dict[GuidelineId, tuple[tuple[float, ...], ...]]:
        contents = [content for contents in guideline_contents for content in contents]
        content_vectors = await self._embed_many(embedder, contents)
        vectors_by_content: dict[str, tuple[float, ...]] = dict(zip(contents, content_vectors))

        return {
            guideline.id: tuple(vectors_by_content[content] for content in contents)
            for guideline, contents in zip(guidelines, guideline_contents)
        }

    async def _embed_one(
        self,
        embedder: Embedder,
        text: str,
    ) -> tuple[float, ...]:
        if cached_result := await self._embedding_cache.get(
            embedder_type=type(embedder),
            texts=[text],
        ):
            return self._as_tuple(cached_result.vectors[0])

        result = await embedder.embed([text])
        await self._embedding_cache.set(
            embedder_type=type(embedder),
            texts=[text],
            vectors=result.vectors,
        )
        return self._as_tuple(result.vectors[0])

    async def _embed_many(
        self,
        embedder: Embedder,
        texts: Sequence[str],
    ) -> list[tuple[float, ...]]:
        cached_vectors: list[tuple[float, ...] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            if cached_result := await self._embedding_cache.get(
                embedder_type=type(embedder),
                texts=[text],
            ):
                cached_vectors[index] = self._as_tuple(cached_result.vectors[0])
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            result = await embedder.embed(missing_texts)

            for index, text, vector in zip(missing_indices, missing_texts, result.vectors):
                await self._embedding_cache.set(
                    embedder_type=type(embedder),
                    texts=[text],
                    vectors=[vector],
                )
                cached_vectors[index] = self._as_tuple(vector)

        assert all(v is not None for v in cached_vectors)
        return [v for v in cached_vectors if v is not None]

    def _build_queries(self, context: EngineContext) -> list[str]:
        queries = [
            query
            for query in (
                self._build_cumulative_query(context),
                self._build_latest_customer_message_query(context),
            )
            if query
        ]

        return list(dict.fromkeys(queries))

    def _build_cumulative_query(self, context: EngineContext) -> str:
        if not context.interaction.messages and not context.state.session_summary:
            return ""

        lines: list[str] = []

        if context.state.session_summary:
            lines.append(f"Session summary: {context.state.session_summary}")

        lines.extend(f"{m.source}: {m.content}" for m in context.interaction.messages)

        return "\n".join(lines)

    def _build_latest_customer_message_query(self, context: EngineContext) -> str:
        latest_customer_message = next(
            (
                message
                for message in reversed(context.interaction.messages)
                if message.source == EventSource.CUSTOMER
            ),
            None,
        )

        if not latest_customer_message:
            return ""

        return f"{latest_customer_message.source}: {latest_customer_message.content}"

    def _list_guideline_contents(self, guideline: Guideline) -> list[str]:
        return [self._guideline_embedding_content(guideline), *guideline.signals]

    def _guideline_embedding_content(self, guideline: Guideline) -> str:
        content = guideline.content

        condition = (content.condition or "").strip()
        action = (content.action or "").strip()
        description = (content.description or "").strip()

        if guideline.title:
            head = f"# {guideline.title}\n\n"
        else:
            head = ""

        if condition and action:
            head += f"When {condition}, then {action}"
        elif condition:
            head += f"Condition: {condition}"
        elif action:
            head += f"Action: {action}"
        else:
            raise ValueError("Guideline must have at least a condition or an action")

        if description:
            return f"{head}\n\n{description}"

        return head

    def _as_tuple(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in vector)

    def _centroid(self, vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
        dimensions = len(vectors[0])
        return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(dimensions))

    def _subtract(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> tuple[float, ...]:
        return tuple(left_value - right_value for left_value, right_value in zip(left, right))

    def _normalize(self, vector: Sequence[float]) -> tuple[float, ...] | None:
        norm = sqrt(sum(v * v for v in vector))

        if norm <= _EPSILON:
            return None

        return tuple(v / norm for v in vector)

    def _dot(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))
