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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp
from typing import Literal, Optional
import time

import numpy as np
import numpy.typing as npt

from parlant.core.async_utils import safe_gather
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.agents import AgentId
from parlant.core.common import xxh3_checksum
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.nlp.embedding import Embedder, EmbeddingCache
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import EventSource
from parlant.core.tracer import Tracer


_EPSILON = 1e-12


@dataclass(frozen=True, eq=False)
class _LogisticModel:
    """A minimal L2-regularized logistic regression, fit by batch gradient descent.

    Kept dependency-light (numpy only) — the recaller trains one of these per policy
    to discriminate that policy's signals from every other policy's. The objective
    mirrors scikit-learn's parameterization: ``C * mean(weighted log-loss) + 0.5||w||^2``,
    so ``C`` is the inverse regularization strength. ``decision`` returns the raw
    ``w·x + b`` (the classifier's ``decision_function``), which is what the recaller
    thresholds and pools over.
    """

    weights: npt.NDArray[np.float64]
    bias: float

    @staticmethod
    def fit(
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        *,
        C: float = 0.5,
        class_weight: Optional[Literal["balanced"]] = None,
        iterations: int = 1000,
        learning_rate: float = 1.0,
    ) -> "_LogisticModel":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        n, d = x.shape

        if class_weight == "balanced":
            positives = max(1.0, float(y.sum()))
            negatives = max(1.0, float(n) - positives)
            sample_weight = np.where(y == 1.0, n / (2.0 * positives), n / (2.0 * negatives))
        else:
            sample_weight = np.ones(n, dtype=np.float64)

        weight_mass = float(sample_weight.sum()) or 1.0
        w = np.zeros(d, dtype=np.float64)
        b = 0.0

        for _ in range(iterations):
            z = x @ w + b
            predictions = 1.0 / (1.0 + np.exp(-z))
            weighted_error = sample_weight * (predictions - y)
            # Mean weighted data-loss gradient (so step size is scale-stable in n),
            # plus the L2 term on the weights only.
            grad_w = C * (x.T @ weighted_error) / weight_mass + w
            grad_b = C * float(weighted_error.sum()) / weight_mass
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        return _LogisticModel(weights=w, bias=b)

    def decision(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(features, dtype=np.float64) @ self.weights + self.bias


@dataclass(frozen=True)
class RecalledGuideline:
    guideline: Guideline
    is_relevant: bool
    score: float


@dataclass(frozen=True)
class GuidelineRecallResult:
    recalled_guidelines: Sequence[RecalledGuideline]
    duration: float


@dataclass(frozen=True, eq=False)
class _TrainedPolicy:
    """A policy's trained relevance discriminant plus its decision boundary.

    ``model`` separates this policy's own (content + signal) directions from every
    other policy's. ``threshold`` is the negative-calibrated operating point — the
    q-th percentile of the *other* policies' decision scores. ``pin_exemplars`` are
    raw unit vectors of ``[__pin__]``-prefixed signals: must-fire overrides that
    force relevance when the live conversation is within ``pin_match_epsilon``.
    """

    model: _LogisticModel
    threshold: float
    pin_exemplars: tuple[npt.NDArray[np.float64], ...]


@dataclass(frozen=True, eq=False)
class _TrainedFrame:
    centroid: npt.NDArray[np.float64]
    by_guideline: Mapping[GuidelineId, _TrainedPolicy]


class GuidelineRecaller:
    DEFAULT_RECALL_THRESHOLD_PERCENTILE = 70.0
    DEFAULT_LOGISTIC_C = 0.5
    DEFAULT_PIN_MATCH_EPSILON = 0.9
    DEFAULT_MAX_NEGATIVES = 2000
    PIN_PREFIX = "[__pin__]"
    _MAX_FRAME_CACHE_SIZE = 8

    def __init__(
        self,
        nlp_service: NLPService,
        tracer: Tracer,
        embedding_cache: EmbeddingCache,
        recall_threshold_percentile: float = DEFAULT_RECALL_THRESHOLD_PERCENTILE,
        logistic_C: float = DEFAULT_LOGISTIC_C,
        pin_match_epsilon: float = DEFAULT_PIN_MATCH_EPSILON,
        max_negatives: int = DEFAULT_MAX_NEGATIVES,
    ) -> None:
        self._nlp_service = nlp_service
        self._tracer = tracer
        self._embedding_cache = embedding_cache
        self._recall_threshold_percentile = recall_threshold_percentile
        self._logistic_C = logistic_C
        self._pin_match_epsilon = pin_match_epsilon
        self._max_negatives = max_negatives
        # One trained frame per agent. A policy's discriminant and its
        # negative-calibrated threshold are only meaningful relative to that agent's
        # OWN competing policies, so frames are never shared across agents.
        self._frames_by_agent: dict[AgentId, _TrainedFrame] = {}
        # Backstop LRU keyed by (agent, inventory checksum) for the window before an
        # agent has been trained (or for a batch its frame doesn't cover): lazily
        # train on the batch.
        self._frame_cache: OrderedDict[
            tuple[AgentId, str, tuple[tuple[str, tuple[str, ...]], ...]], _TrainedFrame
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

    async def retrain(
        self,
        agent_id: AgentId,
        guidelines: Sequence[Guideline],
        progress_report: Optional[ProgressReport] = None,
    ) -> None:
        """Train an agent's per-policy discriminants against that agent's own
        guideline space (``guidelines`` should be the agent's full inventory).

        Offline, idempotent, and reported per guideline via ``progress_report`` (for
        the SDK progress bar / the POST /train job monitor). Stores the agent's frame,
        which recall then uses directly — inference stays a dot product.
        """
        with self._tracer.span("guideline.recall.retrain"):
            if not guidelines:
                self._frames_by_agent[agent_id] = _TrainedFrame(
                    centroid=np.zeros(0, dtype=np.float64), by_guideline={}
                )
                return

            embedder = await self._nlp_service.get_embedder()
            frame = await self._train_frame(embedder, guidelines, progress_report)
            self._frames_by_agent[agent_id] = frame
            self._cache_frame(self._frame_key(agent_id, embedder, guidelines), frame)

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
        frame, query_vectors = await safe_gather(
            self._frame_for(context.agent.id, embedder, guidelines),
            self._embed_many(embedder, queries),
        )

        queries_array = np.asarray(query_vectors, dtype=np.float64)
        directions, direction_valid = self._row_units(queries_array - frame.centroid)
        raw_units, raw_valid = self._row_units(queries_array)
        directions = directions[direction_valid]
        pin_query_units = raw_units[raw_valid]

        if directions.shape[0] == 0:
            return []

        recalled: list[RecalledGuideline] = []
        for guideline in guidelines:
            policy = frame.by_guideline.get(guideline.id)
            if policy is None:
                # Not covered by the current frame (stale inventory; awaiting a
                # /train). Skip rather than guess.
                continue

            scores = policy.model.decision(directions)
            decision_score = float(scores.max())
            classifier_margin = decision_score - policy.threshold
            pin_similarity = self._max_pin_similarity(pin_query_units, policy.pin_exemplars)
            classifier_relevant = classifier_margin > 0.0
            pin_relevant = (
                pin_similarity is not None and pin_similarity > self._pin_match_epsilon
            )
            recalled.append(
                RecalledGuideline(
                    guideline=guideline,
                    is_relevant=classifier_relevant or pin_relevant,
                    score=self._calculate_displayed_score(classifier_margin, pin_similarity),
                )
            )

        return recalled

    async def _frame_for(
        self,
        agent_id: AgentId,
        embedder: Embedder,
        guidelines: Sequence[Guideline],
    ) -> _TrainedFrame:
        requested_ids = {g.id for g in guidelines}

        agent_frame = self._frames_by_agent.get(agent_id)
        if agent_frame is not None and requested_ids <= set(agent_frame.by_guideline):
            return agent_frame

        key = self._frame_key(agent_id, embedder, guidelines)
        if frame := self._frame_cache.get(key):
            self._frame_cache.move_to_end(key)
            return frame

        frame = await self._train_frame(embedder, guidelines, None)
        self._cache_frame(key, frame)
        return frame

    async def _train_frame(
        self,
        embedder: Embedder,
        guidelines: Sequence[Guideline],
        progress_report: Optional[ProgressReport],
    ) -> _TrainedFrame:
        ordered = sorted(guidelines, key=lambda g: str(g.id))

        texts: list[str] = []
        owner: list[int] = []
        is_pin: list[bool] = []
        for index, guideline in enumerate(ordered):
            for text, pinned in self._training_items(guideline):
                texts.append(text)
                owner.append(index)
                is_pin.append(pinned)

        vectors = np.asarray(await self._embed_many(embedder, texts), dtype=np.float64)
        centroid = vectors.mean(axis=0)

        centered_units, centered_valid = self._row_units(vectors - centroid)
        raw_units, raw_valid = self._row_units(vectors)
        owner_array = np.asarray(owner)
        pin_array = np.asarray(is_pin)

        if progress_report:
            await progress_report.stretch(len(ordered))

        by_guideline: dict[GuidelineId, _TrainedPolicy] = {}
        for index, guideline in enumerate(ordered):
            is_own = owner_array == index
            positives = centered_units[is_own & centered_valid]
            negatives = self._sample(centered_units[(~is_own) & centered_valid])

            policy = self._train_policy(positives, negatives)
            pin_exemplars = tuple(raw_units[i] for i in np.where(is_own & pin_array & raw_valid)[0])

            by_guideline[guideline.id] = _TrainedPolicy(
                model=policy[0],
                threshold=policy[1],
                pin_exemplars=pin_exemplars,
            )

            if progress_report:
                await progress_report.increment(1)

        return _TrainedFrame(centroid=centroid, by_guideline=by_guideline)

    def _train_policy(
        self,
        positives: npt.NDArray[np.float64],
        negatives: npt.NDArray[np.float64],
    ) -> tuple[_LogisticModel, float]:
        if positives.shape[0] == 0 or negatives.shape[0] == 0:
            # Degenerate (e.g. content collapses onto the centroid); never fire.
            dimensions = positives.shape[1] if positives.ndim == 2 and positives.shape[1] else 1
            return _LogisticModel(weights=np.zeros(dimensions), bias=0.0), float("inf")

        features = np.vstack([positives, negatives])
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        model = _LogisticModel.fit(features, labels, C=self._logistic_C, class_weight="balanced")
        threshold = float(
            np.percentile(model.decision(negatives), self._recall_threshold_percentile)
        )
        return model, threshold

    def _sample(self, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if rows.shape[0] <= self._max_negatives:
            return rows
        # Deterministic, evenly-spaced subset — keeps retrain reproducible/cacheable.
        indices = np.unique(
            np.linspace(0, rows.shape[0] - 1, self._max_negatives).round().astype(int)
        )
        return np.asarray(rows[indices], dtype=np.float64)

    def _max_pin_similarity(
        self,
        query_units: npt.NDArray[np.float64],
        pin_exemplars: tuple[npt.NDArray[np.float64], ...],
    ) -> float | None:
        if not pin_exemplars or query_units.shape[0] == 0:
            return None

        # query_units and exemplars are unit vectors, so the dot is cosine.
        return max(float((query_units @ exemplar).max()) for exemplar in pin_exemplars)

    def _calculate_displayed_score(
        self,
        classifier_margin: float,
        pin_similarity: float | None,
    ) -> float:
        classifier_score = self._sigmoid(classifier_margin)

        if pin_similarity is None or pin_similarity <= self._pin_match_epsilon:
            return classifier_score

        pin_score = 0.5 + 0.5 * (
            (pin_similarity - self._pin_match_epsilon)
            / max(1.0 - self._pin_match_epsilon, _EPSILON)
        )

        return min(1.0, max(classifier_score, pin_score))

    def _sigmoid(self, x: float) -> float:
        if x >= 0.0:
            z = exp(-x)
            return 1.0 / (1.0 + z)

        z = exp(x)
        return z / (1.0 + z)

    def _training_items(self, guideline: Guideline) -> list[tuple[str, bool]]:
        items: list[tuple[str, bool]] = [(self._guideline_embedding_content(guideline), False)]
        for signal in guideline.signals:
            if signal.startswith(self.PIN_PREFIX):
                items.append((signal[len(self.PIN_PREFIX) :].strip(), True))
            else:
                items.append((signal, False))
        return items

    def _row_units(
        self,
        rows: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        valid = norms.reshape(-1) > _EPSILON
        safe = np.where(norms > _EPSILON, norms, 1.0)
        return rows / safe, valid

    def _frame_key(
        self,
        agent_id: AgentId,
        embedder: Embedder,
        guidelines: Sequence[Guideline],
    ) -> tuple[AgentId, str, tuple[tuple[str, tuple[str, ...]], ...]]:
        ordered = sorted(guidelines, key=lambda g: str(g.id))
        return (
            agent_id,
            embedder.id,
            tuple(
                (
                    str(g.id),
                    tuple(xxh3_checksum(content) for content in self._list_guideline_contents(g)),
                )
                for g in ordered
            ),
        )

    def _cache_frame(
        self,
        key: tuple[AgentId, str, tuple[tuple[str, tuple[str, ...]], ...]],
        frame: _TrainedFrame,
    ) -> None:
        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self._MAX_FRAME_CACHE_SIZE:
            self._frame_cache.popitem(last=False)

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
        # Cumulative conversation plus each individual user turn. Scoring max-pools
        # over these (see _do_recall), which gives the recaller stickiness: a policy
        # that was in play at any earlier turn stays relevant for the conversation.
        queries = [self._build_cumulative_query(context), *self._build_user_turn_queries(context)]
        return list(dict.fromkeys(query for query in queries if query))

    def _build_cumulative_query(self, context: EngineContext) -> str:
        if not context.interaction.messages and not context.state.session_summary:
            return ""

        lines: list[str] = []

        if context.state.session_summary:
            lines.append(f"Session summary: {context.state.session_summary}")

        lines.extend(f"{m.source}: {m.content}" for m in context.interaction.messages)

        return "\n".join(lines)

    def _build_user_turn_queries(self, context: EngineContext) -> list[str]:
        return [
            f"{message.source}: {message.content}"
            for message in context.interaction.messages
            if message.source == EventSource.CUSTOMER
        ]

    def _list_guideline_contents(self, guideline: Guideline) -> list[str]:
        return [self._guideline_embedding_content(guideline), *guideline.signals]

    def _guideline_embedding_content(self, guideline: Guideline) -> str:
        content = guideline.content

        title = (guideline.title or "").strip()
        condition = (content.condition or "").strip()
        action = (content.action or "").strip()
        description = (content.description or "").strip()

        sections: list[str] = []

        if title:
            sections.append(f"# {title}")

        if condition and action:
            sections.append(f"When {condition}, then {action}")
        elif condition:
            sections.append(f"Condition: {condition}")
        elif action:
            sections.append(f"Action: {action}")

        if description:
            sections.append(description)

        return "\n\n".join(sections)

    def _as_tuple(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in vector)
