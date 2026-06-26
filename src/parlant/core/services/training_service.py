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

"""Background training of the guideline recaller's per-policy discriminants.

A training run re-derives every policy's discriminant from the current inventory
(see :meth:`GuidelineRecaller.retrain`). It is re-runnable, so jobs are tracked
in memory rather than persisted: the SDK drives a progress bar off the same
:class:`ProgressReport`, and the API exposes a job whose progress is polled via
``GET /train/{job_id}``.
"""

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from parlant.core.agents import AgentStore
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import ItemNotFoundError, UniqueId, generate_id
from parlant.core.engines.compass.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.services.indexing.common import ProgressReport


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    id: UniqueId
    status: TrainingStatus
    percentage: float
    error: Optional[str] = None


class TrainingService:
    def __init__(
        self,
        recaller: GuidelineRecaller,
        agent_store: AgentStore,
        entity_queries: EntityQueries,
        background_task_service: BackgroundTaskService,
        logger: Logger,
    ) -> None:
        self._recaller = recaller
        self._agent_store = agent_store
        self._entity_queries = entity_queries
        self._background_task_service = background_task_service
        self._logger = logger
        self._jobs: dict[UniqueId, TrainingJob] = {}

    async def create_training_task(self) -> UniqueId:
        job_id = generate_id()
        self._jobs[job_id] = TrainingJob(
            id=job_id,
            status=TrainingStatus.PENDING,
            percentage=0.0,
        )
        await self._background_task_service.start(self._run(job_id), tag=f"train({job_id})")
        return job_id

    async def read_training_job(self, job_id: UniqueId) -> TrainingJob:
        if job_id not in self._jobs:
            raise ItemNotFoundError(job_id, "Training job not found")
        return self._jobs[job_id]

    async def train(self, progress_report: Optional[ProgressReport] = None) -> None:
        """Train one discriminant frame per agent, over that agent's own guideline
        space. Each agent's policies only compete within that agent, so frames are
        never shared. The SDK calls this directly on startup (driving its own progress
        bar); the API path goes through a job. A shared ``progress_report`` accumulates
        across agents."""
        for agent in await self._agent_store.list_agents():
            guidelines = await self._entity_queries.find_guidelines_for_context(agent.id, [])
            await self._recaller.retrain(agent.id, guidelines, progress_report)

    async def _run(self, job_id: UniqueId) -> None:
        job = self._jobs[job_id]
        job.status = TrainingStatus.RUNNING

        async def on_progress(percentage: float) -> None:
            job.percentage = percentage

        try:
            await self.train(ProgressReport(on_progress))
            job.status = TrainingStatus.COMPLETED
            job.percentage = 100.0
        except Exception as exc:
            self._logger.error(
                f"Training job '{job_id}' failed: {exc}\n\n"
                f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            job.status = TrainingStatus.FAILED
            job.error = str(exc)
