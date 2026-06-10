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
from collections.abc import Mapping
import json
import os

from parlant.core.common import JSONSerializable
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.entity_cq import EntityQueries
from parlant.core.loggers import Logger
from parlant.core.tools import ToolContext, ToolId, ToolResult

# A tool call is given this long (seconds) to complete before it's abandoned and
# reported as an error. Overridable via the PARLANT_TOOL_TIMEOUT env var.
DEFAULT_TOOL_TIMEOUT = 300.0


class ToolRunner:
    """Runs a single tool against its service. Failures (including timeouts) are
    captured into an error ToolResult rather than raised, so the loop can feed them
    back to the model like any other result."""

    def __init__(self, logger: Logger, entity_queries: EntityQueries) -> None:
        self._logger = logger
        self._entity_queries = entity_queries

    async def run_tool(
        self,
        context: EngineContext,
        tool: ToolId,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        tool_context = ToolContext(
            agent_id=context.agent.id,
            session_id=context.session.id,
            customer_id=context.customer.id,
        )

        timeout = self._resolve_timeout()

        try:
            self._logger.debug(
                f"Running tool {tool.to_string()} with arguments {json.dumps(arguments, indent=2)}"
            )

            service = await self._entity_queries.read_tool_service(tool.service_name)

            result = await asyncio.wait_for(
                service.call_tool(tool.tool_name, tool_context, arguments),
                timeout=timeout,
            )

            return result
        except asyncio.TimeoutError:
            self._logger.error(f"Tool call timed out after {timeout}s ({tool.to_string()})")
            return ToolResult(
                data="Tool call timed out",
                metadata={"error_details": f"Tool call timed out after {timeout} seconds"},
            )
        except Exception as e:
            self._logger.error(f"Tool call failed ({tool.to_string()}): {e}")
            return ToolResult(data="Tool call error", metadata={"error_details": str(e)})

    def _resolve_timeout(self) -> float:
        """Per-call tool timeout in seconds, from PARLANT_TOOL_TIMEOUT, falling back
        to the default (and on a malformed value, rather than breaking the call)."""
        raw = os.environ.get("PARLANT_TOOL_TIMEOUT")
        if not raw:
            return DEFAULT_TOOL_TIMEOUT
        try:
            return float(raw)
        except ValueError:
            self._logger.warning(
                f"Invalid PARLANT_TOOL_TIMEOUT={raw!r}; using default {DEFAULT_TOOL_TIMEOUT}s"
            )
            return DEFAULT_TOOL_TIMEOUT
