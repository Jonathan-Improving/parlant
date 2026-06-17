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

from parlant.core.agents import Effort
from parlant.core.engines.compass.response_state import EngineContext


def get_dynamic_reasoning_effort_for_matching(context: EngineContext) -> str:
    """Map the context's dynamic effort level to a model ``reasoning_effort`` hint."""
    effort = context.state.dynamic_effort_level

    # Note that low effort has higher reasoning than medium,
    # because the assumption is that medium effort comes
    # with added ARQs in its output anyway.

    match effort:
        case Effort.MIN:
            return "minimal"
        case Effort.LOW:
            return "low"
        case Effort.MEDIUM:
            return "minimal"
        case Effort.HIGH:
            return "low"
        case Effort.MAX:
            return "medium"
