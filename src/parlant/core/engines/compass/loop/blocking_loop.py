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

from io import StringIO

from parlant.core.engines.compass.loop.base_loop import BaseLoop, _LoopState
from parlant.core.engines.compass.response_state import EngineContext
from parlant.core.nlp.react import StepCompleted, TextDelta
from parlant.core.sessions import MessageEventData, Participant, StatusEventData


class BlockingLoop(BaseLoop):
    """Emits the assistant's message once, complete, when the step finishes — no
    incremental updates and no `chunks`, so consumers render it as a single,
    non-streamed message. Text deltas still arrive from the provider; they're just
    not surfaced until the message is whole."""

    async def _update_message(self, context: EngineContext, state: _LoopState) -> None:
        match state.current_event:
            case TextDelta():
                # Don't emit deltas in block mode; only note that a message is being
                # produced this step (gating the completion branch below) and show a
                # typing indicator while it's generated.
                if state.message_buffer is None:  # First text of this step
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )
                    state.message_buffer = StringIO()
            case StepCompleted(result=result) if state.message_buffer is not None:
                # The whole message is available now; emit it once, without `chunks`.
                await context.session_event_emitter.emit_message_event(
                    trace_id=context.tracer.trace_id,
                    data=MessageEventData(
                        message=result.message.text,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                    ),
                )

                state.message_buffer = None

                await self._complete_message_step(context, result)
