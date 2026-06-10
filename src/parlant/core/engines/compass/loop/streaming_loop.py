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


class StreamingLoop(BaseLoop):
    """Emits the assistant's message incrementally: each text delta extends the
    message event's growing buffer + `chunks`, so consumers can render it as it's
    produced. The final step-completion update null-terminates `chunks`."""

    async def _update_message(self, context: EngineContext, state: _LoopState) -> None:
        match state.current_event:
            case TextDelta():
                if state.message_handle is None:  # First message chunk
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )

                    state.message_buffer = StringIO()
                    state.message_buffer.write(state.current_event.text)
                    state.message_chunks = [state.current_event.text]

                    state.message_handle = await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=state.message_buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message_chunks,
                        ),
                    )
                else:  # Subsequent message chunk
                    assert state.message_buffer is not None

                    state.message_buffer.write(state.current_event.text)
                    state.message_chunks.append(state.current_event.text)

                    state.message_handle = await state.message_handle.update(
                        MessageEventData(
                            message=state.message_buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message_chunks,
                        ),
                    )
            case StepCompleted(result=result) if state.message_handle is not None:
                state.message_handle = await state.message_handle.update(
                    MessageEventData(
                        message=result.message.text,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[*state.message_chunks, None],
                    )
                )

                state.message_buffer = None
                state.message_chunks = []
                state.message_handle = None

                # A message was produced on this step's completion (it was already
                # streamed/emitted above).
                await self._complete_message_step(context, result)
