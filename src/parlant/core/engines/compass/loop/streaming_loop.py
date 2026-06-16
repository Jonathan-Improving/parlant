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
from parlant.core.nlp.react import StepCompleted, StreamEvent, TextDelta, ToolCallStarted
from parlant.core.sessions import MessageEventData, Participant, StatusEventData


class StreamingLoop(BaseLoop):
    """Emits the assistant's message incrementally: each text delta extends the
    message event's growing buffer + `chunks`, so consumers can render it as it's
    produced. The final step-completion update null-terminates `chunks`."""

    async def _update_message(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case TextDelta(text=text):
                if state.message_handle is None:  # First message chunk
                    if not self._can_emit_tool_preamble(context, state):
                        if state.message_buffer is None:
                            state.message_buffer = StringIO()
                            state.message_chunks = []

                        state.message_buffer.write(text)
                        state.message_chunks.append(text)
                        return

                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )

                    state.message_buffer = StringIO()
                    state.message_buffer.write(text)
                    state.message_chunks = [text]

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
                    state.user_visible_message_emitted = True
                else:  # Subsequent message chunk
                    assert state.message_buffer is not None

                    state.message_buffer.write(text)
                    state.message_chunks.append(text)

                    state.message_handle = await state.message_handle.update(
                        MessageEventData(
                            message=state.message_buffer.getvalue(),
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=state.message_chunks,
                        ),
                    )
            case ToolCallStarted() if state.message_handle is not None:
                # Text -> tool transition within a step: finalize the in-flight message
                # as its own bubble so post-tool text starts a fresh one (and the tool
                # status follows the message). The buffer is what was actually shown.
                buffered = state.message_buffer.getvalue() if state.message_buffer else ""

                await state.message_handle.update(
                    MessageEventData(
                        message=buffered,
                        participant=Participant(
                            id=context.agent.id, display_name=context.agent.name
                        ),
                        chunks=[*state.message_chunks, None],
                    )
                )

                state.emitted_message_len += len(buffered)
                state.message_buffer = None
                state.message_chunks = []
                state.message_handle = None
            case ToolCallStarted() if state.message_buffer is not None:
                state.suppress_current_tool_message_text = bool(state.message_buffer.getvalue())
                state.message_buffer = None
                state.message_chunks = []
            case StepCompleted(result=result):
                # Emit the authoritative remainder: everything this step's message holds
                # beyond what interrupt-splits already emitted. Anchoring on
                # `result.message.text` (not the raw buffer) keeps the final segment
                # correct even if a provider delivered tail text outside the deltas.
                remaining = result.message.text[state.emitted_message_len :]

                if state.message_handle is not None:
                    await state.message_handle.update(
                        MessageEventData(
                            message=remaining,
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=[*state.message_chunks, None],
                        )
                    )

                    state.message_buffer = None
                    state.message_chunks = []
                    state.message_handle = None
                    state.emitted_message_len = 0

                    await self._complete_message_step(context, result)
                elif result.needs_tools and not self._can_emit_tool_preamble(context, state):
                    state.suppress_current_tool_message_text = bool(result.message.text)
                    state.message_buffer = None
                    state.message_chunks = []
                    state.emitted_message_len = 0
                elif remaining:
                    # Text arrived only in the final message (no deltas streamed) — emit
                    # it once as a complete, terminated message.
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=remaining,
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                            chunks=[remaining, None],
                        ),
                    )
                    state.user_visible_message_emitted = True

                    state.emitted_message_len = 0

                    await self._complete_message_step(context, result)
                else:
                    state.emitted_message_len = 0
