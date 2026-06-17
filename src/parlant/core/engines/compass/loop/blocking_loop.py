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


class BlockingLoop(BaseLoop):
    """Emits the assistant's message once, complete, when the step finishes — no
    incremental updates and no `chunks`, so consumers render it as a single,
    non-streamed message. Text deltas still arrive from the provider; they're just
    not surfaced until the message is whole."""

    async def _update_message(
        self,
        context: EngineContext,
        state: _LoopState,
        event: StreamEvent,
    ) -> None:
        match event:
            case TextDelta(text=text):
                # Don't emit deltas in block mode, but DO accumulate the text: it's
                # needed to split the message on a text -> tool transition and to know
                # what was already shown. Show a typing indicator on the first text of
                # each segment.
                if state.message_buffer is None:  # First text since the last emit
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )
                    state.message_buffer = StringIO()

                state.message_buffer.write(text)
            case ToolCallStarted() if state.message_buffer is not None:
                # Text -> tool transition within a step: emit the text so far as its own
                # complete message before the tool status, so post-tool text isn't glued
                # onto it.
                buffered = state.message_buffer.getvalue()

                if not self._additional_preamble_is_needed(state):
                    state.allowed_tool_message_text_len = 0
                    state.suppress_current_tool_message_text = bool(buffered)
                    state.message_buffer = None
                    return

                preamble = self._tool_preamble_text(buffered)

                if preamble:
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=preamble,
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                        ),
                    )
                    self._mark_user_visible_message_emitted(state)
                    state.emitted_message_len += len(preamble)
                    state.allowed_tool_message_text_len = state.emitted_message_len

                state.message_buffer = None
            case StepCompleted(result=result):
                # Emit the authoritative remainder once, complete and without `chunks`:
                # everything the step's message holds beyond what interrupt-splits
                # already emitted. Anchoring on `result.message.text` (not the buffer)
                # keeps it correct even if a provider delivered text outside the deltas.
                remaining = result.message.text[state.emitted_message_len :]

                if result.needs_tools and state.allowed_tool_message_text_len is not None:
                    state.message_buffer = None
                    state.emitted_message_len = 0
                    return

                if result.needs_tools and not self._additional_preamble_is_needed(state):
                    state.allowed_tool_message_text_len = 0
                    state.suppress_current_tool_message_text = bool(result.message.text)
                    state.message_buffer = None
                    state.emitted_message_len = 0
                    return

                preamble = self._tool_preamble_text(remaining) if result.needs_tools else remaining

                if preamble:
                    await context.session_event_emitter.emit_message_event(
                        trace_id=context.tracer.trace_id,
                        data=MessageEventData(
                            message=preamble,
                            participant=Participant(
                                id=context.agent.id, display_name=context.agent.name
                            ),
                        ),
                    )
                    self._mark_user_visible_message_emitted(state)
                    if result.needs_tools:
                        state.allowed_tool_message_text_len = len(preamble)

                state.message_buffer = None
                state.emitted_message_len = 0

                if result.message.text:
                    await self._complete_message_step(context, result)
