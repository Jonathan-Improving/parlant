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

    async def _surface_message_event(
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
                if state.message.buffer is None:  # First text since the last emit
                    await context.session_event_emitter.emit_status_event(
                        trace_id=context.tracer.trace_id,
                        data=StatusEventData(status="typing"),
                    )
                    state.message.buffer = StringIO()

                state.message.buffer.write(text)
            case ToolCallStarted() if state.message.buffer is not None:
                # Text -> tool transition within a step: emit the text so far as its own
                # complete message before the tool status, so post-tool text isn't glued
                # onto it.
                buffered = state.message.buffer.getvalue()

                if not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.buffer = None
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
                    state.message.emitted_len += len(preamble)
                    state.message.keep_tool_text_prefix(state.message.emitted_len)

                state.message.buffer = None
            case StepCompleted(result=result):
                # Emit the authoritative remainder once, complete and without `chunks`:
                # everything the step's message holds beyond what interrupt-splits
                # already emitted. Anchoring on `result.message.text` (not the buffer)
                # keeps it correct even if a provider delivered text outside the deltas.
                remaining = result.message.text[state.message.emitted_len :]

                if result.needs_tools and state.message.has_custom_commit_policy():
                    state.message.buffer = None
                    state.message.emitted_len = 0
                    return

                if result.needs_tools and not self._tool_preamble_is_allowed(state):
                    state.message.suppress_tool_text()
                    state.message.buffer = None
                    state.message.emitted_len = 0
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
                        state.message.keep_tool_text_prefix(len(preamble))

                state.message.buffer = None
                state.message.emitted_len = 0

                if result.message.text:
                    await self._complete_message_step(context, result)
