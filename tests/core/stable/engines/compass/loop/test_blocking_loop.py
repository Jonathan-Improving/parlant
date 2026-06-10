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

from typing import Any, cast

from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.compass.loop.base_loop import _LoopState
from parlant.core.engines.compass.loop.blocking_loop import BlockingLoop
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    Role,
    StepCompleted,
    StepResult,
    TextDelta,
    TextPart,
    Usage,
)
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.guideline_matching.utils import create_engine_context


def _make_blocking_loop() -> BlockingLoop:
    # _update_message only touches the session event emitter and the loop state,
    # so the heavier collaborators aren't exercised here.
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)

    return BlockingLoop(
        logger=logger,
        tracer=tracer,
        meter=cast(Any, None),
        optimization_policy=cast(Any, None),
        react=cast(Any, None),
        tool_runner=cast(Any, None),
        hooks=EngineHooks(),
    )


async def test_that_blocking_loop_emits_a_single_complete_message_event_without_chunks() -> None:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    loop = _make_blocking_loop()
    state = _LoopState()

    result = StepResult(
        message=Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello there!")]),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )

    # Block mode still receives the provider's text deltas; it just doesn't emit
    # them incrementally — the whole message is emitted once on step completion.
    events = [TextDelta(text="Hello "), TextDelta(text="there!"), StepCompleted(result=result)]
    for event in events:
        await loop._on_new_event(state, event)
        await loop._update_message(context, state)

    emitter = cast(EventBuffer, context.session_event_emitter)
    message_events = [e for e in emitter.events if e.kind == EventKind.MESSAGE]

    # Exactly one message event, carrying the full text and NO `chunks` key (so
    # consumers render it as a complete, non-streamed message).
    assert len(message_events) == 1
    data = cast(dict[str, Any], message_events[0].data)
    assert data["message"] == "Hello there!"
    assert "chunks" not in data

    # A completed message with no tool calls ends the loop.
    assert context.state.prepared_to_respond is True
