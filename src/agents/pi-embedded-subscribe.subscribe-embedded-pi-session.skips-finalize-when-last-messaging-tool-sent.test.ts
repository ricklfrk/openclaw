import type { AssistantMessage } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { createStubSessionHarness } from "./pi-embedded-subscribe.e2e-harness.js";
import { subscribeEmbeddedPiSession } from "./pi-embedded-subscribe.js";

async function emitMessageToolLifecycle(params: {
  emit: (evt: unknown) => void;
  toolCallId: string;
  message: string;
  isError?: boolean;
}) {
  params.emit({
    type: "tool_execution_start",
    toolName: "message",
    toolCallId: params.toolCallId,
    args: { action: "send", to: "+1555", message: params.message },
  });
  await Promise.resolve();
  params.emit({
    type: "tool_execution_end",
    toolName: "message",
    toolCallId: params.toolCallId,
    isError: params.isError ?? false,
    result: params.isError ? { details: { status: "error" } } : "ok",
  });
}

function emitAssistantMessageStartAndEnd(emit: (evt: unknown) => void, text: string) {
  const msg = {
    role: "assistant",
    content: [{ type: "text", text }],
  } as AssistantMessage;
  emit({ type: "message_start", message: msg });
  emit({ type: "message_end", message: msg });
}

describe("subscribeEmbeddedPiSession – finalizeAssistantTexts messaging tool dedup", () => {
  it("excludes text from assistantTexts when it duplicates the last messaging tool send", async () => {
    const { session, emit } = createStubSessionHarness();
    const onBlockReply = vi.fn();
    const subscription = subscribeEmbeddedPiSession({
      session,
      runId: "run",
      onBlockReply,
      blockReplyBreak: "message_end",
    });

    const text = "哥哥～下午好！剛剛那份計畫看完了嗎？如果沒問題的話，我們隨時可以開工喔！";
    await emitMessageToolLifecycle({ emit, toolCallId: "t1", message: text });
    emitAssistantMessageStartAndEnd(emit, text);

    expect(subscription.assistantTexts).toEqual([]);
    expect(onBlockReply).not.toHaveBeenCalled();
  });

  it("keeps text in assistantTexts when messaging tool send failed", async () => {
    const { session, emit } = createStubSessionHarness();
    const onBlockReply = vi.fn();
    const subscription = subscribeEmbeddedPiSession({
      session,
      runId: "run",
      onBlockReply,
      blockReplyBreak: "message_end",
    });

    const text = "Here is your answer, please check it out.";
    await emitMessageToolLifecycle({ emit, toolCallId: "t1", message: text, isError: true });
    emitAssistantMessageStartAndEnd(emit, text);

    expect(subscription.assistantTexts).toEqual([text]);
    expect(onBlockReply).toHaveBeenCalledTimes(1);
  });

  it("keeps text in assistantTexts when it differs from the last messaging tool send", async () => {
    const { session, emit } = createStubSessionHarness();
    const subscription = subscribeEmbeddedPiSession({
      session,
      runId: "run",
      blockReplyBreak: "message_end",
    });

    await emitMessageToolLifecycle({
      emit,
      toolCallId: "t1",
      message: "Sent this to the channel.",
    });
    emitAssistantMessageStartAndEnd(emit, "A completely different response to the user.");

    expect(subscription.assistantTexts).toEqual(["A completely different response to the user."]);
  });

  it("excludes text without onBlockReply (payload-only path)", async () => {
    const { session, emit } = createStubSessionHarness();
    const subscription = subscribeEmbeddedPiSession({
      session,
      runId: "run",
    });

    const text = "This was already sent via the messaging tool call.";
    await emitMessageToolLifecycle({ emit, toolCallId: "t1", message: text });
    emitAssistantMessageStartAndEnd(emit, text);

    expect(subscription.assistantTexts).toEqual([]);
  });
});
