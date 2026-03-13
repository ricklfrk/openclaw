import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it, vi } from "vitest";
import { createInlineCodeState } from "../markdown/code-spans.js";
import {
  handleMessageEnd,
  resolveSilentReplyFallbackText,
} from "./pi-embedded-subscribe.handlers.messages.js";
import type { EmbeddedPiSubscribeContext } from "./pi-embedded-subscribe.handlers.types.js";

describe("resolveSilentReplyFallbackText", () => {
  it("replaces NO_REPLY with latest messaging tool text when available", () => {
    expect(
      resolveSilentReplyFallbackText({
        text: "NO_REPLY",
        messagingToolSentTexts: ["first", "final delivered text"],
      }),
    ).toBe("final delivered text");
  });

  it("keeps original text when response is not NO_REPLY", () => {
    expect(
      resolveSilentReplyFallbackText({
        text: "normal assistant reply",
        messagingToolSentTexts: ["final delivered text"],
      }),
    ).toBe("normal assistant reply");
  });

  it("keeps NO_REPLY when there is no messaging tool text to mirror", () => {
    expect(
      resolveSilentReplyFallbackText({
        text: "NO_REPLY",
        messagingToolSentTexts: [],
      }),
    ).toBe("NO_REPLY");
  });
});

describe("handleMessageEnd", () => {
  function createContext(): EmbeddedPiSubscribeContext {
    return {
      params: {
        runId: "run-1",
        session: { id: "session-1" },
        enforceFinalTag: false,
      } as never,
      state: {
        assistantTexts: [],
        toolMetas: [],
        toolMetaById: new Map(),
        toolSummaryById: new Set(),
        blockReplyBreak: "message_end",
        reasoningMode: "off",
        includeReasoning: false,
        shouldEmitPartialReplies: false,
        streamReasoning: false,
        deltaBuffer: "",
        blockBuffer: "",
        blockState: { thinking: false, final: false, inlineCode: createInlineCodeState() },
        partialBlockState: { thinking: false, final: false, inlineCode: createInlineCodeState() },
        emittedAssistantUpdate: false,
        reasoningStreamOpen: false,
        assistantMessageIndex: 0,
        lastAssistantTextMessageIndex: -1,
        assistantTextBaseline: 0,
        suppressBlockChunks: false,
        compactionInFlight: false,
        pendingCompactionRetry: 0,
        compactionRetryPromise: null,
        unsubscribed: false,
        messagingToolSentTexts: [],
        messagingToolSentTextsNormalized: [],
        messagingToolSentTargets: [],
        messagingToolSentMediaUrls: [],
        pendingMessagingTexts: new Map(),
        pendingMessagingTargets: new Map(),
        pendingMessagingMediaUrls: new Map(),
        successfulCronAdds: 0,
      } as never,
      log: {
        debug: vi.fn(),
        warn: vi.fn(),
      },
      blockChunker: null,
      noteLastAssistant: vi.fn(),
      shouldEmitToolResult: () => false,
      shouldEmitToolOutput: () => false,
      emitToolSummary: vi.fn(),
      emitToolOutput: vi.fn(),
      stripBlockTags: (text) => text,
      emitBlockChunk: vi.fn(),
      flushBlockReplyBuffer: vi.fn(),
      emitReasoningStream: vi.fn(),
      consumeReplyDirectives: (text) => ({ text, mediaUrls: [] }) as never,
      consumePartialReplyDirectives: (text) => ({ text, mediaUrls: [] }) as never,
      resetAssistantMessageState: vi.fn(),
      resetForCompactionRetry: vi.fn(),
      finalizeAssistantTexts: vi.fn(),
      trimMessagingToolSent: vi.fn(),
      ensureCompactionPromise: vi.fn(),
      noteCompactionRetry: vi.fn(),
      resolveCompactionRetry: vi.fn(),
      maybeResolveCompactionWait: vi.fn(),
      recordAssistantUsage: vi.fn(),
      incrementCompactionCount: vi.fn(),
      getUsageTotals: vi.fn(),
      getCompactionCount: () => 0,
    };
  }

  it("does not promote historical tool call thinking during message_end", () => {
    const ctx = createContext();
    const message: AgentMessage = {
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
        },
      ],
      stopReason: "stop",
      timestamp: Date.now(),
    } as never;

    handleMessageEnd(ctx, { type: "message_end", message } as never);

    expect(message.stopReason).toBe("stop");
    expect(message.content).toEqual([
      {
        type: "thinking",
        thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
      },
    ]);
  });
});
