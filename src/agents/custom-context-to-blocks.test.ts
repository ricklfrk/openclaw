import type { AssistantMessage } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { promoteHistoricalToolCallThinkingToBlocks } from "./custom-context-to-blocks.js";

function makeAssistantMessage(
  message: Omit<AssistantMessage, "api" | "provider" | "model" | "usage" | "stopReason"> &
    Partial<Pick<AssistantMessage, "api" | "provider" | "model" | "usage" | "stopReason">>,
): AssistantMessage {
  return {
    api: "responses",
    provider: "openai",
    model: "gpt-5",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    ...message,
  };
}

describe("promoteHistoricalToolCallThinkingToBlocks", () => {
  it("salvages a thinking-only historical tool call into a real toolCall", () => {
    const msg = makeAssistantMessage({
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: `[Historical tool call: exec]
{
  "command": "ls -la"
}`,
        },
      ],
      timestamp: Date.now(),
    });

    promoteHistoricalToolCallThinkingToBlocks(msg);

    expect(msg.content).toEqual([
      {
        type: "toolCall",
        id: expect.stringMatching(/^historical-tool-call-\d+$/),
        name: "exec",
        arguments: { command: "ls -la" },
      },
    ]);
    expect(msg.stopReason).toBe("tool_call");
  });

  it("salvages a text preamble followed by exactly one historical tool call", () => {
    const msg = makeAssistantMessage({
      role: "assistant",
      content: [
        { type: "text", text: "I will run this now." },
        {
          type: "thinking",
          thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
        },
      ],
      timestamp: Date.now(),
    });

    promoteHistoricalToolCallThinkingToBlocks(msg);

    expect(msg.content).toEqual([
      {
        type: "toolCall",
        id: expect.stringMatching(/^historical-tool-call-\d+$/),
        name: "exec",
        arguments: { command: "ls -la" },
      },
    ]);
    expect(msg.stopReason).toBe("tool_call");
  });

  it("does not salvage when visible text appears after the historical tool call", () => {
    const msg = makeAssistantMessage({
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
        },
        { type: "text", text: "Tool call described above." },
      ],
      timestamp: Date.now(),
    });

    promoteHistoricalToolCallThinkingToBlocks(msg);

    expect(msg.content).toEqual([
      {
        type: "thinking",
        thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
      },
      { type: "text", text: "Tool call described above." },
    ]);
    expect(msg.stopReason).toBe("stop");
  });

  it("does not salvage when a real toolCall already exists", () => {
    const msg = makeAssistantMessage({
      role: "assistant",
      content: [
        { type: "toolCall", id: "call_1", name: "read", arguments: { file_path: "x" } },
        {
          type: "thinking",
          thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
        },
      ],
      timestamp: Date.now(),
    });

    promoteHistoricalToolCallThinkingToBlocks(msg);

    expect(msg.content).toEqual([
      { type: "toolCall", id: "call_1", name: "read", arguments: { file_path: "x" } },
      {
        type: "thinking",
        thinking: `[Historical tool call: exec]
{"command":"ls -la"}`,
      },
    ]);
    expect(msg.stopReason).toBe("stop");
  });
});
