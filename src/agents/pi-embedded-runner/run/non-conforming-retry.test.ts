import type { AssistantMessage } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { NON_CONFORMING_FORMAT_HINT, checkNonConformingOutput } from "./non-conforming-retry.js";

function makeAssistant(
  overrides: Partial<AssistantMessage> & { text?: string; thinking?: string } = {},
): AssistantMessage {
  const { text, thinking, ...rest } = overrides;
  const content: Array<Record<string, unknown>> = [];
  if (thinking) {
    content.push({ type: "thinking", thinking });
  }
  if (text) {
    content.push({ type: "text", text });
  }
  return {
    role: "assistant",
    content,
    stopReason: "stop",
    timestamp: Date.now(),
    ...rest,
  } as AssistantMessage;
}

const baseParams = {
  enforceFinalTag: true,
  formatHintInjected: false,
  aborted: false,
  timedOut: false,
  assistantTexts: [] as string[],
  didSendViaMessagingTool: false,
};

describe("checkNonConformingOutput", () => {
  describe("skipProfile driven by formatHintInjected", () => {
    it("first failure (no hint): skipProfile=false for thinking-only", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: false,
        lastAssistant: makeAssistant({ thinking: "Reasoning..." }),
      });
      expect(result).toEqual({ action: "fail", skipProfile: false });
    });

    it("after hint injected: skipProfile=true for thinking-only", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant({ thinking: "Reasoning..." }),
      });
      expect(result).toEqual({ action: "fail", skipProfile: true });
    });

    it("first failure (no hint): skipProfile=false for empty", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: false,
        lastAssistant: makeAssistant(),
      });
      expect(result).toEqual({ action: "fail", skipProfile: false });
    });

    it("after hint injected: skipProfile=true for empty", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant(),
      });
      expect(result).toEqual({ action: "fail", skipProfile: true });
    });

    it("first failure (no hint): skipProfile=false for non-conforming text", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: false,
        lastAssistant: makeAssistant({
          text: "Here is a long response without final tags.",
        }),
      });
      expect(result).toEqual({
        action: "fail",
        skipProfile: false,
        fallbackText: "Here is a long response without final tags.",
      });
    });

    it("after hint injected: skipProfile=true for non-conforming text", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant({
          text: "Here is a long response without final tags.",
        }),
      });
      expect(result).toEqual({
        action: "fail",
        skipProfile: true,
        fallbackText: "Here is a long response without final tags.",
      });
    });

    it("captures stripped fallback text from non-conforming text", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant({
          text: "<think>reasoning</think> more</think>■ reply here with enough length for fallback",
        }),
      });
      expect(result).toEqual({
        action: "fail",
        skipProfile: true,
        fallbackText: "more■ reply here with enough length for fallback",
      });
    });

    it("strips <final> tags from deferred fallback text", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant({
          text: "<think>reasoning</think><final>visible reply with enough length for fallback</final>",
        }),
      });
      expect(result).toEqual({
        action: "fail",
        skipProfile: true,
        fallbackText: "visible reply with enough length for fallback",
      });
    });
  });

  describe("thinking-only / empty keep failing (no fallback text)", () => {
    it("thinking-only at high retry count still returns fail", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant({ thinking: "Still thinking..." }),
      });
      expect(result).toEqual({ action: "fail", skipProfile: true });
    });

    it("empty at high retry count still returns fail", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        formatHintInjected: true,
        lastAssistant: makeAssistant(),
      });
      expect(result).toEqual({ action: "fail", skipProfile: true });
    });
  });

  describe("edge cases for text stripping", () => {
    it("unclosed <think> block → fail", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        lastAssistant: makeAssistant({
          text: "<think>reasoning that never closes and keeps going on and on",
        }),
      });
      expect(result).toEqual({ action: "fail", skipProfile: false });
    });

    it("text-block-with-only-think-content → fail", () => {
      const result = checkNonConformingOutput({
        ...baseParams,
        lastAssistant: makeAssistant({
          text: "<think>all content is inside thinking tags</think>",
        }),
      });
      expect(result).toEqual({ action: "fail", skipProfile: false });
    });
  });

  describe("skip conditions", () => {
    it("returns null when enforceFinalTag is false", () => {
      expect(
        checkNonConformingOutput({
          ...baseParams,
          enforceFinalTag: false,
          lastAssistant: makeAssistant(),
        }),
      ).toBeNull();
    });

    it("returns null when aborted", () => {
      expect(
        checkNonConformingOutput({ ...baseParams, aborted: true, lastAssistant: makeAssistant() }),
      ).toBeNull();
    });

    it("returns null when assistantTexts is non-empty", () => {
      expect(
        checkNonConformingOutput({
          ...baseParams,
          assistantTexts: ["Hello"],
          lastAssistant: makeAssistant(),
        }),
      ).toBeNull();
    });

    it("returns null when stopReason is toolUse", () => {
      expect(
        checkNonConformingOutput({
          ...baseParams,
          lastAssistant: makeAssistant({ stopReason: "toolUse" }),
        }),
      ).toBeNull();
    });

    it("returns null for short text", () => {
      expect(
        checkNonConformingOutput({ ...baseParams, lastAssistant: makeAssistant({ text: "ok" }) }),
      ).toBeNull();
    });

    it("returns null for silent reply", () => {
      expect(
        checkNonConformingOutput({
          ...baseParams,
          lastAssistant: makeAssistant({ text: "NO_REPLY" }),
        }),
      ).toBeNull();
    });
  });

  describe("NON_CONFORMING_FORMAT_HINT", () => {
    it("contains tool call and reply format guidance", () => {
      expect(NON_CONFORMING_FORMAT_HINT).toContain("functionCall");
      expect(NON_CONFORMING_FORMAT_HINT).toContain("<final>");
      expect(NON_CONFORMING_FORMAT_HINT).toContain("FORBIDDEN");
    });
  });
});
