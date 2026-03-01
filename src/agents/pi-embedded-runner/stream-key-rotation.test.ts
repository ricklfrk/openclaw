import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessageEvent } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import type { AuthProfileStore } from "../auth-profiles/types.js";
import {
  canSafelyRotate,
  createKeyRotationState,
  wrapStreamFnWithKeyRotation,
} from "./stream-key-rotation.js";

vi.mock("../auth-profiles/usage.js", () => ({
  calculateAuthProfileCooldownMs: vi.fn().mockReturnValue(60_000),
  isProfileInCooldown: vi.fn().mockReturnValue(false),
  stampProfileLastUsed: vi.fn().mockResolvedValue(undefined),
  markAuthProfileFailure: vi.fn().mockResolvedValue(undefined),
}));

function makeAuthStore(profileIds: string[]): AuthProfileStore {
  const profiles: Record<string, { type: "api_key"; provider: string; key: string }> = {};
  for (const id of profileIds) {
    profiles[id] = { type: "api_key", provider: "lab", key: `key-${id}` };
  }
  return { version: 1, profiles };
}

function makeDoneEvent(text = "ok"): AssistantMessageEvent {
  return {
    type: "done",
    reason: "stop",
    message: {
      role: "assistant",
      content: [{ type: "text", text }],
      api: "google-generative-ai",
      provider: "lab",
      model: "gemini-3.1-pro-preview",
      usage: {
        input: 10,
        output: 5,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 15,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    },
  } as AssistantMessageEvent;
}

function makeRateLimitErrorEvent(): AssistantMessageEvent {
  return {
    type: "error",
    reason: "error",
    error: {
      role: "assistant",
      content: [],
      api: "google-generative-ai",
      provider: "lab",
      model: "gemini-3.1-pro-preview",
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "error",
      errorMessage: "429 RESOURCE_EXHAUSTED: rate limit exceeded",
      timestamp: Date.now(),
    },
  } as AssistantMessageEvent;
}

function makeNonRateLimitErrorEvent(): AssistantMessageEvent {
  return {
    type: "error",
    reason: "error",
    error: {
      role: "assistant",
      content: [],
      api: "google-generative-ai",
      provider: "lab",
      model: "gemini-3.1-pro-preview",
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "error",
      errorMessage: "401 UNAUTHENTICATED: invalid credentials",
      timestamp: Date.now(),
    },
  } as AssistantMessageEvent;
}

/** Create a mock StreamFn that returns predetermined events per call. */
function createMockStreamFn(callResponses: AssistantMessageEvent[][]): {
  streamFn: StreamFn;
  calls: Array<{ apiKey?: string }>;
} {
  const calls: Array<{ apiKey?: string }> = [];
  let callIndex = 0;
  const streamFn: StreamFn = (_model, _context, options) => {
    calls.push({ apiKey: (options as { apiKey?: string })?.apiKey });
    const events = callResponses[callIndex] ?? [makeDoneEvent()];
    callIndex++;
    const stream = createAssistantMessageEventStream();
    for (const event of events) {
      stream.push(event);
    }
    stream.end();
    return stream;
  };
  return { streamFn, calls };
}

const geminiModel = {
  api: "google-generative-ai" as const,
  provider: "lab",
  id: "gemini-3.1-pro-preview",
  input: ["text"] as string[],
  output: ["text"] as string[],
};

async function collectEvents(stream: ReturnType<StreamFn>): Promise<AssistantMessageEvent[]> {
  const resolved = await Promise.resolve(stream);
  const events: AssistantMessageEvent[] = [];
  for await (const event of resolved) {
    events.push(event);
  }
  return events;
}

describe("canSafelyRotate", () => {
  it("returns true for google-generative-ai", () => {
    expect(canSafelyRotate({ api: "google-generative-ai", id: "gemini-3" })).toBe(true);
  });

  it("returns true for google-gemini-cli", () => {
    expect(canSafelyRotate({ api: "google-gemini-cli", id: "gemini-3" })).toBe(true);
  });

  it("returns true for model id containing gemini", () => {
    expect(canSafelyRotate({ api: "openrouter", id: "google/gemini-3-pro" })).toBe(true);
  });

  it("returns false for anthropic", () => {
    expect(canSafelyRotate({ api: "anthropic-messages", id: "claude-4" })).toBe(false);
  });

  it("returns false for openai", () => {
    expect(canSafelyRotate({ api: "openai-completions", id: "gpt-5" })).toBe(false);
  });
});

describe("wrapStreamFnWithKeyRotation", () => {
  it("passes through when only 1 profile candidate", async () => {
    const { streamFn, calls } = createMockStreamFn([[makeDoneEvent()]]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1"],
      resolveApiKey: async () => "key-p1",
      authStore: store,
      rotationState: state,
    });

    const events = await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    // Passthrough: streamFn should be called directly (no wrapper interception)
    expect(calls).toHaveLength(1);
  });

  it("passes through for non-safe models (anthropic)", async () => {
    const { streamFn, calls } = createMockStreamFn([[makeDoneEvent()]]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const anthropicModel = { ...geminiModel, api: "anthropic-messages" as const, id: "claude-4" };
    const events = await collectEvents(
      wrapped(anthropicModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );
    expect(events).toHaveLength(1);
    expect(calls).toHaveLength(1);
    // Should not have overridden apiKey
    expect(calls[0].apiKey).toBeUndefined();
  });

  it("proactively rotates key on each call", async () => {
    const { streamFn, calls } = createMockStreamFn([
      [makeDoneEvent("first")],
      [makeDoneEvent("second")],
      [makeDoneEvent("third")],
    ]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2", "p3"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2", "p3"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const ctx = { systemPrompt: "", messages: [], tools: [] };

    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(calls[0].apiKey).toBe("key-p1");
    expect(state.rotationIndex).toBe(1);

    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(calls[1].apiKey).toBe("key-p2");
    expect(state.rotationIndex).toBe(2);

    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(calls[2].apiKey).toBe("key-p3");
    expect(state.rotationIndex).toBe(0); // wraps around
  });

  it("retries with next key on 429 rate-limit error event", async () => {
    const { streamFn, calls } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeDoneEvent("success from p2")],
    ]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const events = await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );

    // First call used p1 (429), second used p2 (success)
    expect(calls).toHaveLength(2);
    expect(calls[0].apiKey).toBe("key-p1");
    expect(calls[1].apiKey).toBe("key-p2");

    // Only the success events should be in output
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    expect(state.allKeysExhausted).toBe(false);
  });

  it("sets allKeysExhausted when all keys return 429", async () => {
    const { streamFn, calls } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeRateLimitErrorEvent()],
      [makeRateLimitErrorEvent()],
    ]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2", "p3"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2", "p3"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const events = await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );

    expect(calls).toHaveLength(3);
    expect(state.allKeysExhausted).toBe(true);

    // Should output an error event for the caller
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
    const errorMsg = (events[0] as { error?: { errorMessage?: string } }).error?.errorMessage ?? "";
    expect(errorMsg).toContain("exhausted");
  });

  it("does not retry on non-rate-limit errors", async () => {
    const { streamFn, calls } = createMockStreamFn([[makeNonRateLimitErrorEvent()]]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const events = await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );

    // Only 1 call: non-rate-limit error is passed through, no retry
    expect(calls).toHaveLength(1);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
    expect(state.allKeysExhausted).toBe(false);
  });

  it("sets maxRetries: 0 in options to skip internal retries", async () => {
    const receivedOptions: unknown[] = [];
    const streamFn: StreamFn = (_model, _context, options) => {
      receivedOptions.push(options);
      const stream = createAssistantMessageEventStream();
      stream.push(makeDoneEvent());
      stream.end();
      return stream;
    };
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, { maxRetries: 5 }),
    );

    expect((receivedOptions[0] as { maxRetries?: number }).maxRetries).toBe(0);
  });

  it("handles streamFn throwing synchronously on rate limit", async () => {
    let callCount = 0;
    const streamFn: StreamFn = (_model, _context, _options) => {
      callCount++;
      if (callCount === 1) {
        throw new Error("429 Too Many Requests: rate limit exceeded");
      }
      const stream = createAssistantMessageEventStream();
      stream.push(makeDoneEvent());
      stream.end();
      return stream;
    };
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const events = await collectEvents(
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {}),
    );

    expect(callCount).toBe(2);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
  });

  it("resets allKeysExhausted on each new model call", async () => {
    const { streamFn } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeRateLimitErrorEvent()],
      // Third call (second invocation): succeeds
      [makeDoneEvent()],
    ]);
    const state = createKeyRotationState();
    const store = makeAuthStore(["p1", "p2"]);

    const wrapped = wrapStreamFnWithKeyRotation({
      streamFn,
      profileCandidates: ["p1", "p2"],
      resolveApiKey: async (c) => `key-${c}`,
      authStore: store,
      rotationState: state,
    });

    const ctx = { systemPrompt: "", messages: [], tools: [] };

    // First call: both keys 429 → allKeysExhausted = true
    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(state.allKeysExhausted).toBe(true);

    // Second call: resets flag, succeeds
    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(state.allKeysExhausted).toBe(false);
  });
});
