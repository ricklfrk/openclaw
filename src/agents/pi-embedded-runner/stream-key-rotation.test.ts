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

function makeContentFilterErrorEvent(): AssistantMessageEvent {
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
      errorMessage: "Google content filter blocked (finishReason=PROHIBITED_CONTENT)",
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
  name: "Gemini 3.1 Pro Preview",
  baseUrl: "https://generativelanguage.googleapis.com",
  reasoning: true,
  input: ["text"] as ("text" | "image")[],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1_000_000,
  maxTokens: 65_536,
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
  it("calls streamFn once on success with single profile candidate", async () => {
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
    expect(calls).toHaveLength(1);
    // Single-profile still routes through the wrapper so the resolved apiKey is
    // passed down to the inner streamFn (important for wrapGoogleNonStreaming
    // and same-key content-filter retries).
    expect(calls[0].apiKey).toBe("key-p1");
  });

  it("retries same key up to 5 attempts on content-filter block (single profile)", async () => {
    // 4 content-filter blocks then success on 5th attempt.
    const { streamFn, calls } = createMockStreamFn([
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeDoneEvent("success on 5th attempt")],
    ]);
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

    // 5 calls total, all on same key.
    expect(calls).toHaveLength(5);
    expect(calls.every((c) => c.apiKey === "key-p1")).toBe(true);
    // Success events should replay.
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    expect(state.allKeysExhausted).toBe(false);
    // Same-key content-filter retries should NOT mark the profile as skipped.
    expect(state.skipProfiles.has("p1")).toBe(false);
  });

  it("gives up after 5 content-filter attempts on single profile", async () => {
    const { streamFn, calls } = createMockStreamFn([
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()], // 6th should not be reached
    ]);
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

    // Exactly 5 attempts, 6th call must NOT happen.
    expect(calls).toHaveLength(5);
    // After exhausting, profile is marked and rotation reports "all exhausted".
    expect(state.skipProfiles.has("p1")).toBe(true);
    expect(state.allKeysExhausted).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
  });

  it("retries same key on thrown content-filter block (single profile)", async () => {
    let callCount = 0;
    const calls: Array<{ apiKey?: string }> = [];
    const streamFn: StreamFn = (_model, _context, options) => {
      callCount++;
      calls.push({ apiKey: (options as { apiKey?: string })?.apiKey });
      if (callCount < 3) {
        throw new Error("Google content filter blocked (finishReason=PROHIBITED_CONTENT)");
      }
      const stream = createAssistantMessageEventStream();
      stream.push(makeDoneEvent("third attempt succeeded"));
      stream.end();
      return stream;
    };
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

    expect(callCount).toBe(3);
    expect(calls.every((c) => c.apiKey === "key-p1")).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    expect(state.skipProfiles.has("p1")).toBe(false);
  });

  it("rotates on rate-limit with single profile (no same-key retry for 429)", async () => {
    // Rate-limit is NOT retried on the same key — skips straight to exhaust.
    const { streamFn, calls } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeDoneEvent("should not be reached")],
    ]);
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

    expect(calls).toHaveLength(1);
    expect(state.allKeysExhausted).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
  });

  it("content-filter on multi-profile retries same profile before rotating", async () => {
    // p1 returns content-filter 5 times (exhausts per-profile budget), then rotates to p2 which succeeds.
    const { streamFn, calls } = createMockStreamFn([
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeContentFilterErrorEvent()],
      [makeDoneEvent("from p2")],
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

    // 5 on p1, then 1 on p2.
    expect(calls).toHaveLength(6);
    expect(calls.slice(0, 5).every((c) => c.apiKey === "key-p1")).toBe(true);
    expect(calls[5].apiKey).toBe("key-p2");
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    // p1 exhausted → skipped; p2 succeeded → not skipped.
    expect(state.skipProfiles.has("p1")).toBe(true);
    expect(state.skipProfiles.has("p2")).toBe(false);
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
      wrapped(geminiModel, { systemPrompt: "", messages: [], tools: [] }, {
        maxRetries: 5,
      } as Record<string, unknown>),
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

  it("skips profiles in skipProfiles set (empty-response rotation)", async () => {
    const { streamFn, calls } = createMockStreamFn([
      // p1 skipped, p2 skipped → only p3 tried → succeeds
      [makeDoneEvent("from p3")],
    ]);
    const state = createKeyRotationState();
    state.skipProfiles.add("p1");
    state.skipProfiles.add("p2");
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

    expect(calls).toHaveLength(1);
    expect(calls[0].apiKey).toBe("key-p3");
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    expect(state.allKeysExhausted).toBe(false);
  });

  it("sets allKeysExhausted when all non-skipped profiles are rate-limited", async () => {
    const { streamFn, calls } = createMockStreamFn([
      // p3 hits 429 → all exhausted (p1, p2 skipped)
      [makeRateLimitErrorEvent()],
    ]);
    const state = createKeyRotationState();
    state.skipProfiles.add("p1");
    state.skipProfiles.add("p2");
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

    expect(calls).toHaveLength(1);
    expect(state.allKeysExhausted).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
  });

  it("resets allKeysExhausted on each new model call (new turn)", async () => {
    const { streamFn } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeRateLimitErrorEvent()],
      // Third call (second invocation, new turn): succeeds
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
    // 429 profiles are in skipProfiles for this turn
    expect(state.skipProfiles.size).toBe(2);

    // Simulate new turn: clear skipProfiles (outer loop does this between turns)
    state.skipProfiles.clear();
    // Second call: resets flag, succeeds
    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(state.allKeysExhausted).toBe(false);
  });

  it("keeps 429 profiles in skipProfiles within the same turn", async () => {
    const { streamFn } = createMockStreamFn([
      [makeRateLimitErrorEvent()],
      [makeRateLimitErrorEvent()],
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

    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(state.allKeysExhausted).toBe(true);
    expect(state.skipProfiles.has("p1")).toBe(true);
    expect(state.skipProfiles.has("p2")).toBe(true);

    // Same turn (skipProfiles not cleared): still exhausted immediately
    await collectEvents(wrapped(geminiModel, ctx, {}));
    expect(state.allKeysExhausted).toBe(true);
  });
});
