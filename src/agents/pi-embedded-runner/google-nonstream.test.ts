import { afterEach, describe, expect, it, vi } from "vitest";

const { GoogleGenAIMock, generateContentMock } = vi.hoisted(() => {
  const generateContentMock = vi.fn();
  const GoogleGenAIMock = vi.fn(function GoogleGenAI() {
    return {
      models: {
        generateContent: generateContentMock,
      },
    };
  });
  return { GoogleGenAIMock, generateContentMock };
});

vi.mock("@google/genai", () => ({
  GoogleGenAI: GoogleGenAIMock,
}));

import { wrapGoogleNonStreaming } from "./google-nonstream.js";

// Minimal model descriptor matching pi-ai's GoogleModel shape the wrapper touches.
const GEMINI_MODEL = {
  id: "gemini-3.1-pro-preview-customtools",
  provider: "lab",
  api: "google-generative-ai",
  maxTokens: 8192,
  reasoning: true,
  input: ["text"],
  // cost fields satisfy calculateCost's signature with zeros
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
} as unknown as Parameters<ReturnType<typeof wrapGoogleNonStreaming>>[0];

const EMPTY_CONTEXT = {
  systemPrompt: "sys",
  messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "hi" }] }],
  tools: [],
};

// collect the final output or error from the wrapped stream
async function drain(
  stream: AsyncIterable<
    | { type: "done"; message?: { content?: unknown[] } }
    | { type: "error"; error?: { errorMessage?: string } }
    | { type: string }
  >,
): Promise<{ done?: unknown; errorMessage?: string }> {
  const result: { done?: unknown; errorMessage?: string } = {};
  for await (const ev of stream) {
    if ((ev as { type: string }).type === "done") {
      result.done = ev;
    } else if ((ev as { type: string }).type === "error") {
      result.errorMessage = (ev as { error?: { errorMessage?: string } }).error?.errorMessage;
    }
  }
  return result;
}

async function invokeStreamFn(options: Record<string, unknown> = {}) {
  const upstream = vi.fn();
  const wrapped = wrapGoogleNonStreaming(
    upstream as unknown as Parameters<typeof wrapGoogleNonStreaming>[0],
  );
  // StreamFn may return either the stream or a Promise of it.
  return await Promise.resolve(
    wrapped(
      GEMINI_MODEL,
      EMPTY_CONTEXT as never,
      {
        apiKey: "fake",
        ...options,
      } as never,
    ),
  );
}

describe("wrapGoogleNonStreaming silent safety-block detection", () => {
  afterEach(() => {
    generateContentMock.mockReset();
    GoogleGenAIMock.mockClear();
  });

  it("throws content-filter error when Gemini returns finishReason=STOP with empty parts", async () => {
    // Silent block: clean stop but no content parts.  This is what we observed
    // in gateway.err.log at 2026-04-23T00:15:18 — NSFW prompt triggers a
    // safetySettings=OFF-bypassed block that surfaces as empty stop, not as
    // explicit PROHIBITED_CONTENT.  Must throw so key-rotation retries the
    // same key (non-deterministic) up to 5 times before failing over.
    generateContentMock.mockResolvedValue({
      candidates: [{ finishReason: "STOP", content: { parts: [] } }],
      promptFeedback: { blockReason: "OTHER" },
    });

    const stream = await invokeStreamFn();
    const { done, errorMessage } = await drain(stream);
    expect(done).toBeUndefined();
    expect(errorMessage).toMatch(/content filter blocked/i);
    expect(errorMessage).toMatch(/STOP-empty/);
    expect(errorMessage).toMatch(/blockReason=OTHER/);
  });

  it("throws content-filter error when candidate is missing (empty candidates array)", async () => {
    // Another silent-block shape: candidates array empty or candidate has no
    // content at all.  Wrapper builds an output with stopReason="stop" (the
    // default) and zero content — same symptom as above.
    generateContentMock.mockResolvedValue({ candidates: [] });

    const stream = await invokeStreamFn();
    const { done, errorMessage } = await drain(stream);
    expect(done).toBeUndefined();
    expect(errorMessage).toMatch(/content filter blocked/i);
  });

  it("passes through successfully when Gemini returns a text part", async () => {
    generateContentMock.mockResolvedValue({
      candidates: [
        {
          finishReason: "STOP",
          content: { parts: [{ text: "hello" }] },
        },
      ],
    });

    const stream = await invokeStreamFn();
    const { done, errorMessage } = await drain(stream);
    expect(errorMessage).toBeUndefined();
    expect(done).toBeDefined();
  });

  it("passes through successfully when Gemini returns only a tool call (no text)", async () => {
    // Tool-only responses are legitimate (the agent wants to act, not speak).
    // Must NOT trip the silent-block guard.
    generateContentMock.mockResolvedValue({
      candidates: [
        {
          finishReason: "STOP",
          content: {
            parts: [{ functionCall: { name: "do_thing", args: { x: 1 } } }],
          },
        },
      ],
    });

    const stream = await invokeStreamFn();
    const { done, errorMessage } = await drain(stream);
    expect(errorMessage).toBeUndefined();
    expect(done).toBeDefined();
  });

  it("still throws on explicit stopReason=error with empty content", async () => {
    // Original behavior preserved: PROHIBITED_CONTENT / SAFETY arrive as
    // stopReason=error and already threw.  Guard the regression.
    generateContentMock.mockResolvedValue({
      candidates: [{ finishReason: "PROHIBITED_CONTENT", content: { parts: [] } }],
    });

    const stream = await invokeStreamFn();
    const { done, errorMessage } = await drain(stream);
    expect(done).toBeUndefined();
    expect(errorMessage).toMatch(/content filter blocked/i);
    expect(errorMessage).toMatch(/PROHIBITED_CONTENT/);
  });
});
