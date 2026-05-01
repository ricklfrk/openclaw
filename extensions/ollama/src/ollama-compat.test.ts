import { describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../../../src/config/config.js";
import {
  isOllamaCompatProvider,
  resolveOllamaCompatNumCtxEnabled,
  shouldInjectOllamaCompatNumCtx,
  wrapOllamaCompatNumCtx,
} from "./stream.js";

function createOllamaProviderConfig(injectNumCtxForOpenAICompat: boolean): OpenClawConfig {
  return {
    models: {
      providers: {
        ollama: {
          baseUrl: "http://127.0.0.1:11434/v1",
          api: "openai-completions",
          injectNumCtxForOpenAICompat,
          models: [],
        },
      },
    },
  } as OpenClawConfig;
}

describe("isOllamaCompatProvider", () => {
  it("detects native ollama provider id", () => {
    expect(
      isOllamaCompatProvider({
        provider: "ollama",
        api: "openai-completions",
        baseUrl: "https://example.com/v1",
      }),
    ).toBe(true);
  });

  it("detects localhost Ollama OpenAI-compatible endpoint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://127.0.0.1:11434/v1",
      }),
    ).toBe(true);
  });

  it("does not misclassify non-local OpenAI-compatible providers", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "https://api.openrouter.ai/v1",
      }),
    ).toBe(false);
  });

  it("detects remote Ollama-compatible endpoint when provider id hints ollama", () => {
    expect(
      isOllamaCompatProvider({
        provider: "my-ollama",
        api: "openai-completions",
        baseUrl: "http://ollama-host:11434/v1",
      }),
    ).toBe(true);
  });

  it("detects IPv6 loopback Ollama OpenAI-compatible endpoint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://[::1]:11434/v1",
      }),
    ).toBe(true);
  });

  it("does not classify arbitrary remote hosts on 11434 without ollama provider hint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://example.com:11434/v1",
      }),
    ).toBe(false);
  });
});

describe("wrapOllamaCompatNumCtx", () => {
  it("injects num_ctx and preserves downstream onPayload hooks", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = { options: { temperature: 0.1 } };
      options?.onPayload?.(payload, _model);
      payloadSeen = payload;
      return {} as never;
    });
    const downstream = vi.fn();

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 202752);
    void wrapped({} as never, {} as never, { onPayload: downstream } as never);

    expect(baseFn).toHaveBeenCalledTimes(1);
    expect((payloadSeen?.options as Record<string, unknown> | undefined)?.num_ctx).toBe(202752);
    expect(downstream).toHaveBeenCalledTimes(1);
  });

  it("deserializes assistant tool_call arguments for Ollama OpenAI-compatible payloads", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = {
        messages: [
          {
            role: "assistant",
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "read",
                  arguments: '{"path":"/tmp/test.txt"}',
                },
              },
            ],
          },
        ],
      };
      options?.onPayload?.(payload, _model);
      payloadSeen = payload;
      return {} as never;
    });

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 8192);
    void wrapped({} as never, {} as never, undefined as never);

    const messageRecord = (
      payloadSeen?.messages as Array<Record<string, unknown>> | undefined
    )?.[0];
    const toolCall = (messageRecord?.tool_calls as Array<Record<string, unknown>> | undefined)?.[0];

    expect(toolCall?.function).toEqual({
      name: "read",
      arguments: { path: "/tmp/test.txt" },
    });
  });

  it("deserializes assistant function_call arguments for Ollama OpenAI-compatible payloads", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = {
        messages: [
          {
            role: "assistant",
            function_call: {
              name: "exec",
              arguments: '{"command":"pwd"}',
            },
          },
        ],
      };
      options?.onPayload?.(payload, _model);
      payloadSeen = payload;
      return {} as never;
    });

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 8192);
    void wrapped({} as never, {} as never, undefined as never);

    const messageRecord = (
      payloadSeen?.messages as Array<Record<string, unknown>> | undefined
    )?.[0];

    expect(messageRecord?.function_call).toEqual({
      name: "exec",
      arguments: { command: "pwd" },
    });
  });

  it("preserves unsafe integers when deserializing assistant tool_call arguments", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = {
        messages: [
          {
            role: "assistant",
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "read",
                  arguments: '{"path":9223372036854775807,"nested":{"thread":1234567890123456789}}',
                },
              },
            ],
          },
        ],
      };
      options?.onPayload?.(payload, _model);
      payloadSeen = payload;
      return {} as never;
    });

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 8192);
    void wrapped({} as never, {} as never, undefined as never);

    const messageRecord = (
      payloadSeen?.messages as Array<Record<string, unknown>> | undefined
    )?.[0];
    const toolCall = (messageRecord?.tool_calls as Array<Record<string, unknown>> | undefined)?.[0];

    expect(toolCall?.function).toEqual({
      name: "read",
      arguments: {
        path: "9223372036854775807",
        nested: { thread: "1234567890123456789" },
      },
    });
  });

  it("preserves unsafe integers when deserializing assistant function_call arguments", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = {
        messages: [
          {
            role: "assistant",
            function_call: {
              name: "exec",
              arguments: '{"thread":9223372036854775807}',
            },
          },
        ],
      };
      options?.onPayload?.(payload, _model);
      payloadSeen = payload;
      return {} as never;
    });

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 8192);
    void wrapped({} as never, {} as never, undefined as never);

    const messageRecord = (
      payloadSeen?.messages as Array<Record<string, unknown>> | undefined
    )?.[0];

    expect(messageRecord?.function_call).toEqual({
      name: "exec",
      arguments: { thread: "9223372036854775807" },
    });
  });
});

describe("resolveOllamaCompatNumCtxEnabled", () => {
  it("defaults to true when config is missing", () => {
    expect(resolveOllamaCompatNumCtxEnabled({ providerId: "ollama" })).toBe(true);
  });

  it("defaults to true when provider config is missing", () => {
    expect(
      resolveOllamaCompatNumCtxEnabled({
        config: { models: { providers: {} } },
        providerId: "ollama",
      }),
    ).toBe(true);
  });

  it("returns false when provider flag is explicitly disabled", () => {
    expect(
      resolveOllamaCompatNumCtxEnabled({
        config: createOllamaProviderConfig(false),
        providerId: "ollama",
      }),
    ).toBe(false);
  });
});

describe("shouldInjectOllamaCompatNumCtx", () => {
  it("requires openai-completions adapter", () => {
    expect(
      shouldInjectOllamaCompatNumCtx({
        model: {
          provider: "ollama",
          api: "openai-responses",
          baseUrl: "http://127.0.0.1:11434/v1",
        },
      }),
    ).toBe(false);
  });

  it("respects provider flag disablement", () => {
    expect(
      shouldInjectOllamaCompatNumCtx({
        model: {
          provider: "ollama",
          api: "openai-completions",
          baseUrl: "http://127.0.0.1:11434/v1",
        },
        config: createOllamaProviderConfig(false),
        providerId: "ollama",
      }),
    ).toBe(false);
  });
});
