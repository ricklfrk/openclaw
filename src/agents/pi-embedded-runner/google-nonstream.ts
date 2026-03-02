/**
 * Non-streaming Google Generative AI StreamFn.
 *
 * Calls `generateContent` (non-streaming) instead of `generateContentStream`,
 * then converts the single response into pi-ai's event stream format.
 *
 * Motivation: streaming Google API calls can be interrupted mid-flight by
 * safety filters, producing partial/empty content. Non-streaming calls either
 * succeed fully or fail cleanly. Downstream still gets "fake streaming"
 * (typing indicators) because events are replayed through the event stream.
 */
import { GoogleGenAI } from "@google/genai";
import type { StreamFn } from "@mariozechner/pi-agent-core";
import type {
  AssistantMessage,
  StreamOptions,
  TextContent,
  ThinkingContent,
  ToolCall,
} from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream, calculateCost } from "@mariozechner/pi-ai";
// pi-ai internal — no exports map, safe to import by path
import {
  convertMessages,
  convertTools,
  isThinkingPart,
  retainThoughtSignature,
  mapStopReason,
  mapToolChoice,
} from "@mariozechner/pi-ai/dist/providers/google-shared.js";
import { log } from "./logger.js";

let toolCallCounter = 0;

function isNonStreamableGoogleApi(api?: string | null): boolean {
  return api === "google-generative-ai";
}

type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

function clampReasoning(effort: ThinkingLevel): Exclude<ThinkingLevel, "xhigh"> {
  return effort === "xhigh" ? "high" : effort;
}

function isGemini3ProModel(id: string): boolean {
  return id.includes("3-pro") || id.includes("3.1-pro");
}

function isGemini3FlashModel(id: string): boolean {
  return id.includes("3-flash") || id.includes("3.1-flash");
}

/**
 * Map reasoning effort to Gemini 3 thinkingLevel enum.
 * Mirrors pi-ai's getGemini3ThinkingLevel.
 */
function getGemini3ThinkingLevel(effort: string, modelId: string): string {
  if (isGemini3ProModel(modelId)) {
    switch (effort) {
      case "minimal":
      case "low":
        return "LOW";
      case "medium":
      case "high":
        return "HIGH";
    }
  }
  switch (effort) {
    case "minimal":
      return "MINIMAL";
    case "low":
      return "LOW";
    case "medium":
      return "MEDIUM";
    case "high":
      return "HIGH";
  }
  return "LOW";
}

/**
 * Map reasoning effort to token budget for older Gemini models.
 * Mirrors pi-ai's getGoogleBudget.
 */
function getGoogleBudget(
  modelId: string,
  effort: string,
  customBudgets?: Record<string, number>,
): number {
  if (customBudgets?.[effort] !== undefined) {
    return customBudgets[effort];
  }
  if (modelId.includes("2.5-pro")) {
    const b: Record<string, number> = { minimal: 128, low: 2048, medium: 8192, high: 32768 };
    return b[effort] ?? -1;
  }
  if (modelId.includes("2.5-flash")) {
    const b: Record<string, number> = { minimal: 128, low: 2048, medium: 8192, high: 24576 };
    return b[effort] ?? -1;
  }
  return -1;
}

/**
 * Convert raw SimpleStreamOptions (with `reasoning`) into processed options
 * (with `thinking`), mirroring pi-ai's streamSimpleGoogle logic.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- mirrors pi-ai streamSimpleGoogle
function resolveThinkingOptions(model: any, options: any): any {
  const base = {
    temperature: options?.temperature,
    maxTokens: options?.maxTokens || Math.min(model.maxTokens ?? 32000, 32000),
    signal: options?.signal,
    apiKey: options?.apiKey,
    cacheRetention: options?.cacheRetention,
    sessionId: options?.sessionId,
    headers: options?.headers,
    onPayload: options?.onPayload,
    maxRetryDelayMs: options?.maxRetryDelayMs,
    metadata: options?.metadata,
  };

  if (!options?.reasoning) {
    return { ...base, thinking: { enabled: false } };
  }

  const effort = clampReasoning(options.reasoning);
  if (isGemini3ProModel(model.id) || isGemini3FlashModel(model.id)) {
    return {
      ...base,
      thinking: { enabled: true, level: getGemini3ThinkingLevel(effort, model.id) },
    };
  }
  return {
    ...base,
    thinking: {
      enabled: true,
      budgetTokens: getGoogleBudget(model.id, effort, options.thinkingBudgets),
    },
  };
}

function sanitizeSurrogates(text: string): string {
  return text.replace(
    /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g,
    "",
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- mirrors pi-ai createClient
function createClient(model: any, apiKey: string, optionsHeaders?: Record<string, string>) {
  const httpOptions: Record<string, unknown> = {};
  if (model.baseUrl) {
    httpOptions.baseUrl = model.baseUrl;
    httpOptions.apiVersion = "";
  }
  if (model.headers || optionsHeaders) {
    httpOptions.headers = { ...model.headers, ...optionsHeaders };
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: Object.keys(httpOptions).length > 0 ? httpOptions : undefined,
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- mirrors pi-ai buildParams
function buildParams(model: any, context: any, options: any = {}) {
  const contents = convertMessages(model, context);
  const generationConfig: Record<string, unknown> = {};
  if (options.temperature !== undefined) {
    generationConfig.temperature = options.temperature;
  }
  if (options.maxTokens !== undefined) {
    generationConfig.maxOutputTokens = options.maxTokens;
  }

  const config: Record<string, unknown> = {
    ...(Object.keys(generationConfig).length > 0 && generationConfig),
    ...(context.systemPrompt && { systemInstruction: sanitizeSurrogates(context.systemPrompt) }),
    ...(context.tools?.length > 0 && { tools: convertTools(context.tools) }),
  };

  if (context.tools?.length > 0 && options.toolChoice) {
    config.toolConfig = { functionCallingConfig: { mode: mapToolChoice(options.toolChoice) } };
  } else {
    config.toolConfig = undefined;
  }

  // Disable all Gemini safety filters to prevent content blocking.
  // pi-ai's streamGoogle does not set these either; we add them explicitly
  // so non-streaming calls never get silently blocked by safety filters.
  config.safetySettings = [
    { category: "HARM_CATEGORY_HARASSMENT", threshold: "OFF" },
    { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "OFF" },
    { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "OFF" },
    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "OFF" },
  ];

  if (options.thinking?.enabled && model.reasoning) {
    const tc: Record<string, unknown> = { includeThoughts: true };
    if (options.thinking.level !== undefined) {
      tc.thinkingLevel = options.thinking.level;
    } else if (options.thinking.budgetTokens !== undefined) {
      tc.thinkingBudget = options.thinking.budgetTokens;
    }
    config.thinkingConfig = tc;
  }

  if (options.signal) {
    if (options.signal.aborted) {
      throw new Error("Request aborted");
    }
    config.abortSignal = options.signal;
  }

  return { model: model.id, contents, config };
}

/**
 * Wrap a StreamFn so Google model API calls use `generateContent` (non-streaming)
 * instead of `generateContentStream`. The single response is converted into
 * the standard pi-ai event stream so downstream sees the same event sequence.
 *
 * Non-Google models pass through to the original StreamFn unchanged.
 */
export function wrapGoogleNonStreaming(streamFn: StreamFn): StreamFn {
  return (model, context, options) => {
    if (!isNonStreamableGoogleApi(model.api)) {
      return streamFn(model, context, options);
    }

    log.debug(`non-streaming Google call for ${model.provider}/${model.id}`);

    const stream = createAssistantMessageEventStream();

    void (async () => {
      const output: AssistantMessage = {
        role: "assistant",
        content: [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: Date.now(),
      };

      try {
        const resolved = resolveThinkingOptions(model, options);
        const apiKey = resolved.apiKey || "";
        const client = createClient(model, apiKey, resolved.headers);
        const params = buildParams(model, context, resolved);
        resolved.onPayload?.(params);

        const response = await client.models.generateContent(params);

        stream.push({ type: "start", partial: output });

        const candidate = response.candidates?.[0];
        if (candidate?.content?.parts) {
          for (const part of candidate.content.parts) {
            if (part.text !== undefined) {
              // Gemini models sometimes emit thinking as a plain text part
              // (thought !== true) with the raw prefix "think\n".  Reclassify
              // these as thinking blocks so downstream filters handle them.
              const bareThinkPrefix = /^think\s*\n/i;
              const thinking = isThinkingPart(part) || bareThinkPrefix.test(part.text);

              if (thinking) {
                const block: ThinkingContent = {
                  type: "thinking",
                  thinking: part.text,
                  thinkingSignature: retainThoughtSignature(
                    undefined,
                    (part as Record<string, unknown>).thoughtSignature as string | undefined,
                  ),
                };
                output.content.push(block);
                const idx = output.content.length - 1;
                stream.push({ type: "thinking_start", contentIndex: idx, partial: output });
                stream.push({
                  type: "thinking_delta",
                  contentIndex: idx,
                  delta: block.thinking,
                  partial: output,
                });
                stream.push({
                  type: "thinking_end",
                  contentIndex: idx,
                  content: block.thinking,
                  partial: output,
                });
              } else {
                const block: TextContent = {
                  type: "text",
                  text: part.text,
                  textSignature: retainThoughtSignature(
                    undefined,
                    (part as Record<string, unknown>).thoughtSignature as string | undefined,
                  ),
                };
                output.content.push(block);
                const idx = output.content.length - 1;
                stream.push({ type: "text_start", contentIndex: idx, partial: output });
                stream.push({
                  type: "text_delta",
                  contentIndex: idx,
                  delta: block.text,
                  partial: output,
                });
                stream.push({
                  type: "text_end",
                  contentIndex: idx,
                  content: block.text,
                  partial: output,
                });
              }
            }

            if (part.functionCall) {
              const fc = part.functionCall;
              const providedId = (fc as Record<string, unknown>).id as string | undefined;
              const needsNewId =
                !providedId ||
                output.content.some((b) => b.type === "toolCall" && b.id === providedId);
              const toolCallId = needsNewId
                ? `${fc.name}_${Date.now()}_${++toolCallCounter}`
                : providedId;

              const toolCall: ToolCall = {
                type: "toolCall",
                id: toolCallId,
                name: fc.name || "",
                arguments: fc.args ?? {},
                ...((part as Record<string, unknown>).thoughtSignature
                  ? {
                      thoughtSignature: (part as Record<string, unknown>).thoughtSignature as
                        | string
                        | undefined,
                    }
                  : {}),
              };
              output.content.push(toolCall);
              const idx = output.content.length - 1;
              stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
              stream.push({
                type: "toolcall_delta",
                contentIndex: idx,
                delta: JSON.stringify(toolCall.arguments),
                partial: output,
              });
              stream.push({ type: "toolcall_end", contentIndex: idx, toolCall, partial: output });
            }
          }
        }

        if (candidate?.finishReason) {
          output.stopReason = mapStopReason(candidate.finishReason);
          if (output.content.some((b) => b.type === "toolCall")) {
            output.stopReason = "toolUse";
          }
        }

        if (response.usageMetadata) {
          const um = response.usageMetadata;
          output.usage = {
            input: um.promptTokenCount || 0,
            output: (um.candidatesTokenCount || 0) + (um.thoughtsTokenCount || 0),
            cacheRead: um.cachedContentTokenCount || 0,
            cacheWrite: 0,
            totalTokens: um.totalTokenCount || 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          };
          calculateCost(model, output.usage);
        }

        if (resolved.signal?.aborted) {
          throw new Error("Request was aborted");
        }
        if (output.stopReason === "aborted" || output.stopReason === "error") {
          throw new Error("An unknown error occurred");
        }

        stream.push({
          type: "done",
          reason: output.stopReason,
          message: output,
        });
        stream.end();
      } catch (error) {
        output.stopReason = (options as StreamOptions | undefined)?.signal?.aborted
          ? "aborted"
          : "error";
        output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
        stream.push({
          type: "error",
          reason: output.stopReason,
          error: output,
        });
        stream.end();
      }
    })();

    return stream;
  };
}
