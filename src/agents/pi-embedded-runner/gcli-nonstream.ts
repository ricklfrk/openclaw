/**
 * Non-streaming Cloud Code Assist (gcli) StreamFn.
 *
 * Calls CCA's `/v1internal:generateContent` (non-streaming) instead of
 * `streamGenerateContent`, then converts the single response into pi-ai's
 * event stream format — identical approach to `google-nonstream.ts` but
 * adapted for CCA's OAuth authentication and request envelope.
 *
 * Benefits over streaming + buffer:
 * - Non-streaming responses are often more complete and less likely to trip
 *   safety filters mid-flight.
 * - Thinking parts are in the complete response, so we can reliably detect
 *   `thought: true` and bare "think\n" prefixes before emitting any events.
 */
import type { StreamFn } from "@mariozechner/pi-agent-core";
import type {
  AssistantMessage,
  StreamOptions,
  TextContent,
  ThinkingContent,
  ToolCall,
} from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream, calculateCost } from "@mariozechner/pi-ai";
import { log } from "./logger.js";
import {
  extractRetryDelay,
  convertMessages,
  convertTools,
  isThinkingPart,
  retainThoughtSignature,
  mapStopReasonString,
  mapToolChoice,
} from "./pi-ai-google-internals.js";

let toolCallCounter = 0;

const DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com";

const GEMINI_CLI_HEADERS: Record<string, string> = {
  "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
  "X-Goog-Api-Client": "gl-node/22.17.0",
  "Client-Metadata": JSON.stringify({
    ideType: "IDE_UNSPECIFIED",
    platform: "PLATFORM_UNSPECIFIED",
    pluginType: "GEMINI",
  }),
};

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;
const MAX_EMPTY_RETRIES = 2;
const EMPTY_RETRY_BASE_DELAY_MS = 500;

type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

function clampReasoning(effort: ThinkingLevel): Exclude<ThinkingLevel, "xhigh"> {
  return effort === "xhigh" ? "high" : effort;
}

function getThinkingLevel(effort: string, modelId: string): string {
  if (modelId.includes("3-pro") || modelId.includes("3.1-pro")) {
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

function sanitizeSurrogates(text: string): string {
  return text.replace(
    /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g,
    "",
  );
}

function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Request was aborted"));
      return;
    }
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timeout);
      reject(new Error("Request was aborted"));
    });
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- mirrors pi-ai
function resolveThinkingConfig(model: any, options: any): Record<string, unknown> | undefined {
  if (!options?.reasoning || !model.reasoning) {
    return undefined;
  }
  const effort = clampReasoning(options.reasoning);
  const isGemini3 = model.id.includes("3-pro") || model.id.includes("3-flash");
  if (isGemini3) {
    return {
      includeThoughts: true,
      thinkingLevel: getThinkingLevel(effort, model.id),
    };
  }
  // Budget-based for older models
  const defaultBudgets: Record<string, number> = {
    minimal: 1024,
    low: 2048,
    medium: 8192,
    high: 16384,
  };
  const budgets = { ...defaultBudgets, ...options.thinkingBudgets };
  return {
    includeThoughts: true,
    thinkingBudget: budgets[effort] ?? 8192,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- mirrors pi-ai buildRequest
function buildCcaRequest(model: any, context: any, projectId: string, options: any = {}) {
  const contents = convertMessages(model, context);
  const generationConfig: Record<string, unknown> = {};

  if (options.temperature !== undefined) {
    generationConfig.temperature = options.temperature;
  }
  const maxTokens = options.maxTokens || Math.min(model.maxTokens ?? 32000, 32000);
  generationConfig.maxOutputTokens = maxTokens;

  const thinkingConfig = resolveThinkingConfig(model, options);
  if (thinkingConfig) {
    generationConfig.thinkingConfig = thinkingConfig;
    // Adjust maxTokens for budget-based thinking (mirrors pi-ai gcli logic)
    const budget = thinkingConfig.thinkingBudget;
    if (typeof budget === "number" && budget > 0) {
      const adjusted = Math.min(maxTokens + budget, model.maxTokens ?? 65536);
      generationConfig.maxOutputTokens = adjusted;
    }
  }

  const request: Record<string, unknown> = { contents };
  if (options.sessionId) {
    request.sessionId = options.sessionId;
  }
  if (context.systemPrompt) {
    request.systemInstruction = {
      parts: [{ text: sanitizeSurrogates(context.systemPrompt) }],
    };
  }
  if (Object.keys(generationConfig).length > 0) {
    request.generationConfig = generationConfig;
  }
  if (context.tools?.length > 0) {
    const useParameters = model.id.startsWith("claude-");
    request.tools = convertTools(context.tools, useParameters);
    if (options.toolChoice) {
      request.toolConfig = {
        functionCallingConfig: { mode: mapToolChoice(options.toolChoice) },
      };
    }
  }

  // Disable adjustable safety filters so CCA doesn't block on the four
  // configurable harm categories. Core protections (child safety, etc.)
  // remain enforced server-side regardless of this setting.
  request.safetySettings = [
    { category: "HARM_CATEGORY_HARASSMENT", threshold: "OFF" },
    { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "OFF" },
    { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "OFF" },
    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "OFF" },
  ];

  return {
    project: projectId,
    model: model.id,
    request,
    userAgent: "pi-coding-agent",
    requestId: `pi-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`,
  };
}

function extractErrorMessage(errorText: string): string {
  try {
    const parsed = JSON.parse(errorText);
    if (parsed.error?.message) {
      return parsed.error.message as string;
    }
  } catch {
    // not JSON
  }
  return errorText;
}

/**
 * Wrap a StreamFn so google-gemini-cli calls use CCA's non-streaming
 * `generateContent` endpoint. The response is converted into the standard
 * pi-ai event stream so downstream sees the same event sequence.
 *
 * Non-gcli models pass through to the original StreamFn unchanged.
 */
export function wrapGcliNonStreaming(streamFn: StreamFn): StreamFn {
  return (model, context, options) => {
    // Check both api and provider: the config schema only allows
    // api:"google-generative-ai" but the provider is "google-gemini-cli".
    if (model.api !== "google-gemini-cli" && model.provider !== "google-gemini-cli") {
      return streamFn(model, context, options);
    }

    log.debug(`non-streaming CCA call for ${model.provider}/${model.id}`);

    const stream = createAssistantMessageEventStream();

    void (async () => {
      const output: AssistantMessage = {
        role: "assistant",
        content: [],
        api: "google-gemini-cli",
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
        const apiKeyRaw = (options as StreamOptions | undefined)?.apiKey;
        if (!apiKeyRaw) {
          throw new Error(
            "Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate.",
          );
        }
        let accessToken: string;
        let projectId: string;
        try {
          const parsed = JSON.parse(apiKeyRaw);
          accessToken = parsed.token;
          projectId = parsed.projectId;
        } catch {
          throw new Error(
            "Invalid Google Cloud Code Assist credentials. Use /login to re-authenticate.",
          );
        }
        if (!accessToken || !projectId) {
          throw new Error(
            "Missing token or projectId in Google Cloud credentials. Use /login to re-authenticate.",
          );
        }

        const baseUrl = (model as unknown as Record<string, unknown>).baseUrl as string | undefined;
        const endpoint = baseUrl?.trim() || DEFAULT_ENDPOINT;
        const requestBody = buildCcaRequest(model, context, projectId, options);
        (options as StreamOptions | undefined)?.onPayload?.(requestBody, model);

        const requestHeaders: Record<string, string> = {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
          ...GEMINI_CLI_HEADERS,
          ...((options as Record<string, unknown>)?.headers as Record<string, string> | undefined),
        };
        const requestBodyJson = JSON.stringify(requestBody);
        const requestUrl = `${endpoint}/v1internal:generateContent`;
        const signal = (options as StreamOptions | undefined)?.signal;
        const maxRetryDelayMs =
          ((options as Record<string, unknown>)?.maxRetryDelayMs as number) ?? 60000;

        // Fetch with retry for rate limits / transient errors
        let responseBody: string | undefined;
        let lastError: Error | undefined;
        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
          if (signal?.aborted) {
            throw new Error("Request was aborted");
          }
          try {
            const response = await fetch(requestUrl, {
              method: "POST",
              headers: requestHeaders,
              body: requestBodyJson,
              signal,
            });

            if (response.ok) {
              responseBody = await response.text();
              break;
            }

            const errorText = await response.text();
            if (attempt < MAX_RETRIES && isRetryableStatus(response.status)) {
              const serverDelay = extractRetryDelay(errorText, response);
              const delayMs = serverDelay ?? BASE_DELAY_MS * 2 ** attempt;
              if (maxRetryDelayMs > 0 && serverDelay && serverDelay > maxRetryDelayMs) {
                const delaySec = Math.ceil(serverDelay / 1000);
                throw new Error(
                  `Server requested ${delaySec}s retry delay (max: ${Math.ceil(maxRetryDelayMs / 1000)}s). ${extractErrorMessage(errorText)}`,
                );
              }
              await sleep(delayMs, signal);
              continue;
            }
            throw new Error(
              `Cloud Code Assist API error (${response.status}): ${extractErrorMessage(errorText)}`,
            );
          } catch (error) {
            if (
              error instanceof Error &&
              (error.name === "AbortError" || error.message === "Request was aborted")
            ) {
              throw new Error("Request was aborted", { cause: error });
            }
            lastError = error instanceof Error ? error : new Error(String(error));
            if (lastError.message === "fetch failed" && lastError.cause instanceof Error) {
              lastError = new Error(`Network error: ${lastError.cause.message}`);
            }
            if (attempt < MAX_RETRIES) {
              await sleep(BASE_DELAY_MS * 2 ** attempt, signal);
              continue;
            }
            throw lastError;
          }
        }

        if (!responseBody) {
          throw lastError ?? new Error("Failed to get response after retries");
        }

        // Parse non-streaming response. CCA wraps in { response: ... } or
        // returns candidates at the top level.
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(responseBody);
        } catch {
          throw new Error(`Invalid JSON in CCA response: ${responseBody.slice(0, 200)}`);
        }
        let responseData = (parsed.response as Record<string, unknown> | undefined) ?? parsed;

        // Retry on empty response (no candidates / empty parts)
        let candidates = (responseData.candidates as Array<Record<string, unknown>>) ?? [];
        let candidate = candidates[0];
        let parts = (candidate?.content as Record<string, unknown>)?.parts as
          | Array<Record<string, unknown>>
          | undefined;

        for (let emptyAttempt = 0; emptyAttempt < MAX_EMPTY_RETRIES; emptyAttempt++) {
          if (parts && parts.length > 0) {
            break;
          }
          if (signal?.aborted) {
            throw new Error("Request was aborted");
          }

          log.warn(`[gcli-nonstream] empty response on attempt ${emptyAttempt + 1}, retrying`);
          await sleep(EMPTY_RETRY_BASE_DELAY_MS * 2 ** emptyAttempt, signal);

          const retryResp = await fetch(requestUrl, {
            method: "POST",
            headers: requestHeaders,
            body: requestBodyJson,
            signal,
          });
          if (!retryResp.ok) {
            const errText = await retryResp.text();
            throw new Error(
              `Cloud Code Assist API error (${retryResp.status}): ${extractErrorMessage(errText)}`,
            );
          }
          const retryBody = await retryResp.text();
          const retryParsed = JSON.parse(retryBody) as Record<string, unknown>;
          const retryData =
            (retryParsed.response as Record<string, unknown> | undefined) ?? retryParsed;
          candidates = (retryData.candidates as Array<Record<string, unknown>>) ?? [];
          candidate = candidates[0];
          parts = (candidate?.content as Record<string, unknown>)?.parts as
            | Array<Record<string, unknown>>
            | undefined;
          if (retryData.usageMetadata) {
            Object.assign(responseData, { usageMetadata: retryData.usageMetadata });
          }
        }

        // Fall through with empty content instead of throwing — lets the
        // non-conforming retry check trigger key rotation for safety blocks.
        if (!parts || parts.length === 0) {
          log.warn(
            "[gcli-nonstream] empty response after retries; falling through for key rotation",
          );
          parts = [];
        }

        stream.push({ type: "start", partial: output });
        const bareThinkPrefix = /^think\s*\n/i;

        for (const part of parts) {
          if (typeof part.text === "string") {
            const thinking = isThinkingPart(part) || bareThinkPrefix.test(part.text);

            if (thinking) {
              const block: ThinkingContent = {
                type: "thinking",
                thinking: part.text,
                thinkingSignature: retainThoughtSignature(
                  undefined,
                  part.thoughtSignature as string | undefined,
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
                  part.thoughtSignature as string | undefined,
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

          if (part.functionCall && typeof part.functionCall === "object") {
            const fc = part.functionCall as Record<string, unknown>;
            const providedId = fc.id as string | undefined;
            const needsNewId =
              !providedId ||
              output.content.some((b) => b.type === "toolCall" && b.id === providedId);
            const fcName = typeof fc.name === "string" ? fc.name : "fn";
            const toolCallId = needsNewId
              ? `${fcName}_${Date.now()}_${++toolCallCounter}`
              : providedId;

            const toolCall: ToolCall = {
              type: "toolCall",
              id: toolCallId,
              name: fcName,
              arguments: (fc.args as Record<string, unknown>) ?? {},
              ...(part.thoughtSignature
                ? { thoughtSignature: part.thoughtSignature as string }
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

        // Finish reason
        const finishReason = candidate?.finishReason as string | undefined;
        if (finishReason) {
          output.stopReason = mapStopReasonString(finishReason);
          if (output.content.some((b) => b.type === "toolCall")) {
            output.stopReason = "toolUse";
          }
        }

        // Usage
        const usageMetadata = responseData.usageMetadata as Record<string, number> | undefined;
        if (usageMetadata) {
          const promptTokens = usageMetadata.promptTokenCount || 0;
          const cacheReadTokens = usageMetadata.cachedContentTokenCount || 0;
          output.usage = {
            input: promptTokens - cacheReadTokens,
            output:
              (usageMetadata.candidatesTokenCount || 0) + (usageMetadata.thoughtsTokenCount || 0),
            cacheRead: cacheReadTokens,
            cacheWrite: 0,
            totalTokens: usageMetadata.totalTokenCount || 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          };
          calculateCost(model, output.usage);
        }

        if (signal?.aborted) {
          throw new Error("Request was aborted");
        }
        if (output.stopReason === "aborted") {
          throw new Error("Request was aborted");
        }
        // Safety/content-filter blocks arrive as stopReason "error" with empty
        // content. Treat them like empty responses (stopReason "stop") so the
        // non-conforming retry triggers key rotation — different keys may have
        // different safety thresholds.
        if (output.stopReason === "error") {
          if (output.content.length === 0) {
            const reason = (candidate?.finishReason as string) ?? "unknown";
            log.warn(
              `[gcli-nonstream] safety/content-filter block (finishReason=${reason}); treating as empty response for key rotation`,
            );
            output.stopReason = "stop";
          } else {
            throw new Error("An unknown error occurred");
          }
        }

        stream.push({ type: "done", reason: output.stopReason, message: output });
        stream.end();
      } catch (error) {
        output.stopReason = (options as StreamOptions | undefined)?.signal?.aborted
          ? "aborted"
          : "error";
        output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
        stream.push({ type: "error", reason: output.stopReason, error: output });
        stream.end();
      }
    })();

    return stream;
  };
}
