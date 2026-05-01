import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessageEvent } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { log } from "./logger.js";

/**
 * Wrap a StreamFn to buffer the full model response before replaying events downstream.
 *
 * Converts real streaming into "fake streaming": the HTTP request runs to completion
 * first (all SSE events are collected), then events are replayed to downstream
 * consumers. This ensures the complete response is available before any downstream
 * processing, which enables robust reasoning-tag stripping (first-open / last-close)
 * and avoids mid-stream safety-filter interruptions.
 *
 * google-generative-ai and google-gemini-cli are handled by their respective
 * non-streaming wrappers upstream (wrapGoogleNonStreaming / wrapGcliNonStreaming)
 * and are excluded here to avoid double-buffering.
 */
function shouldBufferStream(model: { api?: string; id?: string }): boolean {
  // Already handled by dedicated non-streaming wrappers — skip to avoid double-buffering.
  if (model.api === "google-generative-ai" || model.api === "google-gemini-cli") {
    return false;
  }
  return true;
}

export function wrapStreamFnWithBuffer(streamFn: StreamFn): StreamFn {
  return (model, context, options) => {
    if (!shouldBufferStream(model)) {
      return streamFn(model, context, options);
    }

    log.debug(`buffering stream for ${model.provider}/${model.id}`);

    // Create a new stream for downstream; events are replayed after the real stream finishes.
    const bufferedStream = createAssistantMessageEventStream();

    void (async () => {
      try {
        const realStream = await Promise.resolve(streamFn(model, context, options));
        const events: AssistantMessageEvent[] = [];

        // Consume the entire real stream first — HTTP connection stays open
        // until the model finishes, but downstream doesn't see any events yet.
        for await (const event of realStream) {
          events.push(event);
        }

        log.debug(`buffered ${events.length} events for ${model.id}, replaying`);

        // Replay all collected events to downstream.
        for (const event of events) {
          bufferedStream.push(event);
        }
        bufferedStream.end();
      } catch (error) {
        // The real stream should always terminate with a done/error event,
        // but if something truly unexpected happens, propagate as an error event.
        log.warn(
          `buffered stream error for ${model.id}: ${error instanceof Error ? error.message : String(error)}`,
        );
        bufferedStream.push({
          type: "error",
          reason: "error",
          error: {
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
            stopReason: "error",
            errorMessage: error instanceof Error ? error.message : String(error),
            timestamp: Date.now(),
          },
        });
        bufferedStream.end();
      }
    })();

    return bufferedStream;
  };
}
