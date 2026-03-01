import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessageEvent } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import type { AuthProfileStore } from "../auth-profiles/types.js";
import {
  calculateAuthProfileCooldownMs,
  isProfileInCooldown,
  stampProfileLastUsed,
  markAuthProfileFailure,
} from "../auth-profiles/usage.js";
import {
  isRateLimitErrorMessage,
  isOverloadedErrorMessage,
} from "../pi-embedded-helpers/errors.js";
import { isGoogleModelApi } from "../pi-embedded-helpers/google.js";
import { log } from "./logger.js";

/** Shared mutable state between the streamFn wrapper and the run.ts while-loop. */
export type KeyRotationState = {
  /** Set to true when the wrapper has exhausted all candidate keys on a 429. */
  allKeysExhausted: boolean;
  /** The profile ID that was last used by the wrapper (for error attribution). */
  lastProfileId?: string;
  /** Current rotation index into profileCandidates. */
  rotationIndex: number;
};

export function createKeyRotationState(): KeyRotationState {
  return { allKeysExhausted: false, rotationIndex: 0 };
}

/**
 * Whether the model's transport is safe for key-rotation retry.
 *
 * Safe means that on a 429 error, no partial events have been pushed to
 * the consumer. This is true for:
 * - google-generative-ai: non-streaming generateContent via wrapGoogleNonStreaming
 * - google-gemini-cli / google-vertex: buffered by wrapStreamFnWithBuffer
 * - any model whose id contains "gemini": also buffered
 */
export function canSafelyRotate(model: { api?: string; id?: string }): boolean {
  if (model.api === "google-generative-ai") {
    return true;
  }
  if (isGoogleModelApi(model.api)) {
    return true;
  }
  if (model.id?.toLowerCase().includes("gemini")) {
    return true;
  }
  return false;
}

/**
 * Synchronously mark a profile as in-cooldown in the in-memory store so that
 * subsequent iterations within the same retry loop see it immediately.
 * The async `markAuthProfileFailure` call persists to disk in parallel.
 */
function setCooldownInMemory(store: AuthProfileStore, profileId: string): void {
  store.usageStats = store.usageStats ?? {};
  const prev = store.usageStats[profileId] ?? {};
  const nextErrorCount = (prev.errorCount ?? 0) + 1;
  store.usageStats[profileId] = {
    ...prev,
    errorCount: nextErrorCount,
    cooldownUntil: Date.now() + calculateAuthProfileCooldownMs(nextErrorCount),
  };
}

function isRetriableErrorEvent(event: AssistantMessageEvent): boolean {
  if (event.type !== "error") {
    return false;
  }
  const errorMsg = (event as { error?: { errorMessage?: string } }).error?.errorMessage ?? "";
  return isRateLimitErrorMessage(errorMsg) || isOverloadedErrorMessage(errorMsg);
}

export type WrapStreamFnWithKeyRotationParams = {
  streamFn: StreamFn;
  profileCandidates: Array<string | undefined>;
  /** Resolve the raw API key string for a given profile candidate. */
  resolveApiKey: (candidate: string | undefined) => Promise<string | undefined>;
  authStore: AuthProfileStore;
  agentDir?: string;
  rotationState: KeyRotationState;
};

/**
 * Wrap a StreamFn to proactively rotate API keys on each model call and
 * retry with the next key on rate-limit (429 / 503 overloaded) errors.
 *
 * Must be the outermost wrapper (after wrapStreamFnWithBuffer) so that
 * inner wrappers (non-streaming, buffer) guarantee no partial events on error.
 */
export function wrapStreamFnWithKeyRotation(params: WrapStreamFnWithKeyRotationParams): StreamFn {
  const { streamFn, profileCandidates, resolveApiKey, authStore, agentDir, rotationState } = params;

  if (profileCandidates.length <= 1) {
    return streamFn;
  }

  return (model, context, options) => {
    if (!canSafelyRotate(model)) {
      return streamFn(model, context, options);
    }

    const outputStream = createAssistantMessageEventStream();

    void (async () => {
      // Reset exhaustion flag at the start of each model call.
      rotationState.allKeysExhausted = false;

      const maxAttempts = profileCandidates.length;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const candidateIndex = (rotationState.rotationIndex + attempt) % profileCandidates.length;
        const candidate = profileCandidates[candidateIndex];

        if (candidate && isProfileInCooldown(authStore, candidate)) {
          continue;
        }

        let apiKey: string | undefined;
        try {
          apiKey = await resolveApiKey(candidate);
        } catch {
          log.warn(
            `[key-rotation] failed to resolve API key for profile ${candidate ?? "default"}; skipping`,
          );
          continue;
        }

        if (candidate) {
          rotationState.lastProfileId = candidate;
          void stampProfileLastUsed({
            store: authStore,
            profileId: candidate,
            agentDir,
          });
        }

        const callOptions = {
          ...options,
          ...(apiKey ? { apiKey } : {}),
          maxRetries: 0,
        };

        try {
          const innerStream = await Promise.resolve(streamFn(model, context, callOptions));

          // Consume inner stream (already buffered/non-streaming for safe models).
          const events: AssistantMessageEvent[] = [];
          let rateLimitEvent: AssistantMessageEvent | null = null;

          for await (const event of innerStream) {
            if (isRetriableErrorEvent(event)) {
              rateLimitEvent = event;
            } else {
              events.push(event);
            }
          }

          if (rateLimitEvent) {
            const errMsg =
              (rateLimitEvent as { error?: { errorMessage?: string } }).error?.errorMessage ??
              "rate limited";
            log.warn(
              `[key-rotation] profile ${candidate ?? "default"} hit rate limit: ${errMsg.slice(0, 120)}; trying next key (attempt ${attempt + 1}/${maxAttempts})`,
            );
            if (candidate) {
              setCooldownInMemory(authStore, candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "rate_limit",
                agentDir,
              });
            }
            continue;
          }

          // Success: replay collected events.
          for (const event of events) {
            outputStream.push(event);
          }
          outputStream.end();

          // Advance rotation index for the next model call (proactive rotation).
          rotationState.rotationIndex = (candidateIndex + 1) % profileCandidates.length;

          if (candidate) {
            rotationState.lastProfileId = candidate;
          }
          return;
        } catch (err) {
          // streamFn threw synchronously or the promise rejected (rare path).
          const errMsg = err instanceof Error ? err.message : String(err);
          if (isRateLimitErrorMessage(errMsg) || isOverloadedErrorMessage(errMsg)) {
            log.warn(
              `[key-rotation] profile ${candidate ?? "default"} threw rate limit error: ${errMsg.slice(0, 120)}; trying next key (attempt ${attempt + 1}/${maxAttempts})`,
            );
            if (candidate) {
              setCooldownInMemory(authStore, candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "rate_limit",
                agentDir,
              });
            }
            continue;
          }
          // Non-retriable error: propagate as error event.
          outputStream.push({
            type: "error",
            reason: "error",
            error: {
              role: "assistant" as const,
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
              stopReason: "error" as const,
              errorMessage: errMsg,
              timestamp: Date.now(),
            },
          } as AssistantMessageEvent);
          outputStream.end();
          return;
        }
      }

      // All keys exhausted.
      rotationState.allKeysExhausted = true;
      log.warn(`[key-rotation] all ${maxAttempts} profile keys exhausted due to rate limiting`);

      // Advance rotation index so the next prompt() picks up from where we left off.
      rotationState.rotationIndex =
        (rotationState.rotationIndex + maxAttempts) % profileCandidates.length;

      outputStream.push({
        type: "error",
        reason: "error",
        error: {
          role: "assistant" as const,
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
          stopReason: "error" as const,
          errorMessage:
            "All API key profiles exhausted due to rate limiting (429). Falling back to next model.",
          timestamp: Date.now(),
        },
      } as AssistantMessageEvent);
      outputStream.end();
    })();

    return outputStream;
  };
}
