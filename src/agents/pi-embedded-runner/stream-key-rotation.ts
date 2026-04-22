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
import type { KeyRotationState } from "./stream-key-rotation.types.js";
export type { KeyRotationState } from "./stream-key-rotation.types.js";
export { createKeyRotationState } from "./stream-key-rotation.types.js";

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

/** Content filter blocks are retriable because (a) different API keys may belong
 *  to projects with different safety thresholds, and (b) Gemini's content-safety
 *  classifier is non-deterministic on the same payload — especially for output-
 *  side PROHIBITED_CONTENT flags — so retrying the same key can still succeed. */
function isContentFilterBlockMessage(msg: string): boolean {
  return /content filter blocked/i.test(msg);
}

/**
 * Max attempts per profile when the error is a content-filter block.
 * Includes the initial attempt, so "5" = 1 initial call + up to 4 retries on the
 * same key before rotating to the next profile (or giving up on single-key).
 *
 * Rationale: Google's PROHIBITED_CONTENT filter is non-deterministic for output-
 * side blocks, so same-key retries recover frequently. Rate-limit / overloaded
 * errors do NOT benefit from same-key retry and skip straight to rotation.
 */
const MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE = 5;

function isRetriableErrorEvent(event: AssistantMessageEvent): boolean {
  if (event.type !== "error") {
    return false;
  }
  const errorMsg = (event as { error?: { errorMessage?: string } }).error?.errorMessage ?? "";
  return (
    isRateLimitErrorMessage(errorMsg) ||
    isOverloadedErrorMessage(errorMsg) ||
    isContentFilterBlockMessage(errorMsg)
  );
}

function buildRateLimitExhaustedEvent(model: {
  api?: string;
  provider?: string;
  id?: string;
}): AssistantMessageEvent {
  return {
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
  } as AssistantMessageEvent;
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
 * Wrap a StreamFn to proactively rotate API keys on each model call and:
 *  - retry with the next key on rate-limit (429 / 503 overloaded) errors
 *  - retry with the SAME key (up to MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE)
 *    on content-filter blocks before rotating to the next profile, so single-
 *    profile users still get recovery attempts on non-deterministic Gemini
 *    PROHIBITED_CONTENT / SAFETY blocks.
 *
 * Must be the outermost wrapper (after wrapStreamFnWithBuffer) so that
 * inner wrappers (non-streaming, buffer) guarantee no partial events on error.
 */
export function wrapStreamFnWithKeyRotation(params: WrapStreamFnWithKeyRotationParams): StreamFn {
  const { streamFn, profileCandidates, resolveApiKey, authStore, agentDir, rotationState } = params;

  return (model, context, options) => {
    if (!canSafelyRotate(model)) {
      return streamFn(model, context, options);
    }

    const outputStream = createAssistantMessageEventStream();

    void (async () => {
      // Reset exhaustion flag at the start of each model call.
      rotationState.allKeysExhausted = false;

      // Build two lists: available (not in cooldown, not skipped) and in-cooldown.
      // We try available first, then in-cooldown, so every profile gets a chance
      // before we report "all exhausted". Previously we skipped in-cooldown but
      // still counted them as attempts, so we could exhaust after trying only 2–3 keys.
      const available: string[] = [];
      const inCooldown: string[] = [];
      for (const p of profileCandidates) {
        if (typeof p !== "string" || rotationState.skipProfiles.has(p)) {
          continue;
        }
        if (isProfileInCooldown(authStore, p)) {
          inCooldown.push(p);
        } else {
          available.push(p);
        }
      }
      let toTryOrder = [...available, ...inCooldown];
      // Round-robin: prefer the profile at rotationIndex first on this call.
      const preferred = profileCandidates[rotationState.rotationIndex];
      if (typeof preferred === "string" && toTryOrder.includes(preferred)) {
        toTryOrder = [preferred, ...toTryOrder.filter((p) => p !== preferred)];
      }
      if (toTryOrder.length === 0) {
        rotationState.allKeysExhausted = true;
        log.warn("[key-rotation] no profile candidates available (all in cooldown or skipped).");
        outputStream.push(buildRateLimitExhaustedEvent(model));
        outputStream.end();
        return;
      }

      let triedCount = 0;
      for (const candidate of toTryOrder) {
        let apiKey: string | undefined;
        try {
          apiKey = await resolveApiKey(candidate);
        } catch {
          log.warn(`[key-rotation] failed to resolve API key for profile ${candidate}; skipping`);
          continue;
        }

        rotationState.lastProfileId = candidate;
        void stampProfileLastUsed({
          store: authStore,
          profileId: candidate,
          agentDir,
        });

        const callOptions = {
          ...options,
          ...(apiKey ? { apiKey } : {}),
          maxRetries: 0,
        };

        // Per-profile content-filter retry loop: Gemini's PROHIBITED_CONTENT /
        // SAFETY classifier is non-deterministic (especially on output-side
        // blocks), so the same key often succeeds on a retry. For rate-limit
        // and overloaded errors we still skip straight to rotation — retrying
        // the same key on 429 is counterproductive.
        let contentFilterAttempts = 0;
        let advanceProfile = false;
        let terminated = false;

        while (true) {
          try {
            const innerStream = await Promise.resolve(streamFn(model, context, callOptions));

            const events: AssistantMessageEvent[] = [];
            let rateLimitEvent: AssistantMessageEvent | null = null;
            let contentFilterEvent: AssistantMessageEvent | null = null;

            for await (const event of innerStream) {
              if (isRetriableErrorEvent(event)) {
                const msg =
                  (event as { error?: { errorMessage?: string } }).error?.errorMessage ?? "";
                if (isContentFilterBlockMessage(msg)) {
                  contentFilterEvent = event;
                } else {
                  rateLimitEvent = event;
                }
              } else {
                events.push(event);
              }
            }

            if (contentFilterEvent) {
              contentFilterAttempts++;
              const errMsg =
                (contentFilterEvent as { error?: { errorMessage?: string } }).error?.errorMessage ??
                "content filter";
              if (contentFilterAttempts < MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE) {
                log.warn(
                  `[key-rotation] profile ${candidate} hit content filter: ${errMsg.slice(0, 120)}; retrying same key (attempt ${contentFilterAttempts + 1}/${MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE})`,
                );
                continue;
              }
              triedCount++;
              log.warn(
                `[key-rotation] profile ${candidate} exhausted ${MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE} content-filter attempts; trying next key (${triedCount}/${toTryOrder.length} tried)`,
              );
              setCooldownInMemory(authStore, candidate);
              rotationState.skipProfiles.add(candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "empty_response",
                agentDir,
              });
              advanceProfile = true;
              break;
            }

            if (rateLimitEvent) {
              triedCount++;
              const errMsg =
                (rateLimitEvent as { error?: { errorMessage?: string } }).error?.errorMessage ??
                "rate limited";
              log.warn(
                `[key-rotation] profile ${candidate} hit rate limit: ${errMsg.slice(0, 120)}; trying next key (${triedCount}/${toTryOrder.length} tried)`,
              );
              setCooldownInMemory(authStore, candidate);
              rotationState.skipProfiles.add(candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "rate_limit",
                agentDir,
              });
              advanceProfile = true;
              break;
            }

            // Success: replay collected events.
            for (const event of events) {
              outputStream.push(event);
            }
            outputStream.end();
            const nextIndex = (profileCandidates.indexOf(candidate) + 1) % profileCandidates.length;
            rotationState.rotationIndex = nextIndex;
            terminated = true;
            break;
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);
            if (isContentFilterBlockMessage(errMsg)) {
              contentFilterAttempts++;
              if (contentFilterAttempts < MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE) {
                log.warn(
                  `[key-rotation] profile ${candidate} threw content filter: ${errMsg.slice(0, 120)}; retrying same key (attempt ${contentFilterAttempts + 1}/${MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE})`,
                );
                continue;
              }
              triedCount++;
              log.warn(
                `[key-rotation] profile ${candidate} exhausted ${MAX_CONTENT_FILTER_ATTEMPTS_PER_PROFILE} content-filter attempts; trying next key (${triedCount}/${toTryOrder.length} tried)`,
              );
              setCooldownInMemory(authStore, candidate);
              rotationState.skipProfiles.add(candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "empty_response",
                agentDir,
              });
              advanceProfile = true;
              break;
            }
            if (isRateLimitErrorMessage(errMsg) || isOverloadedErrorMessage(errMsg)) {
              triedCount++;
              log.warn(
                `[key-rotation] profile ${candidate} threw rate limit: ${errMsg.slice(0, 120)}; trying next key (${triedCount}/${toTryOrder.length} tried)`,
              );
              setCooldownInMemory(authStore, candidate);
              rotationState.skipProfiles.add(candidate);
              void markAuthProfileFailure({
                store: authStore,
                profileId: candidate,
                reason: "rate_limit",
                agentDir,
              });
              advanceProfile = true;
              break;
            }
            // Non-retriable error: propagate and stop.
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
            terminated = true;
            break;
          }
        }

        if (terminated) {
          return;
        }
        if (advanceProfile) {
          continue;
        }
      }

      // Only after we actually tried every candidate (available + in-cooldown) and all 429.
      rotationState.allKeysExhausted = true;
      const skipCount = rotationState.skipProfiles.size;
      const skipSuffix = skipCount > 0 ? ` (${skipCount} skipped for empty response)` : "";
      log.warn(
        `[key-rotation] all ${toTryOrder.length} profile keys exhausted due to rate limiting after ${triedCount} attempts${skipSuffix}`,
      );

      rotationState.rotationIndex =
        (profileCandidates.indexOf(toTryOrder[toTryOrder.length - 1]) + 1) %
        profileCandidates.length;

      outputStream.push(buildRateLimitExhaustedEvent(model));
      outputStream.end();
    })();

    return outputStream;
  };
}
