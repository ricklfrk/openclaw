/**
 * Non-conforming output handling for enforceFinalTag mode.
 *
 * When the model produces output that doesn't conform to the expected format
 * (missing <final> tags, thinking-only, or empty), this module detects it and
 * signals the appropriate action.
 *
 * All types share the same escalation for key/profile rotation:
 *   attempt 0 → `fail` (retry same profile — exhaust same-profile keys first)
 *   attempt 1+ → `fail` + skipProfile (move to next profile)
 *
 * Non-conforming text can carry a stripped raw-text fallback candidate, but
 * the caller should only deliver it after all profiles for this turn are
 * exhausted. This guarantees every profile/key path gets one chance first.
 */
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { isSilentReplyText } from "../../../auto-reply/tokens.js";
import { extractAssistantText, extractAssistantThinking } from "../../pi-embedded-utils.js";
import { log } from "../logger.js";

const MIN_TEXT_LENGTH = 20;

/**
 * Backtick-wrapped tag references like `<think>`, `</think>`, `<final>`, `</final>`.
 * These appear when the model reasons about its own output format and accidentally
 * leaks that reasoning into the <final> block.  A genuine user-facing reply or
 * kaomoji would never contain exactly this pattern.
 */
const LEAKED_TAG_REF_RE = /`<\/?(?:think|final)>`/;

/**
 * Detects the "split backtick span" pattern: the model wrote `</think>` or `<final>`
 * inside <think>, the parser closed the tag early, and the trailing backtick+punctuation
 * ended up as the first character(s) of the <final> content.
 *
 * A legitimate reply never starts with `` `, `` or `` `) `` — those are exclusively
 * the closing tail of a backtick code-span that was split by a tag boundary.
 */
const LEAKED_SPLIT_BACKTICK_RE = /^`[,)\s]/;

/**
 * One-time format hint injected into the system prompt after the first
 * non-conforming failure.  Since the session is rolled back, the model
 * cannot see its previous wrong output — this hint nudges it toward
 * correct output structure on the re-roll.
 */
export const NON_CONFORMING_FORMAT_HINT = [
  "[Output format reminder — applies to EVERY assistant turn, including after tool results]",
  "",
  "AFTER A TOOL RESULT your very next turn MUST be one of:",
  "  (A) another native functionCall   — if more tools are needed",
  "  (B) <think>...</think><final>...</final>   — to reply to the user",
  "Outputting ONLY a <think> block after a tool result is INVALID.",
  "",
  "Tool call → native functionCall ONLY (NO wrapping in <think> or text):",
  '  functionCall { name: "tool_name", args: { "key": "value" } }',
  "",
  "Reply to user → MUST end with <final>:",
  "  <think>internal reasoning (optional)</think>",
  "  <final>visible reply to user</final>",
  "",
  "FORBIDDEN:",
  "  INVALID_REASON: historical_tool_call | thinking_only | reply_without_final",
  "  REQUIRED_NEXT_TURN: native_functionCall | <final>reply</final>",
].join("\n");

const THINK_OPEN_RE = /<\s*(?:think(?:ing)?|thought|antthinking)\s*>/i;
const THINK_CLOSE_RE = /<\s*\/\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
const FINAL_TAG_RE = /<\s*\/?\s*final\s*>/gi;

/**
 * Strip thinking content from fallback text using first-open / last-close
 * strategy. Unlike the streaming stripBlockTags (which toggles on each
 * close tag), this scans the complete text and uses the LAST </think> as
 * the true boundary — immune to premature </think> inside thinking text.
 */
export function stripThinkingForFallback(text: string): string {
  const openMatch = THINK_OPEN_RE.exec(text);
  if (!openMatch || openMatch.index === undefined) {
    return text.replace(FINAL_TAG_RE, "").trim();
  }

  THINK_CLOSE_RE.lastIndex = 0;
  let lastCloseEnd = -1;
  for (const m of text.matchAll(THINK_CLOSE_RE)) {
    lastCloseEnd = (m.index ?? 0) + m[0].length;
  }

  let result: string;
  if (lastCloseEnd > openMatch.index) {
    result = text.slice(0, openMatch.index) + text.slice(lastCloseEnd);
  } else {
    result = text.slice(0, openMatch.index);
  }

  return result.replace(FINAL_TAG_RE, "").trim();
}

export function isNonConformingRetryPrompt(text: string | undefined | null): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const trimmed = text.trim();
  return (
    trimmed.startsWith("[System: Your previous output was invalid and was not delivered.]") ||
    trimmed.startsWith(
      "[System: Your previous reply was not delivered because it was not wrapped in `<final>` tags.",
    ) ||
    trimmed.startsWith("[System: Your last output was only a thinking block")
  );
}

export type NonConformingResult = {
  action: "fail";
  skipProfile: boolean;
  fallbackText?: string;
} | null;

/**
 * Check whether the last attempt produced non-conforming or empty output.
 *
 * Key/profile escalation (shared by all types):
 *   !formatHintInjected → `fail` skipProfile=false
 *       (first failure — retry same profile after hint is injected)
 *   formatHintInjected → `fail` skipProfile=true
 *       (same profile already tried with hint — advance to next profile)
 *
 * Non-conforming text (has real content) returns a `fallbackText` candidate,
 * but the caller should deliver it only after all profiles are exhausted.
 *
 * Thinking-only / empty keep failing until all profiles exhausted →
 * FailoverError → model-level fallback.
 */
export function checkNonConformingOutput(params: {
  enforceFinalTag: boolean;
  /** Whether the format hint has already been injected (drives skipProfile). */
  formatHintInjected: boolean;
  aborted: boolean;
  timedOut: boolean;
  assistantTexts: string[];
  didSendViaMessagingTool: boolean;
  lastAssistant: AssistantMessage | undefined;
}): NonConformingResult {
  log.info(
    `[non-conforming-check] enforceFinalTag=${params.enforceFinalTag} aborted=${params.aborted} ` +
      `timedOut=${params.timedOut} assistantTexts=${params.assistantTexts.length} ` +
      `didSendViaTool=${params.didSendViaMessagingTool} ` +
      `stopReason=${params.lastAssistant?.stopReason} ` +
      `hintInjected=${params.formatHintInjected}`,
  );

  if (
    !params.enforceFinalTag ||
    params.aborted ||
    params.timedOut ||
    params.didSendViaMessagingTool ||
    params.lastAssistant?.stopReason !== "stop"
  ) {
    return null;
  }

  // If the model delivered text but it looks like leaked thinking, roll it back.
  // Two patterns are detected:
  //   1. Backtick-wrapped tag refs (e.g. `<final>`, `</think>`) anywhere in the text.
  //   2. "Split backtick span": model wrote `</think>` inside <think>; the parser
  //      closed the tag early and the trailing `` `, `` / `` `) `` became the first
  //      chars of the <final> content.  A real reply never starts with `` `, ``.
  if (params.assistantTexts.length > 0) {
    const hasLeakedTagRef = params.assistantTexts.some(
      (t) => LEAKED_TAG_REF_RE.test(t) || LEAKED_SPLIT_BACKTICK_RE.test(t),
    );
    if (!hasLeakedTagRef) {
      return null;
    }
    log.warn(
      `[non-conforming-check] <final> content looks like leaked thinking — rolled back; fail skipProfile=${params.formatHintInjected}`,
    );
    return { action: "fail", skipProfile: params.formatHintInjected };
  }

  // First failure → retry same profile (hint about to be injected).
  // After hint injected → skip profile (same profile already tried with hint).
  const skipProfile = params.formatHintInjected;

  const rawTextWithTags = extractAssistantText(params.lastAssistant).trim();
  const rawText = stripThinkingForFallback(rawTextWithTags);

  if (rawText.length === 0) {
    const thinkingText = extractAssistantThinking(params.lastAssistant).trim();
    if (thinkingText.length > 0) {
      log.warn(
        `[non-conforming-check] thinking-only (${thinkingText.length} chars); fail skipProfile=${skipProfile}`,
      );
    } else {
      log.warn(`[non-conforming-check] empty (0 tokens); fail skipProfile=${skipProfile}`);
    }
    return { action: "fail", skipProfile };
  }

  if (rawText.length <= MIN_TEXT_LENGTH || isSilentReplyText(rawText)) {
    log.info(`[non-conforming-check] skipped: rawText too short (${rawText.length}) or silent`);
    return null;
  }

  const stripped = stripThinkingForFallback(rawText);
  if (!stripped) {
    log.warn(
      "[non-conforming-check] fallback text empty after stripping thinking; signalling fail",
    );
    return { action: "fail", skipProfile };
  }

  log.info(
    `[non-conforming-check] non-conforming text (${stripped.length} chars); ` +
      `fail skipProfile=${skipProfile} with deferred fallback text`,
  );
  return { action: "fail", skipProfile, fallbackText: stripped };
}
