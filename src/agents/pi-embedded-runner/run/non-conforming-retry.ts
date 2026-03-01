/**
 * Non-conforming output retry for enforceFinalTag mode.
 *
 * When the model produces substantial text without <final> tags the output
 * is suppressed (assistantTexts stays empty). This module detects that case
 * and either returns a follow-up prompt for retry or, after exhausting
 * retries, falls back to delivering the raw text directly.
 */
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { isSilentReplyText } from "../../../auto-reply/tokens.js";
import { extractAssistantText } from "../../pi-embedded-utils.js";
import { log } from "../logger.js";

const MIN_TEXT_LENGTH = 20;
const MAX_RETRIES = 3;

const NON_CONFORMING_FOLLOWUP_PROMPT =
  "[System: Your previous response was not visible to the user because " +
  "it was not wrapped in the required tags. The user cannot see what you wrote. " +
  "Please respond again using the correct format:\n" +
  "- ALL internal reasoning MUST be inside `<think>...</think>`.\n" +
  "- Only the final user-visible reply may appear inside `<final>...</final>`.\n" +
  "- Only text inside `<final>` is shown to the user; everything else is discarded.\n" +
  "- Format: `<think>reasoning here</think>` then `<final>reply here</final>`, with no other text.\n" +
  "Example:\n" +
  "<think>Short internal reasoning.</think>\n" +
  "<final>Hey there! What would you like to do next?</final>\n" +
  "Respond now with the correct format.]";

export type NonConformingResult =
  | { action: "retry"; prompt: string }
  | { action: "fallback"; text: string }
  | null;

/**
 * Check whether the last attempt produced non-conforming output that
 * warrants an automatic retry or a fallback delivery.
 *
 * - retryCount < MAX_RETRIES → `{ action: "retry", prompt }` (re-prompt the model)
 * - retryCount >= MAX_RETRIES → `{ action: "fallback", text }` (deliver raw text)
 * - otherwise → `null` (no action needed)
 */
export function checkNonConformingOutput(params: {
  enforceFinalTag: boolean;
  retryCount: number;
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
      `stopReason=${params.lastAssistant?.stopReason} retryCount=${params.retryCount}`,
  );

  if (
    !params.enforceFinalTag ||
    params.aborted ||
    params.timedOut ||
    params.assistantTexts.length > 0 ||
    params.didSendViaMessagingTool ||
    params.lastAssistant?.stopReason !== "stop"
  ) {
    return null;
  }

  const rawText = extractAssistantText(params.lastAssistant).trim();
  if (rawText.length <= MIN_TEXT_LENGTH || isSilentReplyText(rawText)) {
    log.info(`[non-conforming-check] skipped: rawText too short (${rawText.length}) or silent`);
    return null;
  }

  if (params.retryCount < MAX_RETRIES) {
    return { action: "retry", prompt: NON_CONFORMING_FOLLOWUP_PROMPT };
  }

  return { action: "fallback", text: rawText };
}
