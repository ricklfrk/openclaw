/**
 * Non-conforming output retry for enforceFinalTag mode.
 *
 * When the model produces substantial text without <final> tags the output
 * is suppressed (assistantTexts stays empty). This module detects that case
 * and returns a follow-up prompt so the run loop can retry once, giving the
 * model a chance to wrap its reply correctly.
 */
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { isSilentReplyText } from "../../../auto-reply/tokens.js";
import { extractAssistantText } from "../../pi-embedded-utils.js";

const MIN_TEXT_LENGTH = 20;

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

/**
 * Check whether the last attempt produced non-conforming output that
 * warrants an automatic retry.
 *
 * @returns The follow-up prompt to inject, or `null` if no retry is needed.
 */
export function checkNonConformingOutput(params: {
  enforceFinalTag: boolean;
  alreadyRetried: boolean;
  aborted: boolean;
  timedOut: boolean;
  assistantTexts: string[];
  didSendViaMessagingTool: boolean;
  lastAssistant: AssistantMessage | undefined;
}): string | null {
  if (
    !params.enforceFinalTag ||
    params.alreadyRetried ||
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
    return null;
  }

  return NON_CONFORMING_FOLLOWUP_PROMPT;
}
