/**
 * Rolling-window stripper for old `<relevant-memories>` injections in user
 * messages.
 *
 * Why this exists:
 *   `before_prompt_build` prepends each turn's retrieved memories onto the
 *   user message via `prependContext`. Pi SDK writes that full prepended
 *   text into the session (both the in-memory `_state.messages` and the
 *   `.jsonl` session file), so over many turns the transcript accumulates
 *   stale `<relevant-memories>` blocks.
 *
 *   Under Anthropic-style prompt caching this is actually *good* while the
 *   cache is warm — historical messages stay byte-identical and are fully
 *   cached. The waste only shows up once the cache expires (default 5 min
 *   idle TTL): at that point the full transcript has to be re-sent, and
 *   every old injection is pure dead weight.
 *
 *   Strategy: when the gap since the previous LLM call exceeds the idle
 *   threshold (i.e. the cache would have expired anyway), strip
 *   `<relevant-memories>` blocks from all user messages **older** than the
 *   N most recent, and let the current turn's injection proceed as usual.
 *   No additional cache penalty, direct token savings on the re-write.
 *
 *   During dense back-to-back turns (<threshold) this function is a no-op,
 *   so the warm cache is preserved.
 *
 * Scope:
 *   Mutates the provided `messages` array in place. The `.jsonl` session
 *   file retains the original (unstripped) messages — good for audit. Pi
 *   SDK's in-memory history is the reference passed to `before_prompt_build`,
 *   so mutating here is what the LLM actually sees on the next call.
 */
import type { Message, UserMessage } from "@mariozechner/pi-ai";

const MEMORY_OPEN = "<relevant-memories>";
const MEMORY_CLOSE = "</relevant-memories>";

/**
 * Remove every `<relevant-memories>...</relevant-memories>` block plus any
 * trailing blank lines that the injection format adds (`...</relevant-memories>\n\n`).
 * Leaves the rest of the text intact.
 */
export function stripRelevantMemoriesFromText(text: string): string {
  if (!text || !text.includes(MEMORY_OPEN)) {
    return text;
  }
  let result = "";
  let cursor = 0;
  while (cursor < text.length) {
    const openIdx = text.indexOf(MEMORY_OPEN, cursor);
    if (openIdx < 0) {
      result += text.slice(cursor);
      break;
    }
    const closeIdx = text.indexOf(MEMORY_CLOSE, openIdx + MEMORY_OPEN.length);
    if (closeIdx < 0) {
      // Unterminated block — leave it alone to avoid eating real content.
      result += text.slice(cursor);
      break;
    }
    result += text.slice(cursor, openIdx);
    cursor = closeIdx + MEMORY_CLOSE.length;
    // Swallow up to 2 trailing newlines that the injection format adds.
    let consumed = 0;
    while (consumed < 2 && cursor < text.length && text[cursor] === "\n") {
      cursor++;
      consumed++;
    }
  }
  return result;
}

function stripFromUserMessage(msg: UserMessage): boolean {
  if (typeof msg.content === "string") {
    if (!msg.content.includes(MEMORY_OPEN)) {
      return false;
    }
    const next = stripRelevantMemoriesFromText(msg.content);
    if (next === msg.content) {
      return false;
    }
    msg.content = next;
    return true;
  }
  if (!Array.isArray(msg.content)) {
    return false;
  }
  let changed = false;
  for (const part of msg.content) {
    if (
      part &&
      typeof part === "object" &&
      (part as { type?: unknown }).type === "text" &&
      typeof (part as { text?: unknown }).text === "string"
    ) {
      const textPart = part as { type: "text"; text: string };
      if (!textPart.text.includes(MEMORY_OPEN)) {
        continue;
      }
      const next = stripRelevantMemoriesFromText(textPart.text);
      if (next !== textPart.text) {
        textPart.text = next;
        changed = true;
      }
    }
  }
  return changed;
}

export interface StripOldInjectionsOptions {
  /** Mutable reference to the session messages array. */
  messages: Message[];
  /** Keep the most-recent N user messages' injections intact. */
  keepRecent: number;
}

export interface StripOldInjectionsResult {
  /** Number of user messages whose text was actually modified. */
  stripped: number;
  /** Number of user messages considered eligible for stripping (for logging). */
  scanned: number;
}

/**
 * Strip `<relevant-memories>` blocks from user messages older than the
 * `keepRecent` most-recent user messages. Mutates `messages` in place.
 */
export function stripOldInjections(opts: StripOldInjectionsOptions): StripOldInjectionsResult {
  const { messages, keepRecent } = opts;
  if (!Array.isArray(messages) || messages.length === 0 || keepRecent < 0) {
    return { stripped: 0, scanned: 0 };
  }

  // Indices of user messages, oldest-first.
  const userIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i] as { role?: unknown } | undefined;
    if (m && m.role === "user") {
      userIndices.push(i);
    }
  }

  if (userIndices.length <= keepRecent) {
    return { stripped: 0, scanned: 0 };
  }

  // Everything except the last `keepRecent` entries.
  const eligible = userIndices.slice(0, userIndices.length - keepRecent);
  let stripped = 0;
  for (const idx of eligible) {
    if (stripFromUserMessage(messages[idx] as UserMessage)) {
      stripped++;
    }
  }
  return { stripped, scanned: eligible.length };
}
