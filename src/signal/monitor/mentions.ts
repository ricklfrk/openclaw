import type { SignalMention } from "./event-handler.types.js";

const OBJECT_REPLACEMENT = "\uFFFC";

function isValidMention(mention: SignalMention | null | undefined): mention is SignalMention {
  if (!mention) {
    return false;
  }
  if (!(mention.uuid || mention.number)) {
    return false;
  }
  if (typeof mention.start !== "number" || Number.isNaN(mention.start)) {
    return false;
  }
  if (typeof mention.length !== "number" || Number.isNaN(mention.length)) {
    return false;
  }
  return mention.length > 0;
}

function clampBounds(start: number, length: number, textLength: number) {
  const safeStart = Math.max(0, Math.trunc(start));
  const safeLength = Math.max(0, Math.trunc(length));
  const safeEnd = Math.min(textLength, safeStart + safeLength);
  return { start: safeStart, end: safeEnd };
}

/**
 * When mentions array is missing/empty (e.g. signal-cli drops it), replace U+FFFC
 * with @fallbackIdentifier so the message displays as @bot instead of ￼.
 * Does not affect mention detection (requireMention still uses patterns/native only).
 */
function replacePlaceholderWithFallback(
  message: string,
  fallbackIdentifier: string | null | undefined,
): string {
  if (!message || !fallbackIdentifier?.trim() || !message.includes(OBJECT_REPLACEMENT)) {
    return message;
  }
  return message.split(OBJECT_REPLACEMENT).join(`@${fallbackIdentifier.trim()}`);
}

export function renderSignalMentions(
  message: string,
  mentions?: SignalMention[] | null,
  fallbackIdentifier?: string | null,
): string {
  if (!message) {
    return message;
  }
  if (mentions?.length) {
    let normalized = message;
    const candidates = mentions.filter(isValidMention).toSorted((a, b) => b.start! - a.start!);

    for (const mention of candidates) {
      const identifier = mention.uuid ?? mention.number;
      if (!identifier) {
        continue;
      }

      const { start, end } = clampBounds(mention.start!, mention.length!, normalized.length);
      if (start >= end) {
        continue;
      }
      const slice = normalized.slice(start, end);

      if (!slice.includes(OBJECT_REPLACEMENT)) {
        continue;
      }

      normalized = normalized.slice(0, start) + `@${identifier}` + normalized.slice(end);
    }
    return normalized;
  }
  return replacePlaceholderWithFallback(message, fallbackIdentifier);
}
