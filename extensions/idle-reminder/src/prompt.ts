import type { StoredMessage } from "./state.js";

function formatMessageTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("zh-TW", {
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
}

const FOLLOW_UP_INSTRUCTION =
  "Check if there's anything you should follow up on with the user. " +
  "Say it, then DO it — words without action = nothing happened. " +
  "If nothing to follow up, reply only NO_REPLY.";

export function buildIdleReminderPrompt(messages: StoredMessage[], agentName?: string): string {
  const recent = messages.filter((m) => m.text.trim()).slice(-3);
  if (recent.length === 0) {
    return FOLLOW_UP_INSTRUCTION;
  }
  const lines = recent.map((m) => {
    const roleLabel = m.role === "user" ? "USER" : (agentName?.toUpperCase() ?? "AGENT");
    const time = formatMessageTime(m.timestamp);
    return `${roleLabel} (${time})\n${m.text.trim()}`;
  });
  const replyBlock = `Here are the previous last ${recent.length} messages:\n"""\n${lines.join("\n\n")}\n"""\n\n`;
  return `${replyBlock}${FOLLOW_UP_INSTRUCTION}`;
}
