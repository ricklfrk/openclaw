import { describe, expect, it } from "vitest";
import { buildIdleReminderPrompt } from "./prompt.js";
import type { StoredMessage } from "./state.js";

describe("buildIdleReminderPrompt", () => {
  it("returns a bare follow-up instruction when no messages", () => {
    const prompt = buildIdleReminderPrompt([]);
    expect(prompt).toContain("NO_REPLY");
    expect(prompt).not.toContain("USER");
    expect(prompt).not.toContain("AGENT");
  });

  it("includes recent messages with role labels", () => {
    const messages: StoredMessage[] = [
      { role: "user", text: "hello there", timestamp: Date.now() },
      { role: "agent", text: "hi!", timestamp: Date.now() },
    ];
    const prompt = buildIdleReminderPrompt(messages);
    expect(prompt).toContain("USER");
    expect(prompt).toContain("AGENT");
    expect(prompt).toContain("hello there");
    expect(prompt).toContain("hi!");
  });

  it("uses custom agent name when provided", () => {
    const messages: StoredMessage[] = [{ role: "agent", text: "response", timestamp: Date.now() }];
    const prompt = buildIdleReminderPrompt(messages, "Pi");
    expect(prompt).toContain("PI");
    expect(prompt).not.toContain("AGENT");
  });

  it("limits to last 3 messages", () => {
    const messages: StoredMessage[] = [
      { role: "user", text: "msg1", timestamp: 1000 },
      { role: "agent", text: "msg2", timestamp: 2000 },
      { role: "user", text: "msg3", timestamp: 3000 },
      { role: "agent", text: "msg4", timestamp: 4000 },
    ];
    const prompt = buildIdleReminderPrompt(messages);
    expect(prompt).not.toContain("msg1");
    expect(prompt).toContain("msg2");
    expect(prompt).toContain("msg3");
    expect(prompt).toContain("msg4");
  });

  it("skips empty-text messages", () => {
    const messages: StoredMessage[] = [
      { role: "user", text: "", timestamp: 1000 },
      { role: "agent", text: "  ", timestamp: 2000 },
      { role: "user", text: "real message", timestamp: 3000 },
    ];
    const prompt = buildIdleReminderPrompt(messages);
    expect(prompt).toContain("real message");
    expect(prompt).toContain("last 1 messages");
  });
});
