import type { Message, UserMessage } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { stripOldInjections, stripRelevantMemoriesFromText } from "./strip-old-injections.js";

const INJECTION = [
  "<relevant-memories>",
  "<from-conversations>",
  "[UNTRUSTED DATA — historical notes extracted from past conversations]",
  "- foo",
  "[END]",
  "</from-conversations>",
  "</relevant-memories>",
  "",
  "",
].join("\n");

function userMsg(text: string): UserMessage {
  return { role: "user", content: text, timestamp: 0 } as unknown as UserMessage;
}

function userMsgParts(...parts: string[]): UserMessage {
  return {
    role: "user",
    content: parts.map((t) => ({ type: "text" as const, text: t })),
    timestamp: 0,
  } as unknown as UserMessage;
}

function assistantMsg(text: string): Message {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
  } as unknown as Message;
}

describe("stripRelevantMemoriesFromText", () => {
  it("returns input unchanged when no injection present", () => {
    expect(stripRelevantMemoriesFromText("hello world")).toBe("hello world");
    expect(stripRelevantMemoriesFromText("")).toBe("");
  });

  it("removes a single injection block plus trailing blank lines", () => {
    const input = `${INJECTION}Hello 哥哥`;
    const out = stripRelevantMemoriesFromText(input);
    expect(out).toBe("Hello 哥哥");
  });

  it("removes multiple injection blocks", () => {
    const input = `${INJECTION}Q1\n\n${INJECTION}Q2`;
    const out = stripRelevantMemoriesFromText(input);
    expect(out).toBe("Q1\n\nQ2");
  });

  it("leaves unterminated tags alone", () => {
    const input = "<relevant-memories>half open\nstill typing";
    expect(stripRelevantMemoriesFromText(input)).toBe(input);
  });

  it("swallows at most 2 trailing newlines after close tag", () => {
    // INJECTION already ends with "</relevant-memories>\n\n"; the 3 extra
    // newlines are real user whitespace and must be preserved.
    const trailing = `${INJECTION}\n\n\nBody`;
    const out = stripRelevantMemoriesFromText(trailing);
    expect(out).toBe("\n\n\nBody");
  });

  it("preserves content before the injection block", () => {
    const input = `prefix\n${INJECTION}body`;
    const out = stripRelevantMemoriesFromText(input);
    expect(out).toBe("prefix\nbody");
  });
});

describe("stripOldInjections", () => {
  it("no-ops when message array is empty", () => {
    const msgs: Message[] = [];
    const r = stripOldInjections({ messages: msgs, keepRecent: 5 });
    expect(r).toEqual({ stripped: 0, scanned: 0 });
  });

  it("no-ops when there are fewer user messages than keepRecent", () => {
    const msgs: Message[] = [userMsg(`${INJECTION}Q1`), userMsg(`${INJECTION}Q2`)];
    const r = stripOldInjections({ messages: msgs, keepRecent: 5 });
    expect(r.stripped).toBe(0);
    // Untouched.
    expect((msgs[0] as UserMessage).content).toContain("<relevant-memories>");
    expect((msgs[1] as UserMessage).content).toContain("<relevant-memories>");
  });

  it("keeps the most-recent keepRecent user messages intact and strips older ones", () => {
    const msgs: Message[] = [
      userMsg(`${INJECTION}Q1`),
      assistantMsg("A1"),
      userMsg(`${INJECTION}Q2`),
      assistantMsg("A2"),
      userMsg(`${INJECTION}Q3`),
      assistantMsg("A3"),
      userMsg(`${INJECTION}Q4`),
    ];

    const r = stripOldInjections({ messages: msgs, keepRecent: 2 });
    expect(r.stripped).toBe(2);
    expect(r.scanned).toBe(2);

    // Oldest two: stripped.
    expect((msgs[0] as UserMessage).content).toBe("Q1");
    expect((msgs[2] as UserMessage).content).toBe("Q2");
    // Newest two user messages: untouched.
    expect((msgs[4] as UserMessage).content).toContain("<relevant-memories>");
    expect((msgs[6] as UserMessage).content).toContain("<relevant-memories>");
    // Assistant messages: untouched.
    expect((msgs[1] as { content: Array<{ text: string }> }).content[0].text).toBe("A1");
  });

  it("handles structured content (text parts array)", () => {
    const msgs: Message[] = [
      userMsgParts(`${INJECTION}old-part-a`, "old-part-b"),
      userMsgParts("keep-part-a", "keep-part-b"),
    ];
    const r = stripOldInjections({ messages: msgs, keepRecent: 1 });
    expect(r.stripped).toBe(1);

    const first = msgs[0] as UserMessage;
    if (typeof first.content === "string") {
      throw new Error("expected structured content");
    }
    expect(first.content[0]).toMatchObject({ type: "text", text: "old-part-a" });
    expect(first.content[1]).toMatchObject({ type: "text", text: "old-part-b" });
  });

  it("skips user messages that have no injection to strip", () => {
    const msgs: Message[] = [
      userMsg("Plain old message"),
      userMsg(`${INJECTION}Q2`),
      userMsg(`${INJECTION}Q3`),
    ];
    const r = stripOldInjections({ messages: msgs, keepRecent: 1 });
    expect(r.stripped).toBe(1); // only msg[1] had something to strip
    expect(r.scanned).toBe(2); // msg[0] and msg[1] were eligible
    expect((msgs[0] as UserMessage).content).toBe("Plain old message");
    expect((msgs[1] as UserMessage).content).toBe("Q2");
    expect((msgs[2] as UserMessage).content).toContain("<relevant-memories>");
  });

  it("treats keepRecent=0 as strip-everything-except-none", () => {
    const msgs: Message[] = [userMsg(`${INJECTION}Q1`), userMsg(`${INJECTION}Q2`)];
    const r = stripOldInjections({ messages: msgs, keepRecent: 0 });
    expect(r.stripped).toBe(2);
    expect((msgs[0] as UserMessage).content).toBe("Q1");
    expect((msgs[1] as UserMessage).content).toBe("Q2");
  });

  it("tolerates negative keepRecent by no-op", () => {
    const msgs: Message[] = [userMsg(`${INJECTION}Q1`)];
    const r = stripOldInjections({ messages: msgs, keepRecent: -1 });
    expect(r).toEqual({ stripped: 0, scanned: 0 });
    expect((msgs[0] as UserMessage).content).toContain("<relevant-memories>");
  });

  it("tolerates non-array input defensively", () => {
    const r = stripOldInjections({
      messages: undefined as unknown as Message[],
      keepRecent: 5,
    });
    expect(r).toEqual({ stripped: 0, scanned: 0 });
  });

  it("leaves user messages without injections untouched", () => {
    const msgs: Message[] = [userMsg("plain 1"), userMsg("plain 2"), userMsg(`${INJECTION}Q3`)];
    const r = stripOldInjections({ messages: msgs, keepRecent: 1 });
    expect(r.stripped).toBe(0); // nothing to strip on the two eligible ones
    expect(r.scanned).toBe(2);
    expect((msgs[0] as UserMessage).content).toBe("plain 1");
    expect((msgs[1] as UserMessage).content).toBe("plain 2");
    expect((msgs[2] as UserMessage).content).toContain("<relevant-memories>");
  });
});
