import { describe, expect, it } from "vitest";
import { applyRules, applyRulesToMessage, type Rule } from "./index.ts";

const noopLogger = { warn: () => {} };

const DISCLAIMER_RULES: Rule[] = [
  {
    pattern: "<disclaimer>[\\s\\S]*?</disclaimer>",
    flags: "gm",
    replacement: "",
  },
];

describe("regex-replace applyRules", () => {
  it("strips matched substring from a plain string", () => {
    const out = applyRules(
      "before<disclaimer>SECRET</disclaimer>after",
      DISCLAIMER_RULES,
      noopLogger,
    );
    expect(out).toBe("beforeafter");
  });

  it("passes through unchanged when no rule matches", () => {
    const out = applyRules("nothing to do", DISCLAIMER_RULES, noopLogger);
    expect(out).toBe("nothing to do");
  });

  it("warns and continues on invalid pattern", () => {
    const captured: string[] = [];
    const logger = { warn: (m: string) => captured.push(m) };
    const out = applyRules(
      "left<disclaimer>X</disclaimer>right",
      [{ pattern: "(unterminated", flags: "g", replacement: "!" }, ...DISCLAIMER_RULES],
      logger,
    );
    expect(out).toBe("leftright");
    expect(captured.some((msg) => msg.includes("invalid pattern"))).toBe(true);
  });
});

describe("regex-replace applyRulesToMessage", () => {
  it("transforms string content on user messages", () => {
    const msg = {
      role: "user" as const,
      content: "hello<disclaimer>X</disclaimer>world",
    };
    const out = applyRulesToMessage(
      msg as unknown as Record<string, unknown>,
      DISCLAIMER_RULES,
      noopLogger,
    );
    expect(out).toEqual({ role: "user", content: "helloworld" });
  });

  it("transforms text blocks on assistant messages", () => {
    const msg = {
      role: "assistant" as const,
      content: [{ type: "text", text: "a<disclaimer>b</disclaimer>c" }],
    };
    const out = applyRulesToMessage(
      msg as unknown as Record<string, unknown>,
      DISCLAIMER_RULES,
      noopLogger,
    );
    expect(out).toEqual({
      role: "assistant",
      content: [{ type: "text", text: "ac" }],
    });
  });

  it("walks toolCall arguments and scrubs string fields", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "thinking", thinking: "internal" },
        {
          type: "toolCall",
          id: "t1",
          name: "message",
          arguments: {
            to: "uuid:abc",
            message: "hi<disclaimer>[AI_SYSTEM]</disclaimer>there",
          },
        },
      ],
    };
    const out = applyRulesToMessage(
      msg as unknown as Record<string, unknown>,
      DISCLAIMER_RULES,
      noopLogger,
    );
    expect(out).not.toBeNull();
    const outMsg = out as { content: Array<Record<string, unknown>> };
    expect(outMsg.content[0]).toEqual({ type: "thinking", thinking: "internal" });
    expect(outMsg.content[1]).toMatchObject({
      type: "toolCall",
      id: "t1",
      name: "message",
      arguments: { to: "uuid:abc", message: "hithere" },
    });
  });

  it("preserves non-string toolCall argument values verbatim", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        {
          type: "toolCall",
          id: "t2",
          name: "message",
          arguments: {
            message: "<disclaimer>x</disclaimer>",
            count: 3,
            flags: { primary: true },
          },
        },
      ],
    };
    const out = applyRulesToMessage(
      msg as unknown as Record<string, unknown>,
      DISCLAIMER_RULES,
      noopLogger,
    );
    const outMsg = out as { content: Array<Record<string, unknown>> };
    expect(outMsg.content[0]).toMatchObject({
      arguments: { message: "", count: 3, flags: { primary: true } },
    });
  });

  it("returns null when neither text nor arguments change", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "text", text: "plain" },
        { type: "thinking", thinking: "x" },
      ],
    };
    const out = applyRulesToMessage(
      msg as unknown as Record<string, unknown>,
      DISCLAIMER_RULES,
      noopLogger,
    );
    expect(out).toBeNull();
  });
});
