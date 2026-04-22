import { describe, expect, it } from "vitest";
import { stripThinkBlock, stripThinkFromAssistantMessage } from "./index.ts";

describe("nsfw stripThinkBlock", () => {
  it("removes a simple <think>…</think> block and trailing newline", () => {
    const input = "<think>\nplan\n</think>\nreply";
    expect(stripThinkBlock(input)).toBe("reply");
  });

  it("is greedy from the first <think> to the last </think>", () => {
    const input = "pre<think>a</think>mid<think>b</think>post";
    // User spec: find first <think> and last </think>, remove that span entirely.
    expect(stripThinkBlock(input)).toBe("prepost");
  });

  it("leaves content untouched when only an opening tag is present", () => {
    const input = "<think>unclosed ever";
    expect(stripThinkBlock(input)).toBe(input);
  });

  it("leaves content untouched when there is no think block", () => {
    const input = "plain reply text";
    expect(stripThinkBlock(input)).toBe(input);
  });

  it("strips with interleaved <final>/<disclaimer> tags still preserved", () => {
    const input = "<think>scaffold</think>\n<final>hello</final>\n<disclaimer>d</disclaimer>";
    expect(stripThinkBlock(input)).toBe("<final>hello</final>\n<disclaimer>d</disclaimer>");
  });
});

describe("nsfw stripThinkFromAssistantMessage", () => {
  it("rewrites string content", () => {
    const msg = {
      role: "assistant" as const,
      content: "<think>x</think>reply",
    };
    const out = stripThinkFromAssistantMessage(msg as unknown as Record<string, unknown>);
    expect(out).toEqual({ role: "assistant", content: "reply" });
  });

  it("rewrites text blocks and leaves thinking blocks untouched", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "thinking", thinking: "native hidden reasoning — keep" },
        { type: "text", text: "<think>apex scaffold</think>actual" },
      ],
    };
    const out = stripThinkFromAssistantMessage(msg as unknown as Record<string, unknown>);
    const blocks = (out as { content: Array<Record<string, unknown>> }).content;
    expect(blocks[0]).toEqual({
      type: "thinking",
      thinking: "native hidden reasoning — keep",
    });
    expect(blocks[1]).toEqual({ type: "text", text: "actual" });
  });

  it("walks toolCall arguments string fields", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        {
          type: "toolCall",
          id: "t1",
          name: "message",
          arguments: {
            to: "uuid:abc",
            message: "<think>scaffold</think>hi there",
          },
        },
      ],
    };
    const out = stripThinkFromAssistantMessage(msg as unknown as Record<string, unknown>);
    const block = (out as { content: Array<Record<string, unknown>> }).content[0];
    expect(block).toMatchObject({
      type: "toolCall",
      name: "message",
      arguments: { to: "uuid:abc", message: "hi there" },
    });
  });

  it("returns null when nothing changes", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "text", text: "plain reply" },
        { type: "thinking", thinking: "x" },
      ],
    };
    expect(stripThinkFromAssistantMessage(msg as unknown as Record<string, unknown>)).toBeNull();
  });
});
