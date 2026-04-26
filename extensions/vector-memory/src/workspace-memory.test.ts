import { describe, expect, it } from "vitest";
import {
  stripIndexNoise,
  stripRecallBoilerplate,
  stripRecallNoiseForPrompt,
} from "./workspace-memory.js";

const IMAGE_RESEND_BOILERPLATE =
  "To send an image back, prefer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image. fer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image.jpg (spaces ok, quote if needed) or a safe relative path like MEDIA:./image.jpg. Avoid absolute paths (MEDIA:/...) and ~ paths — they are blocked for security. Keep caption in the text body.";

describe("stripRecallBoilerplate", () => {
  it("removes duplicated image resend instructions before rerank", () => {
    const input = `user: [historical-media-ref; path-preserved-for-resend: /tmp/old.jpg]
${IMAGE_RESEND_BOILERPLATE}
assistant: kept memory content`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("/tmp/old.jpg");
    expect(out).toContain("assistant: kept memory content");
    expect(out).not.toContain("To send an image back");
    expect(out).not.toContain("MEDIA:https://example.com/image.jpg");
  });

  it("handles the non-duplicated canonical wording", () => {
    const input =
      "prefix To send an image back, prefer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image.jpg (spaces ok, quote if needed) or a safe relative path like MEDIA:./image.jpg. Avoid absolute paths (MEDIA:/...) and ~ paths — they are blocked for security. Keep caption in the text body. suffix";

    expect(stripRecallBoilerplate(input)).toBe("prefix  suffix");
  });
});

describe("stripIndexNoise", () => {
  it("removes nested relevant-memories blocks from indexed daily journals", () => {
    const input = `before
<relevant-memories>
  <from-daily-memory>
    stale recalled memory
  </from-daily-memory>
</relevant-memories>
after`;

    const out = stripIndexNoise(input);

    expect(out).not.toContain("<relevant-memories>");
    expect(out).not.toContain("stale recalled memory");
    expect(out).toContain("before");
    expect(out).toContain("after");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
  });

  it("removes image resend boilerplate while preserving line count", () => {
    const input = `line 1
${IMAGE_RESEND_BOILERPLATE}
line 3`;

    const out = stripIndexNoise(input);

    expect(out).not.toContain("To send an image back");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
    expect(out).toContain("line 1");
    expect(out).toContain("line 3");
  });
});

describe("stripRecallNoiseForPrompt", () => {
  it("removes nested relevant memories and resend boilerplate before rerank", () => {
    const input = `intro
<relevant-memories>
  <from-vector-memory>
    stale recalled memory
  </from-vector-memory>
</relevant-memories>
path: /tmp/old.jpg
${IMAGE_RESEND_BOILERPLATE}
real memory`;

    const out = stripRecallNoiseForPrompt(input);

    expect(out).toContain("intro");
    expect(out).toContain("path: /tmp/old.jpg");
    expect(out).toContain("real memory");
    expect(out).not.toContain("<relevant-memories>");
    expect(out).not.toContain("stale recalled memory");
    expect(out).not.toContain("To send an image back");
  });
});
