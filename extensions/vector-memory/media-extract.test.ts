import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { extractMediaFromMessages } from "./index.js";

// A 1x1 PNG (67 bytes) — real bytes so the function can actually read from disk.
const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII=",
  "base64",
);

// Helper: build a user message with a content array mixing text + optional inline blocks.
function userMsg(parts: Array<{ type: string; [k: string]: unknown }>): unknown {
  return { role: "user", content: parts };
}

describe("extractMediaFromMessages", () => {
  let tmpDir: string;
  let imgPath: string;
  let imgPathJpg: string;
  let missingPath: string;

  beforeAll(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "vm-media-extract-"));
    imgPath = join(tmpDir, "test.png");
    imgPathJpg = join(tmpDir, "photo.jpg");
    missingPath = join(tmpDir, "does-not-exist.png");
    writeFileSync(imgPath, PNG_1X1);
    // JPEG extension but PNG bytes is fine — we don't validate magic bytes here.
    writeFileSync(imgPathJpg, PNG_1X1);
  });

  afterAll(() => {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  });

  it("returns [] when no messages contain media", async () => {
    const results = await extractMediaFromMessages([
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ]);
    expect(results).toEqual([]);
  });

  it("extracts inline base64 image blocks (OpenClaw shape)", async () => {
    const results = await extractMediaFromMessages([
      userMsg([
        { type: "text", text: "look at this" },
        { type: "image", data: PNG_1X1.toString("base64"), mimeType: "image/png" },
      ]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mimeType).toBe("image/png");
    expect(results[0].data.equals(PNG_1X1)).toBe(true);
  });

  it("extracts inline base64 image blocks (Anthropic source shape)", async () => {
    const results = await extractMediaFromMessages([
      userMsg([
        { type: "text", text: "hi" },
        {
          type: "image",
          source: { type: "base64", media_type: "image/png", data: PNG_1X1.toString("base64") },
        },
      ]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mimeType).toBe("image/png");
  });

  it("reads from disk when only a single [media attached: /path (mime)] breadcrumb exists", async () => {
    const results = await extractMediaFromMessages([
      userMsg([{ type: "text", text: `check this [media attached: ${imgPath} (image/png)]` }]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mimeType).toBe("image/png");
    expect(results[0].mediaPath).toBe(imgPath);
    expect(results[0].data.equals(PNG_1X1)).toBe(true);
  });

  it("reads from disk for indexed multi-attachment breadcrumbs", async () => {
    const text = [
      "[media attached: 2 files]",
      `[media attached 1/2: ${imgPath} (image/png)]`,
      `[media attached 2/2: ${imgPathJpg} (image/jpeg)]`,
    ].join("\n");
    const results = await extractMediaFromMessages([userMsg([{ type: "text", text }])]);
    expect(results).toHaveLength(2);
    expect(results[0].mediaPath).toBe(imgPath);
    expect(results[0].mimeType).toBe("image/png");
    expect(results[1].mediaPath).toBe(imgPathJpg);
    expect(results[1].mimeType).toBe("image/jpeg");
  });

  it("infers MIME from file extension when breadcrumb omits it", async () => {
    const results = await extractMediaFromMessages([
      userMsg([{ type: "text", text: `[media attached: ${imgPath}]` }]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mimeType).toBe("image/png");
  });

  it("tolerates trailing |url suffix and still reads the file", async () => {
    const results = await extractMediaFromMessages([
      userMsg([
        {
          type: "text",
          text: `[media attached: ${imgPath} (image/png) | https://example.com/a.png]`,
        },
      ]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mediaPath).toBe(imgPath);
  });

  it("skips missing files without throwing", async () => {
    const warnings: string[] = [];
    const results = await extractMediaFromMessages(
      [userMsg([{ type: "text", text: `[media attached: ${missingPath}]` }])],
      { logDebug: (msg) => warnings.push(msg) },
    );
    expect(results).toEqual([]);
    expect(warnings.some((w) => w.includes("failed to read media"))).toBe(true);
  });

  it("skips attachments with unsupported extensions", async () => {
    const oddPath = join(tmpDir, "foo.xyz");
    writeFileSync(oddPath, PNG_1X1);
    const results = await extractMediaFromMessages([
      userMsg([{ type: "text", text: `[media attached: ${oddPath}]` }]),
    ]);
    expect(results).toEqual([]);
  });

  it("skips the '[media attached: N files]' summary line", async () => {
    // Summary-only text with no real path → nothing should be read.
    const results = await extractMediaFromMessages([
      userMsg([{ type: "text", text: "[media attached: 3 files]" }]),
    ]);
    expect(results).toEqual([]);
  });

  it("does not double-capture when an inline block accompanies a breadcrumb", async () => {
    // When both paths provide the same attachment, we should get exactly one result
    // (the inline block wins, and the breadcrumb hint is consumed to prevent a disk re-read).
    const results = await extractMediaFromMessages([
      userMsg([
        { type: "text", text: `[media attached: ${imgPath} (image/png)]` },
        { type: "image", data: PNG_1X1.toString("base64"), mimeType: "image/png" },
      ]),
    ]);
    expect(results).toHaveLength(1);
    expect(results[0].mediaPath).toBe(imgPath);
  });

  it("captures inline + disk when there are more breadcrumbs than inline blocks", async () => {
    const text = [
      `[media attached 1/2: ${imgPath} (image/png)]`,
      `[media attached 2/2: ${imgPathJpg} (image/jpeg)]`,
    ].join("\n");
    const results = await extractMediaFromMessages([
      userMsg([
        { type: "text", text },
        // Only one inline block provided; second attachment should fall back to disk.
        { type: "image", data: PNG_1X1.toString("base64"), mimeType: "image/png" },
      ]),
    ]);
    expect(results).toHaveLength(2);
    // Inline block gets paired with the first breadcrumb, fallback gets the second.
    const paths = results
      .map((r) => r.mediaPath)
      .toSorted((a, b) => (a ?? "").localeCompare(b ?? ""));
    expect(paths).toEqual([imgPath, imgPathJpg].toSorted((a, b) => a.localeCompare(b)));
  });

  it("caps attachments per turn at MAX_MEDIA_PER_TURN (4)", async () => {
    const extraPaths = [];
    for (let i = 0; i < 6; i++) {
      const p = join(tmpDir, `cap-${i}.png`);
      writeFileSync(p, PNG_1X1);
      extraPaths.push(p);
    }
    const text = [
      `[media attached: 6 files]`,
      ...extraPaths.map((p, i) => `[media attached ${i + 1}/6: ${p} (image/png)]`),
    ].join("\n");
    const results = await extractMediaFromMessages([userMsg([{ type: "text", text }])]);
    expect(results).toHaveLength(4);
  });

  it("only scans user messages (assistant attachments are ignored)", async () => {
    const results = await extractMediaFromMessages([
      {
        role: "assistant",
        content: [{ type: "text", text: `[media attached: ${imgPath} (image/png)]` }],
      },
    ]);
    expect(results).toEqual([]);
  });
});
