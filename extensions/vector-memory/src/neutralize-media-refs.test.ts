import { describe, expect, it } from "vitest";
import { neutralizeRecalledMediaRefs } from "./neutralize-media-refs.js";

describe("neutralizeRecalledMediaRefs", () => {
  it("rewrites [media attached: path (mime)] to inert form", () => {
    const input = "[media attached: /Users/mea/.openclaw/media/inbound/a.jpg (image/jpeg)]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out).not.toContain("[media attached:");
    expect(out).toContain("[archived-media-ref:");
    expect(out).toContain("/Users/mea/.openclaw/media/inbound/a.jpg");
    expect(out).toContain("image/jpeg");
  });

  it("rewrites indexed form [media attached N/M: ...]", () => {
    const input = "[media attached 2/3: /tmp/photo.png (image/png)]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out).toMatch(/^\[archived-media-ref 2\/3:/);
    expect(out).toContain("/tmp/photo.png");
  });

  it("rewrites gateway claim-check URIs", () => {
    const input = "[media attached: media://inbound/abc-123.png]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out).toBe("[archived-media-ref: media://inbound/abc-123.png]");
  });

  it("rewrites [Image: source: ...] to inert form", () => {
    const input = "[Image: source: /tmp/screenshot.png]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out).toBe("[archived-image-ref: /tmp/screenshot.png]");
  });

  it("rewrites multiple occurrences in the same string", () => {
    const input =
      "First [media attached: /a.jpg] and second [media attached: /b.png (image/png)] and image [Image: source: /c.gif]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out.match(/\[media attached:/g)).toBeNull();
    expect(out.match(/\[Image: source:/g)).toBeNull();
    expect(out.match(/\[archived-media-ref:/g)).toHaveLength(2);
    expect(out).toContain("[archived-image-ref:");
  });

  it("leaves ordinary text untouched", () => {
    const input = "Nothing special here, just plain recall text.";
    expect(neutralizeRecalledMediaRefs(input)).toBe(input);
  });

  it("handles empty string", () => {
    expect(neutralizeRecalledMediaRefs("")).toBe("");
  });

  it("is case-insensitive", () => {
    const input = "[MEDIA ATTACHED: /tmp/x.jpg]";
    const out = neutralizeRecalledMediaRefs(input);
    expect(out).not.toContain("MEDIA ATTACHED:");
    expect(out).toContain("/tmp/x.jpg");
  });

  it("preserves content that happens to contain the path but not the marker", () => {
    const input = "User mentioned /Users/mea/.openclaw/media/inbound/29b5731a.jpg once.";
    expect(neutralizeRecalledMediaRefs(input)).toBe(input);
  });
});
