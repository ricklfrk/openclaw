import { describe, expect, it } from "vitest";
import { buildAugmentedBm25Query, cleanRecallQuery, expandKeywordVariants } from "./query-clean.js";

describe("cleanRecallQuery", () => {
  it("returns empty for empty input", () => {
    expect(cleanRecallQuery("")).toBe("");
  });

  it("strips http/https URLs", () => {
    const out = cleanRecallQuery(
      "https://www.youtube.com/watch?v=abc123\n\n幫我看... 我查一下POCKET4哪裡有賣",
    );
    expect(out).toBe("幫我看... 我查一下POCKET4哪裡有賣");
  });

  it("strips fenced code blocks (covers envelope JSON in ```json fences)", () => {
    const raw = [
      "找到了",
      "九龍觀塘巧明街117號",
      "",
      "Conversation info (untrusted metadata):",
      "```json",
      '{ "chat_id": "+85264483210", "sender": "Ricklf" }',
      "```",
      "",
      "Sender (untrusted metadata):",
      "```json",
      '{ "label": "Ricklf" }',
      "```",
    ].join("\n");
    const out = cleanRecallQuery(raw);
    expect(out).not.toContain("chat_id");
    expect(out).not.toContain("```");
    expect(out).not.toContain("(untrusted metadata)");
    expect(out).toContain("找到了");
    expect(out).toContain("九龍觀塘巧明街117號");
  });

  it("strips <relevant-memories> blocks defensively", () => {
    const raw = `<relevant-memories>\nold stuff\n</relevant-memories>\n\nfresh question about POCKET4`;
    const out = cleanRecallQuery(raw);
    expect(out).not.toContain("old stuff");
    expect(out).toContain("POCKET4");
  });

  it("strips internal system markers", () => {
    const raw = "System: __openclaw_memory_core_short_term_promotion_dream__\n\n哥哥早安";
    const out = cleanRecallQuery(raw);
    expect(out).not.toContain("__openclaw");
    expect(out).toContain("哥哥早安");
  });

  it("normalizes NFKC (full-width digits/letters collapse)", () => {
    const raw = "ＰＯＣＫＥＴ４"; // full-width letters and digit
    const out = cleanRecallQuery(raw);
    expect(out).toBe("POCKET4");
  });

  it("collapses repeated whitespace", () => {
    expect(cleanRecallQuery("a    b\n\n\nc\t\td")).toBe("a b c d");
  });

  it("preserves CJK content untouched", () => {
    expect(cleanRecallQuery("人家想吃 Häagen-Dazs")).toBe("人家想吃 Häagen-Dazs");
  });
});

describe("expandKeywordVariants", () => {
  it("returns [] for empty input", () => {
    expect(expandKeywordVariants("")).toEqual([]);
  });

  it("splits ALL-CAPS+digit (POCKET4 → pocket4 + pocket 4 + parts)", () => {
    const v = expandKeywordVariants("POCKET4 哪裡有賣");
    expect(v).toContain("pocket4");
    expect(v).toContain("pocket 4");
    expect(v).toContain("pocket");
    expect(v).toContain("4");
  });

  it("splits camelCase / Pascal+digit (S24Ultra)", () => {
    const v = expandKeywordVariants("S24Ultra 螢幕");
    expect(v).toContain("s24ultra");
    // Split form: digit boundary then letter, then upper boundary
    expect(v.some((s) => s.includes("s 24") || s.includes("24 ultra"))).toBe(true);
    expect(v).toContain("ultra");
  });

  it("splits camelCase (iPhone)", () => {
    const v = expandKeywordVariants("iPhone 15");
    expect(v).toContain("iphone");
    expect(v).toContain("i phone");
    expect(v).toContain("phone");
  });

  it("does not generate variants for already-spaced canonical forms", () => {
    const v = expandKeywordVariants("DJI Pocket 4");
    expect(v).toContain("dji");
    expect(v).toContain("pocket");
    // No camel/digit boundary inside any token, so no extra split forms
    // beyond the base lowercased runs.
    expect(v).not.toContain("d j i");
  });

  it("ignores pure CJK (handled by retriever's CJK bigram tokenizer)", () => {
    // CJK-only input → no Latin runs, no variants emitted.
    expect(expandKeywordVariants("我查一下哪裡有賣")).toEqual([]);
  });

  it("dedupes and caps at maxVariants", () => {
    const v = expandKeywordVariants("AAA1 AAA1 AAA1 AAA1", 3);
    expect(v.length).toBeLessThanOrEqual(3);
  });
});

describe("buildAugmentedBm25Query", () => {
  it("returns clean query when no variants", () => {
    expect(buildAugmentedBm25Query("hello world", [])).toBe("hello world");
  });

  it("prepends variants for BM25 keyword matching", () => {
    const out = buildAugmentedBm25Query("我查一下POCKET4哪裡有賣", [
      "pocket4",
      "pocket 4",
      "pocket",
      "4",
    ]);
    expect(out.startsWith("pocket4 pocket 4 pocket 4 ")).toBe(true);
    expect(out).toContain("我查一下POCKET4哪裡有賣");
  });

  it("returns variants only when clean query is empty", () => {
    expect(buildAugmentedBm25Query("", ["pocket4", "pocket 4"])).toBe("pocket4 pocket 4");
  });
});
