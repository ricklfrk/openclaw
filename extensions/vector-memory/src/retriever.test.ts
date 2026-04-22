import { describe, expect, it } from "vitest";
import { tokenize, localBm25Rerank } from "./retriever.js";

describe("tokenize", () => {
  it("extracts latin tokens", () => {
    const tokens = tokenize("Hello world 123");
    expect(tokens).toContain("hello");
    expect(tokens).toContain("world");
    expect(tokens).toContain("123");
  });

  it("extracts CJK bigrams", () => {
    const tokens = tokenize("用戶偏好");
    expect(tokens).toContain("用戶");
    expect(tokens).toContain("戶偏");
    expect(tokens).toContain("偏好");
  });

  it("handles mixed text", () => {
    const tokens = tokenize("User 偏好 TypeScript");
    expect(tokens).toContain("user");
    expect(tokens).toContain("typescript");
    expect(tokens).toContain("偏好");
  });

  it("returns empty for empty string", () => {
    expect(tokenize("")).toEqual([]);
  });
});

describe("localBm25Rerank", () => {
  function makeResult(text: string, score: number) {
    return {
      entry: { text, metadata: "{}", vector: [] as number[] },
      score,
      sources: {},
    };
  }

  it("returns empty array for empty input", () => {
    expect(localBm25Rerank("test", [])).toEqual([]);
  });

  it("boosts results that match query terms", () => {
    const results = [
      makeResult("Some unrelated document about cats", 0.8),
      makeResult("TypeScript compiler optimization techniques", 0.6),
    ];

    const reranked = localBm25Rerank("TypeScript optimization", results);
    // The TypeScript match should now be ranked higher or have boosted score
    expect(reranked[0].entry.text).toContain("TypeScript");
    expect(reranked[0].score).toBeGreaterThanOrEqual(reranked[1].score);
  });

  it("preserves all results", () => {
    const results = [makeResult("aaa", 0.5), makeResult("bbb", 0.5), makeResult("ccc", 0.5)];
    const reranked = localBm25Rerank("aaa", results);
    expect(reranked.length).toBe(3);
  });

  it("sets reranked source with bm25-lite method", () => {
    const results = [makeResult("test query match", 0.7)];
    const reranked = localBm25Rerank("test", results);
    expect((reranked[0].sources as Record<string, unknown>).reranked).toEqual(
      expect.objectContaining({ method: "bm25-lite" }),
    );
  });

  it("handles CJK query matching", () => {
    const results = [
      makeResult("User prefers dark mode", 0.7),
      makeResult("用戶偏好深色主題", 0.6),
    ];
    const reranked = localBm25Rerank("用戶偏好", results);
    expect(reranked[0].entry.text).toContain("用戶偏好");
  });
});
