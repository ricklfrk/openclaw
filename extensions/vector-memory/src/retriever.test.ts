import { describe, expect, it } from "vitest";
import type { Embedder } from "./embedder.js";
import {
  DEFAULT_RETRIEVAL_CONFIG,
  MemoryRetriever,
  localBm25Rerank,
  tokenize,
} from "./retriever.js";
import type { MemoryEntry, MemorySearchResult, MemoryStore } from "./store.js";

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

// ============================================================================
// Time-decay gating — admission ordering regression test
// ============================================================================
//
// Regression for 2026-04-26 bug: a 60-day half-life time-decay setting was
// effectively a no-op for admission because applyTimeDecay used to run AFTER
// the hardMinScore filter, so stale memories with borderline-relevant raw
// scores could still slip into the prompt. Only the final ranking saw the
// decayed score. This test locks in the corrected ordering (decay → filter).

function makeEntry(overrides: Partial<MemoryEntry>): MemoryEntry {
  return {
    id: "e",
    text: "short body",
    vector: [1, 0, 0],
    category: "entity",
    importance: 1,
    timestamp: Date.now(),
    metadata: "{}",
    ...overrides,
  };
}

function makeStubStore(rows: MemorySearchResult[]): MemoryStore {
  return {
    hasFtsSupport: false,
    vectorSearch: async (_vec: number[], _limit: number, _minScore?: number) => rows,
  } as unknown as MemoryStore;
}

function makeStubEmbedder(): Embedder {
  return {
    embedQuery: async () => [1, 0, 0],
    dimensions: 3,
  } as unknown as Embedder;
}

describe("MemoryRetriever — time decay gates admission before hardMinScore", () => {
  const DAY = 86_400_000;

  function buildRetriever(
    rows: MemorySearchResult[],
    overrides: Partial<typeof DEFAULT_RETRIEVAL_CONFIG> = {},
  ): MemoryRetriever {
    return new MemoryRetriever(makeStubStore(rows), makeStubEmbedder(), {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      hardMinScore: 0.4,
      timeDecayHalfLifeDays: 60,
      recencyWeight: 0,
      lengthNormAnchor: 0,
      filterNoise: false,
      ...overrides,
    });
  }

  it("drops a very old borderline-score entry while keeping a fresh one", async () => {
    const now = Date.now();
    const fresh: MemorySearchResult = {
      entry: makeEntry({ id: "fresh", timestamp: now }),
      score: 0.55,
    };
    const stale: MemorySearchResult = {
      entry: makeEntry({ id: "stale", timestamp: now - 1000 * DAY }),
      score: 0.55,
    };

    const retriever = buildRetriever([fresh, stale]);
    const results = await retriever.retrieve({ query: "anything", limit: 5 });

    expect(results).toHaveLength(1);
    expect(results[0].entry.id).toBe("fresh");
  });

  it("keeps both entries when both are fresh (decay is a no-op in that case)", async () => {
    const now = Date.now();
    const a: MemorySearchResult = {
      entry: makeEntry({ id: "a", timestamp: now }),
      score: 0.55,
    };
    const b: MemorySearchResult = {
      entry: makeEntry({ id: "b", timestamp: now - 1 * DAY }),
      score: 0.55,
    };

    const retriever = buildRetriever([a, b]);
    const results = await retriever.retrieve({ query: "anything", limit: 5 });

    expect(results).toHaveLength(2);
  });

  it("also gates a high-score stale entry whose decayed score falls below threshold", async () => {
    const now = Date.now();
    // Raw score 0.62 would pass hardMinScore=0.4 if decay were not applied.
    // With 1000-day age and 60-day half-life, decay factor → floor 0.5,
    // so decayed score ≈ 0.31, which is below 0.4 → must be filtered.
    const stale: MemorySearchResult = {
      entry: makeEntry({ id: "stale", timestamp: now - 1000 * DAY }),
      score: 0.62,
    };

    const retriever = buildRetriever([stale]);
    const results = await retriever.retrieve({ query: "anything", limit: 5 });

    expect(results).toHaveLength(0);
  });

  it("keeps a stale entry with a very strong raw score (decay alone cannot floor below 50%)", async () => {
    const now = Date.now();
    // Raw score 0.9 × decay floor 0.5 = 0.45 ≥ hardMinScore 0.4 → must survive.
    const stale: MemorySearchResult = {
      entry: makeEntry({ id: "strong-stale", timestamp: now - 1000 * DAY }),
      score: 0.9,
    };

    const retriever = buildRetriever([stale]);
    const results = await retriever.retrieve({ query: "anything", limit: 5 });

    expect(results).toHaveLength(1);
    expect(results[0].entry.id).toBe("strong-stale");
  });

  it("disables decay when timeDecayHalfLifeDays=0 (reverts to pure score filter)", async () => {
    const now = Date.now();
    const stale: MemorySearchResult = {
      entry: makeEntry({ id: "stale", timestamp: now - 1000 * DAY }),
      score: 0.55,
    };

    const retriever = buildRetriever([stale], { timeDecayHalfLifeDays: 0 });
    const results = await retriever.retrieve({ query: "anything", limit: 5 });

    expect(results).toHaveLength(1);
  });
});

describe("DEFAULT_RETRIEVAL_CONFIG", () => {
  it("uses hardMinScore=0.4 (bumped from 0.35 on 2026-04-26)", () => {
    expect(DEFAULT_RETRIEVAL_CONFIG.hardMinScore).toBe(0.4);
  });
});
