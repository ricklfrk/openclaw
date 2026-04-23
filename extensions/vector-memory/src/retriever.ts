/**
 * Hybrid Memory Retriever — vector + BM25 + rerank + RRF + MMR.
 * From memory-lancedb-pro, stripped of decay engine / tier manager / access tracker.
 */

import type { Embedder } from "./embedder.js";
import { filterNoise } from "./noise-filter.js";
import type { MemoryStore, MemorySearchResult } from "./store.js";

// ============================================================================
// Configuration
// ============================================================================

export interface RetrievalConfig {
  mode: "hybrid" | "vector";
  vectorWeight: number;
  bm25Weight: number;
  minScore: number;
  rerank: "cross-encoder" | "lightweight" | "none";
  candidatePoolSize: number;
  recencyHalfLifeDays: number;
  recencyWeight: number;
  filterNoise: boolean;
  rerankApiKey?: string;
  rerankModel?: string;
  rerankEndpoint?: string;
  rerankProvider?: RerankProvider;
  lengthNormAnchor: number;
  hardMinScore: number;
  timeDecayHalfLifeDays: number;
}

export interface RetrievalContext {
  query: string;
  limit: number;
  category?: string;
  entityTags?: string[];
}

export interface RetrievalResult extends MemorySearchResult {
  sources: {
    vector?: { score: number; rank: number };
    bm25?: { score: number; rank: number };
    fused?: { score: number };
    reranked?: { score: number };
  };
}

export const DEFAULT_RETRIEVAL_CONFIG: RetrievalConfig = {
  mode: "hybrid",
  vectorWeight: 0.7,
  bm25Weight: 0.3,
  minScore: 0.3,
  rerank: "cross-encoder",
  candidatePoolSize: 20,
  recencyHalfLifeDays: 14,
  recencyWeight: 0.1,
  filterNoise: true,
  rerankModel: "jina-reranker-v3",
  rerankEndpoint: "https://api.jina.ai/v1/rerank",
  lengthNormAnchor: 500,
  hardMinScore: 0.35,
  timeDecayHalfLifeDays: 60,
};

// ============================================================================
// Utility Functions
// ============================================================================

function clamp01(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return Number.isFinite(fallback) ? fallback : 0;
  }
  return Math.min(1, Math.max(0, value));
}

function clamp01WithFloor(value: number, floor: number): number {
  const safeFloor = clamp01(floor, 0);
  return Math.max(safeFloor, clamp01(value, safeFloor));
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vector dimensions must match");
  }
  let dotProduct = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const norm = Math.sqrt(normA) * Math.sqrt(normB);
  return norm === 0 ? 0 : dotProduct / norm;
}

// ============================================================================
// Local BM25-lite Rerank
// ============================================================================

/**
 * Tokenize text for BM25 scoring.
 * Splits on whitespace + punctuation; adds CJK character bigrams.
 */
export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  // Latin / alphanumeric tokens
  for (const m of text.toLowerCase().matchAll(/[a-z0-9\u00c0-\u024f]+/g)) {
    tokens.push(m[0]);
  }
  // CJK bigrams (Chinese / Japanese / Korean)
  const cjk = text.replace(
    /[^\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/g,
    "",
  );
  for (let i = 0; i < cjk.length - 1; i++) {
    tokens.push(cjk.slice(i, i + 2));
  }
  if (cjk.length === 1) {
    tokens.push(cjk);
  }
  return tokens;
}

interface ScoredResult {
  entry: { text: string; metadata: string; vector: number[] };
  score: number;
  sources?: Record<string, unknown>;
}

/**
 * BM25-lite reranking over the candidate pool.
 * Blends keyword relevance (BM25) with the original retrieval score.
 */
export function localBm25Rerank<T extends ScoredResult>(query: string, results: T[]): T[] {
  if (results.length === 0) {
    return results;
  }

  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return results;
  }

  const k1 = 1.5;
  const b = 0.75;

  // Build per-document token frequency lists and collect corpus stats
  const docTokens = results.map((r) => {
    let text = r.entry.text;
    try {
      const meta = JSON.parse(r.entry.metadata || "{}");
      if (meta.l0_abstract) {
        text += ` ${meta.l0_abstract}`;
      }
      if (meta.l2_content) {
        text += ` ${meta.l2_content}`;
      }
    } catch {}
    return tokenize(text);
  });

  const N = results.length;
  const avgDl = docTokens.reduce((sum, d) => sum + d.length, 0) / N || 1;

  // Document frequency for each query term
  const df = new Map<string, number>();
  for (const qt of queryTokens) {
    if (df.has(qt)) {
      continue;
    }
    let count = 0;
    for (const dt of docTokens) {
      if (dt.includes(qt)) {
        count++;
      }
    }
    df.set(qt, count);
  }

  // Score each document
  const bm25Scores: number[] = docTokens.map((dt) => {
    const dl = dt.length;
    let score = 0;
    const tfMap = new Map<string, number>();
    for (const t of dt) {
      tfMap.set(t, (tfMap.get(t) ?? 0) + 1);
    }
    for (const qt of queryTokens) {
      const tf = tfMap.get(qt) ?? 0;
      if (tf === 0) {
        continue;
      }
      const docFreq = df.get(qt) ?? 0;
      const idf = Math.log((N - docFreq + 0.5) / (docFreq + 0.5) + 1);
      score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgDl))));
    }
    return score;
  });

  // Normalize BM25 scores to [0, 1]
  const maxBm25 = Math.max(...bm25Scores, 1e-9);
  const normalized = bm25Scores.map((s) => s / maxBm25);

  // Blend with original scores
  return results
    .map((r, i) => ({
      ...r,
      score: clamp01(r.score * 0.5 + normalized[i] * 0.5, r.score),
      sources: { ...r.sources, reranked: { score: normalized[i], method: "bm25-lite" } },
    }))
    .toSorted((a, b) => b.score - a.score);
}

// ============================================================================
// Rerank Provider Adapters
// ============================================================================

type RerankProvider = "jina" | "siliconflow" | "voyage" | "pinecone" | "dashscope" | "tei";

interface RerankItem {
  index: number;
  score: number;
}

function buildRerankRequest(
  provider: RerankProvider,
  apiKey: string,
  model: string,
  query: string,
  candidates: string[],
  topN: number,
): { headers: Record<string, string>; body: Record<string, unknown> } {
  const headers: Record<string, string> =
    provider === "pinecone"
      ? {
          "Content-Type": "application/json",
          "Api-Key": apiKey,
          "X-Pinecone-API-Version": "2024-10",
        }
      : { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` };

  switch (provider) {
    case "tei":
      return { headers, body: { query, texts: candidates } };
    case "dashscope":
      return { headers, body: { model, input: { query, documents: candidates } } };
    case "pinecone":
      return {
        headers,
        body: {
          model,
          query,
          documents: candidates.map((text) => ({ text })),
          top_n: topN,
          rank_fields: ["text"],
        },
      };
    case "voyage":
      return { headers, body: { model, query, documents: candidates, top_k: topN } };
    default:
      return { headers, body: { model, query, documents: candidates, top_n: topN } };
  }
}

function parseRerankResponse(provider: RerankProvider, data: unknown): RerankItem[] | null {
  const parseItems = (items: unknown, scoreKeys: string[]): RerankItem[] | null => {
    if (!Array.isArray(items)) {
      return null;
    }
    const parsed: RerankItem[] = [];
    for (const raw of items as Array<Record<string, unknown>>) {
      const index = typeof raw?.index === "number" ? raw.index : Number(raw?.index);
      if (!Number.isFinite(index)) {
        continue;
      }
      let score: number | null = null;
      for (const key of scoreKeys) {
        const n = typeof raw?.[key] === "number" ? raw[key] : Number(raw?.[key]);
        if (Number.isFinite(n)) {
          score = n;
          break;
        }
      }
      if (score === null) {
        continue;
      }
      parsed.push({ index, score });
    }
    return parsed.length > 0 ? parsed : null;
  };

  const obj =
    data && typeof data === "object" && !Array.isArray(data)
      ? (data as Record<string, unknown>)
      : undefined;
  switch (provider) {
    case "tei":
      return (
        parseItems(data, ["score", "relevance_score"]) ??
        parseItems(obj?.results, ["score", "relevance_score"]) ??
        parseItems(obj?.data, ["score", "relevance_score"])
      );
    case "dashscope": {
      const output = obj?.output as Record<string, unknown> | undefined;
      return (
        parseItems(output?.results, ["relevance_score", "score"]) ??
        parseItems(obj?.results, ["relevance_score", "score"])
      );
    }
    case "pinecone":
      return (
        parseItems(obj?.data, ["score", "relevance_score"]) ??
        parseItems(obj?.results, ["score", "relevance_score"])
      );
    case "voyage":
      return (
        parseItems(obj?.data, ["relevance_score", "score"]) ??
        parseItems(obj?.results, ["relevance_score", "score"])
      );
    default:
      return (
        parseItems(obj?.results, ["relevance_score", "score"]) ??
        parseItems(obj?.data, ["relevance_score", "score"])
      );
  }
}

// ============================================================================
// Memory Retriever
// ============================================================================

export class MemoryRetriever {
  constructor(
    private store: MemoryStore,
    private embedder: Embedder,
    private config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
  ) {}

  async retrieve(context: RetrievalContext): Promise<RetrievalResult[]> {
    const { query, limit, category, entityTags } = context;
    const safeLimit = Math.min(Math.max(limit, 1), 20);

    let results: RetrievalResult[];

    if (this.config.mode === "vector" || !this.store.hasFtsSupport) {
      results = await this.vectorOnlyRetrieval(query, safeLimit, category);
    } else {
      results = await this.hybridRetrieval(query, safeLimit, category);
    }

    if (entityTags && entityTags.length > 0) {
      results = this.applyEntityTagBoost(results, entityTags);
    }

    return results;
  }

  // --------------------------------------------------------------------------
  // Vector-only retrieval
  // --------------------------------------------------------------------------

  private async vectorOnlyRetrieval(
    query: string,
    limit: number,
    category?: string,
  ): Promise<RetrievalResult[]> {
    const queryVector = await this.embedder.embedQuery(query);
    const results = await this.store.vectorSearch(queryVector, limit, this.config.minScore);
    const filtered = category ? results.filter((r) => r.entry.category === category) : results;

    let mapped = filtered.map(
      (result, index) =>
        Object.assign({}, result, {
          sources: { vector: { score: result.score, rank: index + 1 } },
        }) as RetrievalResult,
    );

    // Pre-boost floor: discard results where the raw vector score is too low
    // to be genuinely relevant. Prevents recencyBoost/importanceWeight from
    // inflating unrelated results past hardMinScore.
    mapped = mapped.filter((r) => (r.sources.vector?.score ?? r.score) >= 0.45);

    mapped = this.applyImportanceWeight(this.applyRecencyBoost(mapped));
    mapped = this.applyLengthNormalization(mapped);
    mapped = mapped.filter((r) => r.score >= this.config.hardMinScore);
    mapped = this.applyTimeDecay(mapped);
    if (this.config.filterNoise) {
      mapped = filterNoise(mapped, (r) => r.entry.text);
    }
    mapped = this.applyMMRDiversity(mapped);
    return mapped.slice(0, limit);
  }

  // --------------------------------------------------------------------------
  // Hybrid retrieval (vector + BM25 + RRF + rerank)
  // --------------------------------------------------------------------------

  private async hybridRetrieval(
    query: string,
    limit: number,
    category?: string,
  ): Promise<RetrievalResult[]> {
    const candidatePoolSize = Math.max(this.config.candidatePoolSize, limit * 2);
    const queryVector = await this.embedder.embedQuery(query);

    const [vectorResults, bm25Results] = await Promise.all([
      this.runVectorSearch(queryVector, candidatePoolSize, category),
      this.runBM25Search(query, candidatePoolSize, category),
    ]);

    let fusedResults = await this.fuseResults(vectorResults, bm25Results);
    fusedResults = fusedResults.filter((r) => r.score >= this.config.minScore);

    // Pre-boost floor: require meaningful raw relevance from at least one signal.
    // A result backed only by a weak vector match (< 0.45) with no BM25 hit is
    // almost certainly noise that recency/importance boosts would inflate.
    fusedResults = fusedResults.filter((r) => {
      const vs = r.sources.vector?.score ?? 0;
      const bs = r.sources.bm25?.score ?? 0;
      return vs >= 0.45 || bs >= 0.5;
    });

    let reranked =
      this.config.rerank !== "none"
        ? await this.rerankResults(query, queryVector, fusedResults.slice(0, limit * 2))
        : fusedResults;

    reranked = this.applyImportanceWeight(this.applyRecencyBoost(reranked));
    reranked = this.applyLengthNormalization(reranked);
    reranked = reranked.filter((r) => r.score >= this.config.hardMinScore);
    reranked = this.applyTimeDecay(reranked);
    if (this.config.filterNoise) {
      reranked = filterNoise(reranked, (r) => r.entry.text);
    }
    reranked = this.applyMMRDiversity(reranked);
    return reranked.slice(0, limit);
  }

  private async runVectorSearch(
    queryVector: number[],
    limit: number,
    category?: string,
  ): Promise<Array<MemorySearchResult & { rank: number }>> {
    const results = await this.store.vectorSearch(queryVector, limit, 0.1);
    const filtered = category ? results.filter((r) => r.entry.category === category) : results;
    return filtered.map((result, index) => Object.assign({}, result, { rank: index + 1 }));
  }

  private async runBM25Search(
    query: string,
    limit: number,
    category?: string,
  ): Promise<Array<MemorySearchResult & { rank: number }>> {
    const results = await this.store.bm25Search(query, limit);
    const filtered = category ? results.filter((r) => r.entry.category === category) : results;
    return filtered.map((result, index) => Object.assign({}, result, { rank: index + 1 }));
  }

  // --------------------------------------------------------------------------
  // RRF Fusion
  // --------------------------------------------------------------------------

  private async fuseResults(
    vectorResults: Array<MemorySearchResult & { rank: number }>,
    bm25Results: Array<MemorySearchResult & { rank: number }>,
  ): Promise<RetrievalResult[]> {
    const vectorMap = new Map(vectorResults.map((r) => [r.entry.id, r]));
    const bm25Map = new Map(bm25Results.map((r) => [r.entry.id, r]));
    const allIds = new Set([...vectorMap.keys(), ...bm25Map.keys()]);

    const fusedResults: RetrievalResult[] = [];
    for (const id of allIds) {
      const vectorResult = vectorMap.get(id);
      const bm25Result = bm25Map.get(id);

      if (!vectorResult && bm25Result) {
        try {
          if (!(await this.store.hasId(id))) {
            continue;
          }
        } catch {}
      }

      const baseResult = vectorResult || bm25Result!;
      const vectorScore = vectorResult ? vectorResult.score : 0;
      const bm25Score = bm25Result ? bm25Result.score : 0;
      const weightedFusion =
        vectorScore * this.config.vectorWeight + bm25Score * this.config.bm25Weight;
      const fusedScore = vectorResult
        ? clamp01(Math.max(weightedFusion, bm25Score >= 0.75 ? bm25Score * 0.92 : 0), 0.1)
        : clamp01(bm25Result!.score, 0.1);

      fusedResults.push({
        entry: baseResult.entry,
        score: fusedScore,
        sources: {
          vector: vectorResult ? { score: vectorResult.score, rank: vectorResult.rank } : undefined,
          bm25: bm25Result ? { score: bm25Result.score, rank: bm25Result.rank } : undefined,
          fused: { score: fusedScore },
        },
      });
    }
    return fusedResults.toSorted((a, b) => b.score - a.score);
  }

  // --------------------------------------------------------------------------
  // Rerank (cross-encoder API with cosine fallback)
  // --------------------------------------------------------------------------

  private async rerankResults(
    query: string,
    queryVector: number[],
    results: RetrievalResult[],
  ): Promise<RetrievalResult[]> {
    if (results.length === 0) {
      return results;
    }

    if (this.config.rerank === "cross-encoder" && this.config.rerankApiKey) {
      try {
        const provider = this.config.rerankProvider || "jina";
        const model = this.config.rerankModel || "jina-reranker-v3";
        const endpoint = this.config.rerankEndpoint || "https://api.jina.ai/v1/rerank";
        const documents = results.map((r) => r.entry.text);
        const { headers, body } = buildRerankRequest(
          provider,
          this.config.rerankApiKey,
          model,
          query,
          documents,
          results.length,
        );

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(endpoint, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(timeout);

        if (response.ok) {
          const data: unknown = await response.json();
          const parsed = parseRerankResponse(provider, data);
          if (parsed) {
            const returnedIndices = new Set(parsed.map((r) => r.index));
            const reranked = parsed
              .filter((item) => item.index >= 0 && item.index < results.length)
              .map((item) => {
                const original = results[item.index];
                const bm25Score = original.sources.bm25?.score ?? 0;
                const floor =
                  bm25Score >= 0.75
                    ? original.score * 0.95
                    : bm25Score >= 0.6
                      ? original.score * 0.9
                      : original.score * 0.5;
                const blendedScore = clamp01WithFloor(
                  item.score * 0.6 + original.score * 0.4,
                  floor,
                );
                return Object.assign({}, original, {
                  score: blendedScore,
                  sources: Object.assign({}, original.sources, {
                    reranked: { score: item.score },
                  }),
                });
              });
            const unreturned = results
              .filter((_, idx) => !returnedIndices.has(idx))
              .map((r) =>
                Object.assign({}, r, {
                  score: clamp01WithFloor(r.score * 0.8, r.score * 0.5),
                }),
              );
            return [...reranked, ...unreturned].toSorted((a, b) => b.score - a.score);
          }
        }
      } catch {}
    }

    // Fallback: BM25-lite local rerank (keyword relevance blended with original score)
    return localBm25Rerank(query, results);
  }

  // --------------------------------------------------------------------------
  // Post-processing
  // --------------------------------------------------------------------------

  private applyRecencyBoost(results: RetrievalResult[]): RetrievalResult[] {
    const { recencyHalfLifeDays, recencyWeight } = this.config;
    if (!recencyHalfLifeDays || recencyHalfLifeDays <= 0 || !recencyWeight) {
      return results;
    }
    const now = Date.now();
    return results
      .map((r) => {
        const ts = r.entry.timestamp && r.entry.timestamp > 0 ? r.entry.timestamp : now;
        const ageDays = (now - ts) / 86_400_000;
        const boost = Math.exp(-ageDays / recencyHalfLifeDays) * recencyWeight;
        return { ...r, score: clamp01(r.score + boost, r.score) };
      })
      .toSorted((a, b) => b.score - a.score);
  }

  private applyImportanceWeight(results: RetrievalResult[]): RetrievalResult[] {
    const baseWeight = 0.7;
    return results
      .map((r) => {
        const importance = r.entry.importance ?? 0.7;
        const factor = baseWeight + (1 - baseWeight) * importance;
        return { ...r, score: clamp01(r.score * factor, r.score * baseWeight) };
      })
      .toSorted((a, b) => b.score - a.score);
  }

  private applyLengthNormalization(results: RetrievalResult[]): RetrievalResult[] {
    const anchor = this.config.lengthNormAnchor;
    if (!anchor || anchor <= 0) {
      return results;
    }
    return results
      .map((r) => {
        const ratio = r.entry.text.length / anchor;
        const logRatio = Math.log2(Math.max(ratio, 1));
        const factor = 1 / (1 + 0.5 * logRatio);
        return { ...r, score: clamp01(r.score * factor, r.score * 0.3) };
      })
      .toSorted((a, b) => b.score - a.score);
  }

  private applyTimeDecay(results: RetrievalResult[]): RetrievalResult[] {
    const halfLife = this.config.timeDecayHalfLifeDays;
    if (!halfLife || halfLife <= 0) {
      return results;
    }
    const now = Date.now();
    return results
      .map((r) => {
        const ts = r.entry.timestamp && r.entry.timestamp > 0 ? r.entry.timestamp : now;
        const ageDays = (now - ts) / 86_400_000;
        const factor = 0.5 + 0.5 * Math.exp(-ageDays / halfLife);
        return { ...r, score: clamp01(r.score * factor, r.score * 0.5) };
      })
      .toSorted((a, b) => b.score - a.score);
  }

  /**
   * Boost results whose metadata.entity_tags overlap with the query's entity tags.
   * Each matching tag adds a small additive boost, rewarding precise entity matches
   * that pure embedding similarity might underweight.
   */
  private applyEntityTagBoost(results: RetrievalResult[], queryTags: string[]): RetrievalResult[] {
    if (queryTags.length === 0) {
      return results;
    }
    const queryTagSet = new Set(queryTags.map((t) => t.toLowerCase()));
    return results
      .map((r) => {
        let tags: string[] = [];
        try {
          const meta = JSON.parse(r.entry.metadata || "{}");
          if (Array.isArray(meta.entity_tags)) {
            tags = meta.entity_tags;
          }
        } catch {}
        if (tags.length === 0) {
          return r;
        }
        const matches = tags.filter((t) => queryTagSet.has(t.toLowerCase())).length;
        if (matches === 0) {
          return r;
        }
        // Boost: up to +0.12 for full overlap (diminishing per tag)
        const boost = Math.min(matches * 0.04, 0.12);
        return { ...r, score: clamp01(r.score + boost, r.score) };
      })
      .toSorted((a, b) => b.score - a.score);
  }

  private applyMMRDiversity(
    results: RetrievalResult[],
    similarityThreshold = 0.85,
  ): RetrievalResult[] {
    if (results.length <= 1) {
      return results;
    }
    const selected: RetrievalResult[] = [];
    const deferred: RetrievalResult[] = [];
    for (const candidate of results) {
      const tooSimilar = selected.some((s) => {
        const sVec = s.entry.vector;
        const cVec = candidate.entry.vector;
        if (!sVec?.length || !cVec?.length) {
          return false;
        }
        return (
          cosineSimilarity(
            Array.from(sVec as Iterable<number>),
            Array.from(cVec as Iterable<number>),
          ) > similarityThreshold
        );
      });
      if (tooSimilar) {
        deferred.push(candidate);
      } else {
        selected.push(candidate);
      }
    }
    return [...selected, ...deferred];
  }

  updateConfig(newConfig: Partial<RetrievalConfig>): void {
    Object.assign(this.config, newConfig);
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createRetriever(
  store: MemoryStore,
  embedder: Embedder,
  config?: Partial<RetrievalConfig>,
): MemoryRetriever {
  return new MemoryRetriever(store, embedder, { ...DEFAULT_RETRIEVAL_CONFIG, ...config });
}
