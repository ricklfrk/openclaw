/**
 * Shared rerank utility — reusable across the `conversations` (extracted),
 * `workspace`, and `daily` retrieval scopes.
 *
 * Input: a query + an ordered list of passage strings (the exact text the
 * reranker should score — callers are expected to pass the final display form,
 * e.g. a chunk already expanded with context-window neighbours).
 *
 * Output: a parallel array of normalized scores in [0, 1] (same length & order
 * as `documents`), plus the method that produced them. Returns `null` when all
 * configured backends fail; the caller should then fall back to its own
 * fusion score.
 */

import { DEFAULT_LOCAL_ONNX_RERANK_MODEL, scoreQueryPassagePairs } from "./local-onnx-rerank.js";
import { tokenize } from "./retriever.js";

export type RerankProvider = "jina" | "siliconflow" | "voyage" | "pinecone" | "dashscope" | "tei";

export type RerankMethod = "cross-encoder" | "local-onnx" | "local-bm25" | "none";

export interface RerankPassagesConfig {
  /** Which backend to try first. */
  method: "cross-encoder" | "local-onnx" | "lightweight" | "none";
  /** Required for `cross-encoder`. */
  apiKey?: string;
  model?: string;
  endpoint?: string;
  provider?: RerankProvider;
  /** Abort timeout for cross-encoder HTTP call. Default 5000. */
  timeoutMs?: number;
  logger?: (level: "info" | "error", message: string) => void;
}

export interface RerankPassagesResult {
  /** Same length & order as input `documents`. Values in [0, 1]. */
  scores: number[];
  method: RerankMethod;
}

// ---------------------------------------------------------------------------
// Cross-encoder provider adapters (extracted from retriever.ts)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Normalize cross-encoder provider scores onto [0, 1]. Jina, SiliconFlow,
 * Voyage etc. already return 0..1 relevance; TEI/Dashscope sometimes emit
 * raw logits. Sigmoid handles both safely — values already in [0, 1] map
 * close to the identity of a sigmoid near 0.5 and stay monotonic.
 */
function normalizeCrossEncoderScore(raw: number): number {
  if (raw >= 0 && raw <= 1) {
    return raw;
  }
  return 1 / (1 + Math.exp(-raw));
}

/**
 * Rerank `documents` against `query`. Tries configured backend first, then
 * falls back to local BM25-lite. Returns null only if the documents list is
 * empty.
 *
 * Score semantics:
 *   - cross-encoder / local-onnx: normalized rerank score in [0, 1].
 *   - local-bm25: BM25-lite keyword relevance in [0, 1] (no original score
 *     blend — callers blend with their own baseline).
 */
export async function rerankPassages(
  query: string,
  documents: string[],
  config: RerankPassagesConfig,
): Promise<RerankPassagesResult | null> {
  if (documents.length === 0) {
    return null;
  }
  if (config.method === "none") {
    return { scores: documents.map(() => 0), method: "none" };
  }

  // 1) Hosted cross-encoder (Jina et al.)
  if (config.method === "cross-encoder" && config.apiKey) {
    try {
      const provider = config.provider || "jina";
      const model = config.model || "jina-reranker-v3";
      const endpoint = config.endpoint || "https://api.jina.ai/v1/rerank";
      const timeoutMs = config.timeoutMs ?? 5000;
      const { headers, body } = buildRerankRequest(
        provider,
        config.apiKey,
        model,
        query,
        documents,
        documents.length,
      );
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const response = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (response.ok) {
        const data: unknown = await response.json();
        const parsed = parseRerankResponse(provider, data);
        if (parsed) {
          const scores = Array.from<number>({ length: documents.length }).fill(0);
          for (const item of parsed) {
            if (item.index >= 0 && item.index < documents.length) {
              scores[item.index] = normalizeCrossEncoderScore(item.score);
            }
          }
          return { scores, method: "cross-encoder" };
        }
        config.logger?.(
          "error",
          `cross-encoder rerank response parse failed (${provider}/${model}); falling back`,
        );
      } else {
        config.logger?.(
          "error",
          `cross-encoder rerank HTTP ${response.status} (${provider}/${model}); falling back`,
        );
      }
    } catch (err) {
      config.logger?.("error", `cross-encoder rerank failed, falling back: ${String(err)}`);
    }
  }

  // 2) Local ONNX cross-encoder (bge-reranker-v2-m3 q8)
  if (config.method === "local-onnx") {
    try {
      const rawScores = await scoreQueryPassagePairs(query, documents, {
        modelId: config.model || DEFAULT_LOCAL_ONNX_RERANK_MODEL,
        logger: config.logger,
      });
      if (rawScores.length === documents.length) {
        // bge-reranker emits logits (~ -10..+10). Sigmoid → [0, 1].
        const scores = rawScores.map((s) => 1 / (1 + Math.exp(-s)));
        return { scores, method: "local-onnx" };
      }
    } catch (err) {
      config.logger?.("error", `local-onnx rerank failed, falling back: ${String(err)}`);
    }
  }

  // 3) BM25-lite fallback — always available, no deps.
  return {
    scores: scoreBm25Lite(query, documents),
    method: "local-bm25",
  };
}

// ---------------------------------------------------------------------------
// BM25-lite: pure keyword relevance, no baseline-score blending. Callers
// blend with their own fusion score if they want a combined ranking.
// ---------------------------------------------------------------------------

function scoreBm25Lite(query: string, documents: string[]): number[] {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0 || documents.length === 0) {
    return documents.map(() => 0);
  }

  const k1 = 1.5;
  const b = 0.75;
  const docTokens = documents.map((d) => tokenize(d));
  const N = documents.length;
  const avgDl = docTokens.reduce((sum, d) => sum + d.length, 0) / N || 1;

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

  const raw = docTokens.map((dt) => {
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

  const maxRaw = Math.max(...raw, 1e-9);
  return raw.map((s) => s / maxRaw);
}
