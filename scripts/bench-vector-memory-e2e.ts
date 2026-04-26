/**
 * End-to-end recall-latency bench for vector-memory against the live
 * LanceDB at ~/.openclaw/memory/vector-memory/<agentId>/.
 *
 * Measures what a real `before_prompt_build` hook would pay, minus the
 * plugin-host/WS hops:
 *   - openAI embedding call (Gemini via openai-compat)
 *   - LanceDB vector + BM25 hybrid retrieve
 *   - rerank (local-onnx by default, per user config)
 *
 * Prints per-stage timings so we can compare candidatePoolSize impact.
 *
 * Usage:
 *   pnpm tsx scripts/bench-vector-memory-e2e.ts \
 *     --agent main --query "哥哥最喜歡吃什麼" [--runs 5]
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { createEmbedder } from "../extensions/vector-memory/src/embedder.ts";
import { createRetriever } from "../extensions/vector-memory/src/retriever.ts";
import type { RetrievalConfig } from "../extensions/vector-memory/src/retriever.ts";
import { MemoryStore } from "../extensions/vector-memory/src/store.ts";

function parseArg(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  if (idx < 0) {
    return undefined;
  }
  return process.argv[idx + 1];
}

function summarize(label: string, samples: number[]): void {
  const sorted = [...samples].toSorted((a, b) => a - b);
  const min = sorted[0] ?? 0;
  const max = sorted[sorted.length - 1] ?? 0;
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  console.log(
    `[${label}] n=${samples.length} min=${min.toFixed(1)}ms median=${median.toFixed(1)}ms mean=${mean.toFixed(1)}ms max=${max.toFixed(1)}ms`,
  );
}

async function main() {
  const agent = parseArg("--agent") ?? "main";
  const query = parseArg("--query") ?? "哥哥最喜歡吃什麼食物,特別是宵夜時段";
  const runs = Number(parseArg("--runs") ?? "5");
  // Realistic top-K the plugin asks for (autoRecallMaxItems default = 3).
  // retriever internally uses pool = Math.max(candidatePoolSize, limit*2),
  // so passing limit=3 actually honors candidatePoolSize=10.
  const limit = Number(parseArg("--limit") ?? "3");

  const configPath = path.join(os.homedir(), ".openclaw", "openclaw.json");
  const cfg = JSON.parse(fs.readFileSync(configPath, "utf8")) as {
    plugins: { entries: Record<string, { config?: Record<string, unknown> }> };
  };
  const vmCfg = (cfg.plugins.entries["vector-memory"]?.config ?? {}) as {
    retrieval?: Partial<RetrievalConfig>;
    embedding?: Record<string, unknown>;
  };

  const retrievalConfig: RetrievalConfig = {
    mode: "hybrid",
    vectorWeight: 0.7,
    bm25Weight: 0.3,
    minScore: 0.3,
    rerank: "cross-encoder",
    rerankProvider: "jina",
    rerankModel: "jina-reranker-v3",
    candidatePoolSize: 10,
    recencyHalfLifeDays: 14,
    recencyWeight: 0.1,
    filterNoise: true,
    lengthNormAnchor: 500,
    hardMinScore: 0.35,
    timeDecayHalfLifeDays: 60,
    ...vmCfg.retrieval,
    rerankLogger: (level, message) => {
      console.log(`[rerank:${level}] ${message}`);
    },
  };

  const dbRoot = path.join(os.homedir(), ".openclaw", "memory", "vector-memory", agent);
  console.log(`[bench] agent=${agent} db=${dbRoot}`);
  console.log(
    `[bench] rerank=${retrievalConfig.rerank} model=${retrievalConfig.rerankModel} pool=${retrievalConfig.candidatePoolSize}`,
  );

  const embedder = createEmbedder(vmCfg.embedding as never);
  const vectorDim =
    (vmCfg.embedding as { dims?: number; dimensions?: number } | undefined)?.dims ??
    (vmCfg.embedding as { dimensions?: number } | undefined)?.dimensions ??
    1536;
  const store = new MemoryStore({ dbPath: dbRoot, vectorDim });

  const retriever = createRetriever(store, embedder, retrievalConfig);

  // Warmup #1: opens the LanceDB (loads FTS index). First call goes
  // through the vector-only path because hasFtsSupport is still false
  // until initialize() finishes, so we do a second warmup to make sure
  // we're timing the real hybrid+rerank path.
  const tWarm1 = performance.now();
  await retriever.retrieve({ query, limit });
  console.log(`[warmup#1 DB open] ${(performance.now() - tWarm1).toFixed(1)}ms`);

  const tWarm2 = performance.now();
  await retriever.retrieve({ query, limit });
  const warm2 = performance.now() - tWarm2;
  console.log(`[warmup#2 rerank load] ${warm2.toFixed(1)}ms | stages:`, retriever.lastTimings);

  const totals: number[] = [];
  const vectors: number[] = [];
  const reranks: number[] = [];
  const rerankDocs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await retriever.retrieve({ query, limit });
    totals.push(performance.now() - t0);
    vectors.push(retriever.lastTimings.vectorAndBm25);
    reranks.push(retriever.lastTimings.rerank);
    rerankDocs.push(retriever.lastTimings.rerankDocs);
  }

  console.log("");
  summarize("total    ", totals);
  summarize("vec+bm25 ", vectors);
  summarize(`rerank[${retriever.lastTimings.rerankMode}×${rerankDocs[0] ?? 0}docs]`, reranks);
}

main().catch((err) => {
  console.error("bench failed:", err);
  process.exit(1);
});
