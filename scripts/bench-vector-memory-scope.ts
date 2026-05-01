/**
 * Scope-level recall bench for vector-memory. Targets the
 * `<from-workspace-memory>` / `<from-daily-memory>` path that does NOT go
 * through the extracted-conversations retriever.
 *
 * Runs `retrieveScope` + `formatScopedBlock` against the live per-agent
 * LanceDB sub-dirs (`__workspace__`, `__dailies__`) under
 * `~/.openclaw/memory/vector-memory/<agent>/`.
 *
 * Prints, for the given --maxChars values, both the wall-clock timings
 * and the exact block body a real `before_prompt_build` hook would emit,
 * so we can compare injection quality (how much of each chunk survives
 * the slice) and spot obvious truncation at a glance.
 *
 * Usage:
 *   pnpm tsx scripts/bench-vector-memory-scope.ts \
 *     --agent main --scope workspace --query "哥哥最喜歡吃什麼"
 *
 * Optional flags:
 *   --runs 3                 how many warm retrieve iterations to time
 *   --max-chars 600,6000     comma list of budgets to format with
 *   --show-body              print the full formatted block body
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { createEmbedder } from "../extensions/vector-memory/src/embedder.ts";
import { StoreManager } from "../extensions/vector-memory/src/store.ts";
import {
  DEFAULT_DAILY_SCOPE,
  DEFAULT_WORKSPACE_SCOPE,
  formatScopedBlock,
  retrieveScope,
  type DailyScopeConfig,
  type ScopedRetrievalResult,
  type WorkspaceScopeConfig,
} from "../extensions/vector-memory/src/workspace-memory.ts";

function parseArg(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  if (idx < 0) {
    return undefined;
  }
  return process.argv[idx + 1];
}

function hasFlag(flag: string): boolean {
  return process.argv.includes(flag);
}

function summarize(label: string, samples: number[]): void {
  if (samples.length === 0) {
    console.log(`[${label}] no samples`);
    return;
  }
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
  const rawScope = parseArg("--scope") ?? "workspace";
  if (rawScope !== "workspace" && rawScope !== "daily") {
    throw new Error(`--scope must be 'workspace' or 'daily', got '${rawScope}'`);
  }
  const scope: "workspace" | "daily" = rawScope;
  const query = parseArg("--query") ?? "哥哥最喜歡吃什麼食物,特別是宵夜時段";
  const runs = Number(parseArg("--runs") ?? "3");
  const showBody = hasFlag("--show-body");
  const maxCharsList = (parseArg("--max-chars") ?? "600,6000")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  const ctxOverride = parseArg("--ctx");

  const configPath = path.join(os.homedir(), ".openclaw", "openclaw.json");
  const cfg = JSON.parse(fs.readFileSync(configPath, "utf8")) as {
    plugins: { entries: Record<string, { config?: Record<string, unknown> }> };
  };
  const vmCfg = (cfg.plugins.entries["vector-memory"]?.config ?? {}) as {
    workspaceMemory?: Partial<WorkspaceScopeConfig>;
    dailyMemory?: Partial<DailyScopeConfig>;
    embedding?: Record<string, unknown>;
  };

  const userScopeCfg = scope === "workspace" ? vmCfg.workspaceMemory : vmCfg.dailyMemory;
  const defaultScope = scope === "workspace" ? DEFAULT_WORKSPACE_SCOPE : DEFAULT_DAILY_SCOPE;

  // Merge recall config from user config over defaults so we respect the
  // real runtime candidatePool / minScore / contextWindowChunks values.
  const recall = {
    ...defaultScope.recall,
    ...userScopeCfg?.recall,
    ...(ctxOverride !== undefined ? { contextWindowChunks: Number(ctxOverride) } : {}),
  };

  const dbRoot = path.join(os.homedir(), ".openclaw", "memory", "vector-memory");
  const vectorDim =
    (vmCfg.embedding as { dims?: number; dimensions?: number } | undefined)?.dims ??
    (vmCfg.embedding as { dimensions?: number } | undefined)?.dimensions ??
    1536;

  const embedder = createEmbedder(vmCfg.embedding as never);
  const storeManager = new StoreManager(dbRoot, vectorDim);
  const store =
    scope === "workspace"
      ? storeManager.getWorkspaceStore(agent)
      : storeManager.getDailyStore(agent);

  console.log(`[bench] agent=${agent} scope=${scope} query="${query}"`);
  console.log(
    `[bench] recall.candidatePool=${recall.candidatePool} maxItems=${recall.maxItems} maxChars(runtime)=${recall.maxChars} contextWindowChunks=${recall.contextWindowChunks} minScore=${recall.minScore}`,
  );

  // Warmup: opens LanceDB, loads FTS index.
  const tWarm = performance.now();
  const warmResults = await retrieveScope({ query, recall, store, embedder });
  console.log(
    `[warmup] ${(performance.now() - tWarm).toFixed(1)}ms -> ${warmResults.length} results`,
  );

  const totals: number[] = [];
  let lastResults: ScopedRetrievalResult[] = warmResults;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    lastResults = await retrieveScope({ query, recall, store, embedder });
    totals.push(performance.now() - t0);
  }

  console.log("");
  summarize(`retrieve[${scope}]`, totals);
  console.log("");

  if (lastResults.length === 0) {
    console.log(`[bench] no results for query — nothing to format`);
    return;
  }

  // Corpus stats over the candidate pool actually returned (already sorted
  // by final score). These are the chunks a real hook would format into
  // the block.
  const charsPerResult = lastResults.map((r) => r.displayText.length);
  const totalChars = charsPerResult.reduce((a, b) => a + b, 0);
  const avgChars = Math.round(totalChars / charsPerResult.length);
  const maxChunkChars = charsPerResult.reduce((a, b) => Math.max(a, b), 0);
  console.log(
    `[candidate stats] results=${lastResults.length} totalChars=${totalChars} avgChars=${avgChars} maxChunkChars=${maxChunkChars}`,
  );
  console.log("");

  // For each configured maxChars budget, simulate formatScopedBlock and
  // report the actual injected length + whether everything fit.
  const itemsForBlock = lastResults.slice(0, recall.maxItems).map((r) => ({
    sourceFile: r.sourceFile,
    lineStart: r.lineStart,
    lineEnd: r.lineEnd,
    text: r.displayText,
  }));

  for (const budget of maxCharsList) {
    const t0 = performance.now();
    const block = formatScopedBlock(itemsForBlock, budget);
    const formatMs = performance.now() - t0;
    if (!block) {
      console.log(`[maxChars=${budget}] (no block)`);
      continue;
    }
    const fullyFits = itemsForBlock.every((it) => block.body.includes(it.text.slice(0, 64)));
    console.log(
      `[maxChars=${budget}] used=${block.used}/${budget} bodyLen=${block.body.length} items=${itemsForBlock.length} formatTime=${formatMs.toFixed(2)}ms fullyFits≈${fullyFits}`,
    );
    if (showBody) {
      console.log("--- <from-" + scope + "-memory> body ---");
      console.log(block.body);
      console.log("--- end ---\n");
    }
  }
}

main().catch((err) => {
  console.error("bench failed:", err);
  process.exit(1);
});
