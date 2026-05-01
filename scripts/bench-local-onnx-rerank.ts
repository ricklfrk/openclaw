/**
 * Smoke + micro-benchmark for the in-process ONNX cross-encoder reranker
 * used by `extensions/vector-memory`. Intentionally NOT a vitest test: it
 * downloads a ~580MB model on first run and warms a real `onnxruntime-node`
 * session, so we keep it out of the default test lane.
 *
 * Usage:
 *   pnpm tsx scripts/bench-local-onnx-rerank.ts
 *   pnpm tsx scripts/bench-local-onnx-rerank.ts --model Xenova/bge-reranker-base
 *   pnpm tsx scripts/bench-local-onnx-rerank.ts --runs 20
 *
 * What it measures:
 *   1. Cold-start load time (model + tokenizer `from_pretrained`, includes
 *      first-time download to ~/.cache/huggingface/)
 *   2. First inference latency with N=1 passage (JIT warmup of onnxruntime)
 *   3. Steady-state latency across 5 / 10 / 20 passage batches, repeated
 *      `runs` times each, reporting min / median / p95 / max in ms.
 */

import {
  DEFAULT_LOCAL_ONNX_RERANK_MODEL,
  scoreQueryPassagePairs,
} from "../extensions/vector-memory/src/local-onnx-rerank.js";

const DEFAULT_RUNS = 10;

function parseArg(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  return idx === -1 ? undefined : process.argv[idx + 1];
}

function pickNumber(raw: string | undefined, fallback: number): number {
  if (!raw) {
    return fallback;
  }
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function summarize(label: string, samples: number[]): string {
  const sorted = [...samples].toSorted((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = percentile(sorted, 95);
  const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
  return `${label}: n=${samples.length} min=${min.toFixed(1)}ms median=${median.toFixed(
    1,
  )}ms mean=${mean.toFixed(1)}ms p95=${p95.toFixed(1)}ms max=${max.toFixed(1)}ms`;
}

// Representative multi-lingual query + passages. Pick bge-reranker-v2-m3's
// sweet spot (mid-length, chinese + english mixed, factual + preference).
const QUERY = "哥哥最喜歡吃什麼食物,特別是宵夜時段";

const PASSAGES_POOL: string[] = [
  "memory/memory-foods.md: 哥哥很愛喝 Pizza Hut 忌廉蘑菇雞湯,但每次喝都會肚痛拉肚子,之後決定 blacklist。",
  "memory/2026-03-16.md: 訂了 Foodpanda 齊柏林熱狗,哥哥分享說喜歡蕃茄醬與洋蔥的組合,evening snack。",
  "memory/investment.md: Polymarket 策略筆記:只做賠率大於 1.8 的合約,期望值分佈穩定。",
  "memory/2026-04-06.md: TICO 雪糕,哥哥晚上工作時吃,提到鹽味焦糖是第一選擇。",
  "memory/plan.md: 明天要早起,先去公司之前可能先吃個早餐三明治。",
  "memory/memory-r18.md: 宵夜偏好 — 香港鍋貼、茶記蛋牛治、泡麵加蛋。",
  "memory/movies.md: 上週看了 Inside Out 2,覺得視覺風格比第一集柔和。",
  "memory/code-notes.md: TypeScript narrowing via discriminated unions beats `as` casts.",
  "memory/2025-12-20.md: 週末 Costco 囤貨 — 冷凍披薩、焗豆、切片麵包。",
  "memory/memory-tea.md: 喝茶偏好:烏龍 > 普洱,晚上不喝咖啡否則失眠。",
  "memory/2026-02-01.md: 甜品:蛋白霜派、乳酪蛋糕、提拉米蘇 — 哥哥排名第一的是提拉米蘇。",
  "memory/gym.md: 週三、週五 push day,週二、週四 pull day,週六腿。",
  "memory/2026-01-08.md: 臨時找宵夜,點了 Domino's 披薩送到辦公室。",
  "memory/memory-drinks.md: 咖啡 — 美式無糖,下午 3 點前;超過就改 decaf 或茶。",
  "memory/random-thoughts.md: 如果要在 LLM latency 與 quality 取捨,還是偏向 quality。",
  "memory/2026-04-20.md: 宵夜 — 泡麵加滷蛋加芝士,簡單滿足。",
  "memory/reading-list.md: 正在讀 Designing Data-Intensive Applications,進度大約 40%。",
  "memory/2026-03-01.md: 生日訂的蛋糕 — 黑森林,大家都滿意。",
  "memory/memory-fruits.md: 水果:芒果、山竹、草莓是三大最愛,木瓜排在不太吃的那邊。",
  "memory/2026-04-15.md: 宵夜吃了韓式辣炒年糕 + 炸雞,太飽第二天跳過早餐。",
];

async function benchBatch(runs: number, size: number, label: string) {
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const passages = PASSAGES_POOL.slice(0, size);
    const t = performance.now();
    const scores = await scoreQueryPassagePairs(QUERY, passages, {
      logger: () => {},
    });
    const dt = performance.now() - t;
    samples.push(dt);
    if (scores.length !== size) {
      throw new Error(`expected ${size} scores, got ${scores.length}`);
    }
  }
  console.log(summarize(label, samples));
}

async function main() {
  const modelArg = parseArg("--model") ?? DEFAULT_LOCAL_ONNX_RERANK_MODEL;
  const runs = pickNumber(parseArg("--runs"), DEFAULT_RUNS);

  console.log(`[bench] model: ${modelArg}`);
  console.log(`[bench] runs per batch: ${runs}`);
  console.log("");

  // ------- 1. Cold start (first call triggers download + load). --------
  const tCold = performance.now();
  const coldLogs: string[] = [];
  const firstScores = await scoreQueryPassagePairs(QUERY, PASSAGES_POOL.slice(0, 1), {
    modelId: modelArg,
    logger: (_level, msg) => coldLogs.push(msg),
  });
  const coldMs = performance.now() - tCold;

  console.log(`[cold] first call (download + load + first inference): ${coldMs.toFixed(0)}ms`);
  for (const line of coldLogs) {
    console.log(`[cold] ${line}`);
  }
  console.log(`[cold] first-score sanity: ${firstScores[0].toFixed(4)}`);
  console.log("");

  // ------- 2. Warm steady-state at several batch sizes ----------------
  // Run once to separate the "first-inference after load" slow tick from
  // the steady-state numbers we care about.
  await scoreQueryPassagePairs(QUERY, PASSAGES_POOL.slice(0, 3), { logger: () => {} });

  for (const size of [1, 5, 10, 20]) {
    await benchBatch(runs, size, `[warm, n=${size}]`);
  }

  // ------- 3. One full relevance demonstration ------------------------
  console.log("");
  console.log("[demo] query:", QUERY);
  const demoScores = await scoreQueryPassagePairs(QUERY, PASSAGES_POOL, { logger: () => {} });
  const ranked = PASSAGES_POOL.map((p, i) => ({ p, s: demoScores[i] }))
    .toSorted((a, b) => b.s - a.s)
    .slice(0, 5);
  console.log("[demo] top 5:");
  for (const { p, s } of ranked) {
    console.log(`   ${s.toFixed(3)}  ${p.slice(0, 80)}${p.length > 80 ? "..." : ""}`);
  }
}

main().catch((err) => {
  console.error("[bench] failed:", err);
  process.exit(1);
});
