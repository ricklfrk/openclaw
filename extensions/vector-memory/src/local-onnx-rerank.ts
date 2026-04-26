/**
 * In-process ONNX cross-encoder reranker.
 *
 * Wraps `@huggingface/transformers` (onnxruntime-node) so `retriever.ts`
 * can score (query, passage) pairs locally without a docker/TEI server.
 *
 * Design decisions:
 *  - Single daemon-lifetime singleton. Each agent / session shares the same
 *    model instance; first use pays the ~5–15s cold load, subsequent calls
 *    reuse it. We do NOT load at module import time — only on first score
 *    request — so agents that never enable `local-onnx` don't pay the
 *    ~500–800MB RSS cost or first-download bandwidth.
 *  - Dynamic import of the transformers package so a missing / broken
 *    install manifests as a runtime warning (with BM25-lite fallback in
 *    the caller) rather than blocking plugin registration.
 *  - Concurrency-safe lazy init: concurrent first calls share one loader
 *    promise.
 *  - Default model: onnx-community/bge-reranker-v2-m3-ONNX
 *    (multilingual, ~580MB quantized, XLM-RoBERTa seq-classification head).
 *  - We return RAW logits (higher = more relevant). The caller is
 *    responsible for any normalization or blending with fused scores.
 *
 * Non-goals (for now):
 *  - Batching across multiple concurrent rerank calls. Each call runs its
 *    own tokenize + forward pass; concurrent callers race CPU.
 *  - GPU / CoreML execution provider wiring. Default WASM/CPU backend is
 *    fast enough for 10–20 docs at prompt time; revisit only if log
 *    timings show rerank > 200ms in practice.
 */

// Use import type to avoid a hard require at module-load time. The real
// module is resolved dynamically the first time we load the model.
type TransformersModule = typeof import("@huggingface/transformers");

// Structural types capturing just the surface we actually touch so we
// can avoid `any` while still not coupling to transformers.js's sprawling
// internal class hierarchy (which is a moving target between minor
// versions). The real runtime objects are AutoTokenizer / AutoModelFor*
// instances returned by dynamic import; at call sites we only need the
// callable shape + a logits tensor shape back from the model.
interface LogitsTensor {
  data: ArrayLike<number>;
  dims?: number[];
}
// transformers.js's real Tokenizer/Model class types are sprawling and
// churn between minor versions. We hold the actual instances as opaque
// `unknown` boxes and re-cast to minimal call signatures at the two call
// sites below — this keeps the public surface honest without importing
// the full class hierarchy or falling back to `any`.
type TokenizerCallable = (...args: unknown[]) => unknown;
type ModelCallable = (...args: unknown[]) => Promise<{ logits: LogitsTensor }>;

interface LoadedReranker {
  modelId: string;
  dtype: LocalOnnxDtype;
  tokenizer: unknown;
  model: unknown;
  loadedAt: number;
}

let loaderPromise: Promise<LoadedReranker> | null = null;
let currentModelId: string | null = null;
let currentDtype: LocalOnnxDtype | null = null;

export const DEFAULT_LOCAL_ONNX_RERANK_MODEL = "onnx-community/bge-reranker-v2-m3-ONNX";

/**
 * Precision variant to load. HuggingFace repos (e.g. the bge-reranker
 * ONNX one) ship MULTIPLE precisions: full fp32 `model.onnx` (with an
 * external ~2GB `model.onnx_data` weight blob), plus inline-weight
 * variants `model_fp16.onnx`, `model_quantized.onnx`, `model_int8.onnx`,
 * etc. We default to "q8" — int8 quantization with inline weights —
 * because:
 *   1. The file is self-contained (no external data file to miss).
 *   2. Size drops from ~2.2GB to ~580MB.
 *   3. int8 on modern CPUs is faster than fp32 with <1% ranking
 *      quality loss on MTEB-style benchmarks for this model family.
 * Override via opts.dtype if you need better fidelity ("fp16") or
 * smaller footprint ("q4"/"bnb4"). Caller is responsible for making
 * sure the chosen variant exists in the repo.
 */
export type LocalOnnxDtype = "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "q4f16" | "bnb4";

export const DEFAULT_LOCAL_ONNX_DTYPE: LocalOnnxDtype = "q8";

/**
 * Upper bound on the (query, passage) pair token length we feed to the
 * model. bge-reranker-v2-m3 supports up to 8192 but longer sequences are
 * quadratically expensive at prompt time; 512 is the standard training
 * regime and matches how most passages (~200–400 tokens) look.
 */
const MAX_LENGTH = 512;

export interface LocalOnnxRerankOptions {
  modelId?: string;
  dtype?: LocalOnnxDtype;
  /** Optional structured logger. Receives level ("info"|"error") and a human-readable message. */
  logger?: (level: "info" | "error", message: string) => void;
}

/**
 * Score a list of passages against a single query. Returns raw logits
 * in the SAME order as `passages`.
 *
 * Throws if the model fails to load or inference fails. Callers should
 * catch and fall back to a cheaper reranker (e.g. BM25-lite).
 */
export async function scoreQueryPassagePairs(
  query: string,
  passages: string[],
  opts: LocalOnnxRerankOptions = {},
): Promise<number[]> {
  if (passages.length === 0) {
    return [];
  }
  const modelId = opts.modelId || DEFAULT_LOCAL_ONNX_RERANK_MODEL;
  const dtype = opts.dtype || DEFAULT_LOCAL_ONNX_DTYPE;
  const { tokenizer, model } = await ensureLoaded(modelId, dtype, opts.logger);
  const tokenizerFn = tokenizer as TokenizerCallable;
  const modelFn = model as ModelCallable;

  // bge-reranker pair input: repeat the query once per passage and use
  // tokenizer's `text_pair` argument to let the tokenizer construct the
  // [CLS] query [SEP] passage [SEP] sequence natively (matches training).
  const queries = passages.map(() => query);
  const inputs = tokenizerFn(queries, {
    text_pair: passages,
    padding: true,
    truncation: true,
    max_length: MAX_LENGTH,
  });

  const output = await modelFn(inputs);
  const logits = output?.logits;
  if (!logits || !logits.data) {
    throw new Error("local-onnx rerank: model returned no logits");
  }
  // bge-reranker-v2-m3 emits a single score per pair (num_labels=1), so
  // logits.data has length === passages.length in row-major order.
  const scores: number[] = Array.from(logits.data);
  if (scores.length !== passages.length) {
    throw new Error(`local-onnx rerank: expected ${passages.length} scores, got ${scores.length}`);
  }
  return scores;
}

/**
 * Internal: ensure the model for `modelId` is loaded. Concurrent callers
 * share the same loader promise; a subsequent call with a DIFFERENT
 * modelId reloads (rare, but supported so tests / live config reload
 * work without a daemon restart).
 */
async function ensureLoaded(
  modelId: string,
  dtype: LocalOnnxDtype,
  logger?: LocalOnnxRerankOptions["logger"],
): Promise<LoadedReranker> {
  if (loaderPromise && currentModelId === modelId && currentDtype === dtype) {
    return loaderPromise;
  }

  currentModelId = modelId;
  currentDtype = dtype;
  loaderPromise = (async () => {
    const tStart = Date.now();
    logger?.(
      "info",
      `local-onnx rerank: loading model ${modelId} (dtype=${dtype}; first use may download ~580MB)`,
    );

    let transformers: TransformersModule;
    try {
      transformers = await import("@huggingface/transformers");
    } catch (err) {
      throw new Error(
        `local-onnx rerank: failed to import @huggingface/transformers — is the dependency installed?`,
        { cause: err },
      );
    }

    const { AutoTokenizer, AutoModelForSequenceClassification, env } = transformers;

    // Persist the model cache OUTSIDE of node_modules so it survives
    // `rm -rf node_modules` / `pnpm install --force`. Order of precedence:
    //   1. explicit env.cacheDir the user already set (respect it)
    //   2. $HF_HOME env var (standard HuggingFace convention)
    //   3. ~/.cache/huggingface/transformers (Linux/macOS XDG-ish default)
    // We only override #1 when it still points at the default
    // node_modules/.../.cache path, so a user who already configured a
    // custom directory keeps their choice.
    try {
      const currentCache = (env as { cacheDir?: string | null }).cacheDir;
      const isDefaultCache =
        typeof currentCache === "string" && currentCache.includes("node_modules/");
      if (!currentCache || isDefaultCache) {
        const hfHome = process.env.HF_HOME;
        const home = process.env.HOME || process.env.USERPROFILE || "";
        const desired = hfHome
          ? `${hfHome}/transformers`
          : home
            ? `${home}/.cache/huggingface/transformers`
            : null;
        if (desired) {
          (env as { cacheDir: string }).cacheDir = desired;
          logger?.("info", `local-onnx rerank: model cache dir = ${desired}`);
        }
      }
    } catch (err) {
      // Non-fatal: fall back to whatever default transformers.js picks.
      logger?.("error", `local-onnx rerank: failed to set cache dir: ${String(err)}`);
    }

    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(modelId),
      // `dtype` tells transformers.js to pull the matching ONNX variant
      // (e.g. "q8" -> onnx/model_quantized.onnx). Without this it
      // defaults to fp32 which for large models requires a separate
      // model.onnx_data weight blob that isn't always fetched reliably.
      AutoModelForSequenceClassification.from_pretrained(modelId, { dtype }),
    ]);

    const loadedAt = Date.now();
    logger?.("info", `local-onnx rerank: model ${modelId} ready in ${loadedAt - tStart}ms`);

    return { modelId, dtype, tokenizer, model, loadedAt };
  })().catch((err) => {
    // Reset cache so the NEXT call retries (otherwise a transient failure
    // would permanently poison the reranker for this process).
    loaderPromise = null;
    currentModelId = null;
    currentDtype = null;
    throw err;
  });

  return loaderPromise;
}

/** Test-only: wipe the cached loader so tests can inject a fake module. */
export function __resetLocalOnnxRerankerForTesting(): void {
  loaderPromise = null;
  currentModelId = null;
  currentDtype = null;
}
