/**
 * Workspace & Daily Memory Indexer + Retriever.
 *
 * Indexes curated markdown files under an agent's workspace `memory/` directory
 * into per-agent sub-DBs (`__workspace__` and `__dailies__`) for auto-recall.
 *
 * Scopes:
 *   - workspace : MEMORY.md + memory-*.md + other non-daily *.md
 *   - daily     : memory/YYYY-MM-DD(-slug)?.md journal files
 *
 * Stored as standard MemoryEntry rows in a MemoryStore so we can reuse the
 * existing hybrid search (vector + BM25) + FTS index. Key differences vs
 * extracted memories:
 *   - timestamp encodes the file's effective date (mtime or filename date),
 *     so vector-memory's existing time-decay formula applies naturally.
 *   - metadata carries { source_file, file_hash, chunk_index, chunk_total,
 *     line_start, line_end, file_date? } so we can do context-window
 *     expansion and efficient re-indexing.
 *   - category is always "other" (no LLM extraction happens here).
 */

import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import { basename, join, relative } from "node:path";
import { chunkDocument, type ChunkerConfig } from "./chunker.js";
import type { Embedder } from "./embedder.js";
import { rerankPassages, type RerankPassagesConfig } from "./rerank-shared.js";
import type { MemoryEntry, MemorySearchResult, MemoryStore } from "./store.js";

// ============================================================================
// Config Types
// ============================================================================

export type MemoryScope = "workspace" | "daily";

export interface ScopeChunkingConfig {
  maxChunkSize: number;
  minChunkSize: number;
  overlapSize: number;
  maxLinesPerChunk: number;
  semanticSplit: boolean;
}

export interface ScopeRecallConfig {
  candidatePool: number;
  maxItems: number;
  maxChars: number;
  contextWindowChunks: number;
  /**
   * Entry gate. Applied twice to mirror the conversations retriever:
   *   (a) vector search cutoff (drop items with cosine < minScore);
   *   (b) post-fusion floor (drop items whose fused vec+BM25 score < minScore).
   * Default 0.3.
   */
  minScore: number;
  /**
   * Post-rerank exit gate. After the cross-encoder rerank score is blended
   * with the fused baseline, items with the final blended score below this
   * threshold are dropped (conversations retriever uses the same default
   * 0.40). Set to 0 to disable.
   */
  hardMinScore?: number;
  timeDecay: {
    enabled: boolean;
    halfLifeDays: number;
    source?: "filename" | "mtime";
  };
  /**
   * Rerank blend weight for the cross-encoder score vs the fused vector+BM25
   * (time-decay-adjusted) score. `rerankWeight` in [0, 1]; the baseline
   * fusion score gets `1 - rerankWeight`. Default 0.6 (matches the
   * conversations retriever). Set to 0 to bypass rerank even when configured.
   */
  rerankWeight?: number;
}

export interface ScopeSyncConfig {
  intervalMinutes: number;
  onStartup: boolean;
}

export interface WorkspaceScopeConfig {
  enabled: boolean;
  memoryDir: string;
  chunking: ScopeChunkingConfig;
  recall: ScopeRecallConfig;
  sync: ScopeSyncConfig;
  excludeGlobs: string[];
}

export interface DailyScopeConfig {
  enabled: boolean;
  chunking: ScopeChunkingConfig;
  recall: ScopeRecallConfig;
  sync: ScopeSyncConfig;
}

/**
 * Chunking defaults tuned for small, rerank-friendly passages:
 *   - maxChunkSize 500: a single chunk ≈ one meaningful paragraph / bullet.
 *   - overlapSize 100:  keeps semantic boundaries fluid across chunks.
 * With `contextWindowChunks = 1` the recall item displayText becomes
 * roughly 3 × 500 = ~1500 chars — the size the cross-encoder reranker
 * sees and the size the caller injects into the prompt.
 */
export const DEFAULT_SCOPE_CHUNKING: ScopeChunkingConfig = {
  maxChunkSize: 500,
  minChunkSize: 100,
  overlapSize: 100,
  maxLinesPerChunk: 20,
  semanticSplit: true,
};

export const DEFAULT_WORKSPACE_SCOPE: WorkspaceScopeConfig = {
  enabled: false,
  memoryDir: "memory",
  chunking: DEFAULT_SCOPE_CHUNKING,
  recall: {
    candidatePool: 10,
    maxItems: 3,
    // 3 items × ~1500 chars each (chunk + prev + next) = ~4500 char block.
    maxChars: 4500,
    contextWindowChunks: 1,
    minScore: 0.3,
    hardMinScore: 0.4,
    timeDecay: { enabled: false, halfLifeDays: 365 },
    rerankWeight: 0.6,
  },
  sync: { intervalMinutes: 10, onStartup: true },
  excludeGlobs: ["memory/.archived/**", "memory/.dreams/**", "memory/dreaming/**"],
};

export const DEFAULT_DAILY_SCOPE: DailyScopeConfig = {
  enabled: false,
  chunking: DEFAULT_SCOPE_CHUNKING,
  recall: {
    candidatePool: 10,
    maxItems: 3,
    maxChars: 4500,
    contextWindowChunks: 1,
    minScore: 0.3,
    hardMinScore: 0.4,
    timeDecay: { enabled: true, halfLifeDays: 30, source: "filename" },
    rerankWeight: 0.6,
  },
  sync: { intervalMinutes: 5, onStartup: true },
};

// ============================================================================
// Index-time Noise Stripping
// ============================================================================

/**
 * Remove machine-generated metadata that would pollute embeddings and BM25.
 *
 * Targets, in order:
 *   1. Dreaming stage blocks — `<!-- openclaw:dreaming:<kind>:start -->` …
 *      `<!-- openclaw:dreaming:<kind>:end -->` (light / rem / future kinds).
 *      These contain candidate lists with `confidence: 0.00` and
 *      `evidence: memory/...md:4-5` lines that match almost every query
 *      via BM25 (keywords like "Candidate", "confidence", "evidence")
 *      without carrying real memory content.
 *   2. Labelled untrusted-metadata JSON blocks emitted by channels
 *      (Signal / Discord / etc) when a message is quoted into daily
 *      notes — e.g. `Conversation info (untrusted …): ```json {…} ```.`
 *      These carry `message_id` / `sender_id` / `e164` etc, which are
 *      useless for recall and skew BM25 toward channel ID noise.
 *   3. Bare `{ "message_id": …, "sender_id": … }` JSON fences that some
 *      raw session dumps include without the preceding label.
 *
 * We preserve the **original line count** by replacing each matched
 * region with as many `\n` as it contained. This keeps `line_start` /
 * `line_end` metadata accurate relative to the real file, so when the
 * recall UI shows `memory/xxx.md#L123-145` the line numbers still
 * match what the user sees in their editor.
 */
export function stripIndexNoise(text: string): string {
  const preserveLines = (match: string): string => {
    let n = 0;
    for (let i = 0; i < match.length; i++) {
      if (match.charCodeAt(i) === 10) {
        n++;
      }
    }
    return n > 0 ? "\n".repeat(n) : "";
  };
  let out = text.replace(
    /<!-- openclaw:dreaming:[a-z]+:start -->[\s\S]*?<!-- openclaw:dreaming:[a-z]+:end -->/g,
    preserveLines,
  );
  // Orphan-opener: dreamer crashed / was interrupted mid-write, leaving a
  // `:start -->` with no matching `:end -->`. Strip from the orphan tag up
  // to the next top-level heading (`\n# `) or end-of-text, so the dangling
  // `confidence:` / `evidence:` candidate lines don't bleed into real
  // memory content.
  out = out.replace(
    /<!-- openclaw:dreaming:[a-z]+:start -->[\s\S]*?(?=\n#{1,6}\s|$)/g,
    preserveLines,
  );
  // Orphan closer left behind when the matching start was trimmed earlier.
  out = out.replace(/<!-- openclaw:dreaming:[a-z]+:end -->\n?/g, preserveLines);
  out = out.replace(
    /(?:Conversation info|Sender|Replied message)\s*\(untrusted[^)]*\):\s*```json\s*\{[\s\S]*?\}\s*```/g,
    preserveLines,
  );
  out = out.replace(
    /```json\s*\{[^}]*"message_id"\s*:[^}]*"sender_id"\s*:[^}]*\}\s*```/g,
    preserveLines,
  );
  return out;
}

// ============================================================================
// File Discovery
// ============================================================================

const DAILY_BASENAME_RE = /^(\d{4})-(\d{2})-(\d{2})(?:-[^/]+)?\.md$/i;

export function isDailyBasename(name: string): boolean {
  return DAILY_BASENAME_RE.test(name);
}

export function parseDailyDateMs(name: string): number | null {
  const m = DAILY_BASENAME_RE.exec(name);
  if (!m) {
    return null;
  }
  // Noon UTC: avoids TZ-edge flips that would push a file into the next day.
  const iso = `${m[1]}-${m[2]}-${m[3]}T12:00:00Z`;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, "__DOUBLESTAR__")
    .replace(/\*/g, "[^/]*")
    .replace(/__DOUBLESTAR__/g, ".*");
  return new RegExp(`^${escaped}$`);
}

function matchesAnyGlob(relPath: string, globs: string[]): boolean {
  if (globs.length === 0) {
    return false;
  }
  return globs.some((g) => globToRegExp(g).test(relPath));
}

/**
 * Recursively walk a directory, returning absolute paths of regular files.
 * Skips directories whose name starts with `.` (hidden) or `__` (internal).
 */
async function walkMarkdownFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  let entries: import("node:fs").Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    const abs = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.startsWith(".") || entry.name.startsWith("__")) {
        continue;
      }
      out.push(...(await walkMarkdownFiles(abs)));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      out.push(abs);
    }
  }
  return out;
}

export interface DiscoveredFile {
  absPath: string;
  relPath: string;
  scope: MemoryScope;
  mtimeMs: number;
  dailyDateMs?: number;
}

export async function discoverMemoryFiles(
  workspaceDir: string,
  memoryDir: string,
  excludeGlobs: string[],
): Promise<DiscoveredFile[]> {
  const memRoot = join(workspaceDir, memoryDir);
  if (!existsSync(memRoot)) {
    return [];
  }
  const abs = await walkMarkdownFiles(memRoot);
  const out: DiscoveredFile[] = [];
  for (const p of abs) {
    const rel = relative(workspaceDir, p).replace(/\\/g, "/");
    if (matchesAnyGlob(rel, excludeGlobs)) {
      continue;
    }
    let mtimeMs = Date.now();
    try {
      mtimeMs = (await stat(p)).mtimeMs;
    } catch {}
    const base = basename(p);
    const dailyMs = parseDailyDateMs(base);
    if (dailyMs != null) {
      out.push({ absPath: p, relPath: rel, scope: "daily", mtimeMs, dailyDateMs: dailyMs });
    } else {
      out.push({ absPath: p, relPath: rel, scope: "workspace", mtimeMs });
    }
  }
  return out;
}

// ============================================================================
// Indexer
// ============================================================================

function sha1(s: string): string {
  return createHash("sha1").update(s).digest("hex");
}

function deterministicChunkId(scope: MemoryScope, relPath: string, chunkIndex: number): string {
  return sha1(`${scope}::${relPath}#${chunkIndex}`);
}

function countLinesUpTo(text: string, upTo: number): number {
  let n = 1;
  const limit = Math.min(upTo, text.length);
  for (let i = 0; i < limit; i++) {
    if (text.charCodeAt(i) === 10) {
      n++;
    }
  }
  return n;
}

export interface SyncStats {
  indexed: number;
  skipped: number;
  removed: number;
  failed: number;
  chunks: number;
}

export interface IndexerOptions {
  store: MemoryStore;
  embedder: Embedder;
  scope: MemoryScope;
  chunkingConfig: ChunkerConfig;
  log?: (msg: string) => void;
  /**
   * Map a file's effective date (in ms) onto the `timestamp` field of a
   * MemoryEntry. For workspace scope this is the file mtime; for daily scope
   * this is the filename-parsed midnight-UTC date (preferred) or mtime fallback.
   */
  timestampFor: (file: DiscoveredFile) => number;
}

/**
 * Index a single file: dedupe by file_hash, else delete existing chunks
 * and re-embed. No-ops when file hash unchanged.
 */
export async function indexFile(
  file: DiscoveredFile,
  opts: IndexerOptions,
): Promise<{ indexed: boolean; chunks: number }> {
  const { store, embedder, scope, chunkingConfig, timestampFor } = opts;
  const log = opts.log ?? (() => {});

  let raw: string;
  try {
    raw = await readFile(file.absPath, "utf-8");
  } catch (err) {
    log(`vector-memory/workspace: read failed ${file.relPath}: ${String(err)}`);
    throw err;
  }

  const text = raw.trim();
  if (text.length === 0) {
    // Empty file: purge any previously-indexed chunks.
    const removed = await store.deleteByMetadataKV("source_file", file.relPath);
    if (removed > 0) {
      log(`vector-memory/workspace: [${scope}] emptied ${file.relPath} (purged ${removed})`);
    }
    return { indexed: false, chunks: 0 };
  }

  const fileHash = sha1(text);

  // Fast path: if any existing chunk from this file shares the same hash,
  // skip re-embedding entirely.
  const existing = await store.findByMetadataKV("source_file", file.relPath, 50);
  if (existing.length > 0) {
    try {
      const meta = JSON.parse(existing[0].metadata || "{}");
      if (meta.file_hash === fileHash) {
        return { indexed: false, chunks: existing.length };
      }
    } catch {}
    // Hash changed: drop stale chunks before inserting fresh ones.
    await store.deleteByMetadataKV("source_file", file.relPath);
  }

  // Strip machine-generated noise (dream candidate blocks, untrusted-metadata
  // JSON envelopes) before chunking so embeddings and BM25 index only real
  // memory content. `stripIndexNoise` preserves newline count, so the
  // `line_start` / `line_end` metadata still maps to the same lines in the
  // on-disk file — users clicking `memory/xxx.md#L123` still land correctly.
  const indexText = stripIndexNoise(text);
  const chunked = chunkDocument(indexText, chunkingConfig);
  if (chunked.chunks.length === 0) {
    return { indexed: false, chunks: 0 };
  }

  const lineRanges: Array<{ start: number; end: number }> = [];
  {
    let cursor = 0;
    for (const chunk of chunked.chunks) {
      const head = chunk.slice(0, Math.min(chunk.length, 60)).trimStart();
      const found = head.length > 0 ? indexText.indexOf(head, cursor) : cursor;
      const startOffset = found >= 0 ? found : cursor;
      const endOffset = Math.min(indexText.length, startOffset + chunk.length);
      lineRanges.push({
        start: countLinesUpTo(indexText, startOffset),
        end: countLinesUpTo(indexText, endOffset),
      });
      cursor = Math.max(
        cursor,
        startOffset + Math.max(1, chunk.length - chunkingConfig.overlapSize),
      );
    }
  }

  const vectors = await embedder.embedBatch(chunked.chunks);
  if (vectors.length !== chunked.chunks.length) {
    throw new Error(
      `vector count mismatch: got ${vectors.length}, expected ${chunked.chunks.length}`,
    );
  }

  const ts = timestampFor(file);
  const entries: MemoryEntry[] = chunked.chunks.map((chunkText, idx) => {
    const metadata: Record<string, unknown> = {
      source: scope === "daily" ? "daily_memory" : "workspace_memory",
      source_file: file.relPath,
      file_hash: fileHash,
      chunk_index: idx,
      chunk_total: chunked.chunks.length,
      line_start: lineRanges[idx]?.start ?? 1,
      line_end: lineRanges[idx]?.end ?? 1,
      l0_abstract: chunkText.slice(0, 180),
      memory_category: "other",
    };
    if (scope === "daily" && file.dailyDateMs != null) {
      metadata.file_date = new Date(file.dailyDateMs).toISOString().split("T")[0];
    }
    return {
      id: deterministicChunkId(scope, file.relPath, idx),
      text: chunkText,
      vector: vectors[idx],
      category: "other",
      importance: 0.5,
      timestamp: ts,
      metadata: JSON.stringify(metadata),
    };
  });

  await store.bulkAdd(entries);
  return { indexed: true, chunks: entries.length };
}

/**
 * Sync all files of a given scope for one agent. Handles:
 *   - New files → index
 *   - Changed files (hash diff) → re-index
 *   - Deleted files → purge
 */
export async function syncScope(args: {
  workspaceDir: string;
  memoryDir: string;
  excludeGlobs: string[];
  scope: MemoryScope;
  store: MemoryStore;
  embedder: Embedder;
  chunkingConfig: ChunkerConfig;
  recallConfig: ScopeRecallConfig;
  log?: (msg: string) => void;
}): Promise<SyncStats> {
  const log = args.log ?? (() => {});
  const stats: SyncStats = { indexed: 0, skipped: 0, removed: 0, failed: 0, chunks: 0 };

  const all = await discoverMemoryFiles(args.workspaceDir, args.memoryDir, args.excludeGlobs);
  const relevant = all.filter((f) => f.scope === args.scope);
  const relevantSet = new Set(relevant.map((f) => f.relPath));

  const timestampFor = (f: DiscoveredFile): number => {
    if (args.scope === "daily" && args.recallConfig.timeDecay.source !== "mtime") {
      return f.dailyDateMs ?? f.mtimeMs;
    }
    return f.mtimeMs;
  };

  for (const file of relevant) {
    try {
      const result = await indexFile(file, {
        store: args.store,
        embedder: args.embedder,
        scope: args.scope,
        chunkingConfig: args.chunkingConfig,
        log,
        timestampFor,
      });
      if (result.indexed) {
        stats.indexed++;
        stats.chunks += result.chunks;
      } else {
        stats.skipped++;
        stats.chunks += result.chunks;
      }
    } catch (err) {
      stats.failed++;
      log(`vector-memory/workspace: [${args.scope}] index failed ${file.relPath}: ${String(err)}`);
    }
  }

  // Purge chunks of files that no longer exist on disk (or were moved out of
  // the relevant scope, e.g. renamed from MEMORY.md → 2026-04-26-memory.md).
  try {
    const known = new Set<string>();
    for await (const batch of args.store.iterateAll()) {
      for (const entry of batch) {
        try {
          const meta = JSON.parse(entry.metadata || "{}");
          const src = meta.source_file as string | undefined;
          if (typeof src === "string") {
            known.add(src);
          }
        } catch {}
      }
    }
    for (const src of known) {
      if (!relevantSet.has(src)) {
        const n = await args.store.deleteByMetadataKV("source_file", src);
        if (n > 0) {
          stats.removed += n;
          log(`vector-memory/workspace: [${args.scope}] purged ${n} stale chunks from ${src}`);
        }
      }
    }
  } catch (err) {
    log(`vector-memory/workspace: [${args.scope}] purge sweep failed: ${String(err)}`);
  }

  return stats;
}

// ============================================================================
// Retriever
// ============================================================================

export interface ScopedRetrievalResult {
  entry: MemoryEntry;
  score: number;
  rawVectorScore: number;
  rawBm25Score: number;
  decayMultiplier: number;
  displayText: string;
  sourceFile: string;
  chunkIndex: number;
  chunkTotal: number;
  lineStart: number;
  lineEnd: number;
  fileDate?: string;
  /** Normalized rerank score in [0, 1], if reranker ran. */
  rerankScore?: number;
  /** Which backend produced `rerankScore`. */
  rerankMethod?: string;
}

function parseMetadata(entry: MemoryEntry): Record<string, unknown> {
  try {
    return JSON.parse(entry.metadata || "{}");
  } catch {
    return {};
  }
}

function mergeRankings(
  vector: MemorySearchResult[],
  bm25: MemorySearchResult[],
  vectorWeight: number,
  bm25Weight: number,
): Map<string, { entry: MemoryEntry; fused: number; vec: number; bm: number }> {
  const merged = new Map<string, { entry: MemoryEntry; fused: number; vec: number; bm: number }>();
  for (const r of vector) {
    merged.set(r.entry.id, {
      entry: r.entry,
      fused: r.score * vectorWeight,
      vec: r.score,
      bm: 0,
    });
  }
  for (const r of bm25) {
    const existing = merged.get(r.entry.id);
    if (existing) {
      existing.fused += r.score * bm25Weight;
      existing.bm = r.score;
    } else {
      merged.set(r.entry.id, {
        entry: r.entry,
        fused: r.score * bm25Weight,
        vec: 0,
        bm: r.score,
      });
    }
  }
  return merged;
}

/**
 * Soft time-decay with a 50% floor — matches `retriever.ts` `applyTimeDecay`
 * used by the conversations scope so all three memory scopes share the same
 * decay shape:
 *
 *   factor = 0.5 + 0.5 * exp(-ageDays / halfLife)
 *   score  = fused * factor        (factor ∈ [0.5, 1.0])
 *
 * Because `factor` is bounded below by 0.5, even very old items keep at
 * least half their fused score — so they can still be recalled when the
 * candidate pool is otherwise empty. This is intentional: purely
 * exponential half-life (0.5^n) was erasing 3-month-old daily entries
 * entirely (age=90d, halfLife=30d → ×0.125), which hid usable memory.
 *
 * At age = halfLife:   factor ≈ 0.684
 * At age = 2*halfLife: factor ≈ 0.568
 * At age → ∞:          factor = 0.50
 */
function applyTimeDecay(
  fused: number,
  timestampMs: number,
  halfLifeDays: number,
  nowMs: number,
): number {
  if (!Number.isFinite(halfLifeDays) || halfLifeDays <= 0) {
    return fused;
  }
  const ageDays = Math.max(0, (nowMs - timestampMs) / (24 * 3600_000));
  const factor = 0.5 + 0.5 * Math.exp(-ageDays / halfLifeDays);
  return fused * factor;
}

export interface ScopedRetrieveOptions {
  query: string;
  recall: ScopeRecallConfig;
  store: MemoryStore;
  embedder: Embedder;
  nowMs?: number;
  log?: (msg: string) => void;
  /**
   * Optional rerank backend. When provided, the full candidate pool
   * (context-window-expanded into displayText) is sent to the reranker
   * before overlap dedup + top-K selection.
   */
  rerank?: RerankPassagesConfig;
}

/**
 * Hybrid (vector + BM25) retrieval against a single scope store, then:
 *   1. Fuse + time-decay the candidate pool (default 20 items).
 *   2. Expand every candidate into its full displayText (chunk + N chunks
 *      before/after, controlled by `contextWindowChunks`). This is the
 *      passage the reranker scores AND the passage the caller injects.
 *   3. (Optional) Rerank the expanded passages with a cross-encoder and
 *      blend the rerank score with the fused+decayed score.
 *   4. Overlap dedup — if two top items' chunk-index ranges intersect
 *      (same source_file, |Δindex| ≤ 2·contextWindowChunks), keep the
 *      higher-ranked one and backfill with the next non-overlapping
 *      candidate (so content isn't repeated in the injected block).
 *   5. Return top `maxItems`.
 */
export async function retrieveScope(opts: ScopedRetrieveOptions): Promise<ScopedRetrievalResult[]> {
  const { query, recall, store, embedder } = opts;
  const nowMs = opts.nowMs ?? Date.now();
  const log = opts.log ?? (() => {});

  const candidatePool = Math.max(recall.maxItems, recall.candidatePool);

  let queryVec: number[];
  try {
    queryVec = await embedder.embedQuery(query);
  } catch (err) {
    log(`vector-memory/workspace: embed query failed: ${String(err)}`);
    return [];
  }

  const [vectorHits, bm25Hits] = await Promise.all([
    store.vectorSearch(queryVec, candidatePool, recall.minScore).catch((err) => {
      log(`vector-memory/workspace: vectorSearch failed: ${String(err)}`);
      return [] as MemorySearchResult[];
    }),
    store.bm25Search(query, candidatePool).catch((err) => {
      log(`vector-memory/workspace: bm25Search failed: ${String(err)}`);
      return [] as MemorySearchResult[];
    }),
  ]);

  if (vectorHits.length === 0 && bm25Hits.length === 0) {
    return [];
  }

  // 1) Fuse rankings (70/30 vector/BM25 — matches main retriever default).
  const merged = mergeRankings(vectorHits, bm25Hits, 0.7, 0.3);

  // 1a) Post-fusion entry gate (mirrors conversations retriever):
  //     - fused score >= minScore: drop items whose combined signal is weak.
  //     - raw-signal floor: require vec >= 0.45 OR bm25 >= 0.5 so we don't
  //       admit BM25-only noise that recency/rerank boosts would inflate.
  //     Items failing the floor are dropped BEFORE we spend rerank compute.
  let fusedEntries = [...merged.values()].filter((m) => m.fused >= recall.minScore);
  const droppedByMinScore = merged.size - fusedEntries.length;
  const beforeFloor = fusedEntries.length;
  fusedEntries = fusedEntries.filter((m) => m.vec >= 0.45 || m.bm >= 0.5);
  const droppedByFloor = beforeFloor - fusedEntries.length;
  if (droppedByMinScore > 0 || droppedByFloor > 0) {
    log(
      `vector-memory/workspace: gate filtered minScore=-${droppedByMinScore} rawFloor=-${droppedByFloor} (kept ${fusedEntries.length}/${merged.size})`,
    );
  }
  if (fusedEntries.length === 0) {
    return [];
  }

  // 2) Time decay.
  const halfLife = recall.timeDecay.enabled ? recall.timeDecay.halfLifeDays : 0;
  const withDecay: Array<{
    entry: MemoryEntry;
    fused: number;
    vec: number;
    bm: number;
    decayed: number;
    multiplier: number;
  }> = [];
  for (const m of fusedEntries) {
    const decayed =
      halfLife > 0 ? applyTimeDecay(m.fused, m.entry.timestamp, halfLife, nowMs) : m.fused;
    const multiplier = m.fused > 0 ? decayed / m.fused : 1;
    withDecay.push({ ...m, decayed, multiplier });
  }

  // 3) Top candidatePool by decayed score.
  const pool = withDecay.toSorted((a, b) => b.decayed - a.decayed).slice(0, candidatePool);

  // 4) Expand each candidate into its full displayText (chunk + context).
  //    We do this BEFORE rerank so the reranker sees the same passage the
  //    caller will inject — cross-encoders care about full context.
  const expanded: ScopedRetrievalResult[] = await Promise.all(
    pool.map((row) => expandContext(row, parseMetadata(row.entry), store, recall, nowMs)),
  );

  // 5) Optional rerank over the full candidatePool (not just limit*N).
  const rerankWeight = Math.max(0, Math.min(1, recall.rerankWeight ?? 0.6));
  let reranked: ScopedRetrievalResult[] = expanded;
  if (opts.rerank && opts.rerank.method !== "none" && rerankWeight > 0 && expanded.length > 0) {
    const tRerank = performance.now();
    const rr = await rerankPassages(
      query,
      expanded.map((r) => r.displayText),
      opts.rerank,
    );
    const rerankMs = Math.round(performance.now() - tRerank);
    if (rr) {
      reranked = expanded
        .map((item, idx) => {
          const rrScore = rr.scores[idx] ?? 0;
          const blended = rrScore * rerankWeight + item.score * (1 - rerankWeight);
          return {
            ...item,
            score: blended,
            rerankScore: rrScore,
            rerankMethod: rr.method,
          };
        })
        .toSorted((a, b) => b.score - a.score);
      log(
        `vector-memory/workspace: rerank ${rr.method} over ${expanded.length} docs in ${rerankMs}ms`,
      );
    } else {
      log(`vector-memory/workspace: rerank returned null for ${expanded.length} docs`);
    }
  }

  // 5a) Post-rerank exit gate (mirrors conversations retriever's hardMinScore).
  //     Drops items whose blended rerank+fusion score is below threshold —
  //     prevents low-quality reranker tails from sneaking into top-K just
  //     because the pool ran out of strong candidates.
  const hardMin = recall.hardMinScore ?? 0;
  if (hardMin > 0 && reranked.length > 0) {
    const before = reranked.length;
    reranked = reranked.filter((r) => r.score >= hardMin);
    if (reranked.length < before) {
      log(
        `vector-memory/workspace: hardMinScore=${hardMin.toFixed(2)} dropped ${before - reranked.length}/${before}`,
      );
    }
  }

  // 6) Overlap dedup — remove any item whose chunk-window intersects a
  //    higher-ranked item's window (same file, |Δindex| ≤ 2·ctxWin). Then
  //    backfill with the next non-overlapping candidate. This avoids
  //    repeating the same text across top-K when a file has consecutive
  //    high-scoring chunks.
  const ctxWin = Math.max(0, recall.contextWindowChunks);
  const selected: ScopedRetrievalResult[] = [];
  const rejectedForOverlap = new Set<string>();
  for (const cand of reranked) {
    if (selected.length >= recall.maxItems) {
      break;
    }
    const overlaps = selected.some(
      (s) =>
        s.sourceFile === cand.sourceFile && Math.abs(s.chunkIndex - cand.chunkIndex) <= 2 * ctxWin,
    );
    if (overlaps) {
      rejectedForOverlap.add(cand.entry.id);
      continue;
    }
    selected.push(cand);
  }
  if (rejectedForOverlap.size > 0) {
    log(`vector-memory/workspace: overlap dedup skipped ${rejectedForOverlap.size} candidate(s)`);
  }
  return selected;
}

async function expandContext(
  row: {
    entry: MemoryEntry;
    decayed: number;
    multiplier: number;
    vec: number;
    bm: number;
  },
  meta: Record<string, unknown>,
  store: MemoryStore,
  recall: ScopeRecallConfig,
  _nowMs: number,
): Promise<ScopedRetrievalResult> {
  const src = (meta.source_file as string) || "unknown";
  const chunkIdx = Number(meta.chunk_index ?? 0);
  const chunkTotal = Number(meta.chunk_total ?? 1);
  const lineStart = Number(meta.line_start ?? 1);
  const lineEnd = Number(meta.line_end ?? 1);
  const fileDate = meta.file_date as string | undefined;

  let displayText = row.entry.text;

  if (recall.contextWindowChunks > 0 && chunkTotal > 1) {
    const neighbours = await store.findByMetadataKV("source_file", src, chunkTotal + 2);
    const byIndex = new Map<number, MemoryEntry>();
    for (const n of neighbours) {
      try {
        const nm = JSON.parse(n.metadata || "{}");
        const idx = Number(nm.chunk_index ?? -1);
        if (idx >= 0) {
          byIndex.set(idx, n);
        }
      } catch {}
    }
    const pieces: string[] = [];
    for (let i = chunkIdx - recall.contextWindowChunks; i < chunkIdx; i++) {
      const n = byIndex.get(i);
      if (n) {
        pieces.push(n.text);
      }
    }
    pieces.push(row.entry.text);
    for (let i = chunkIdx + 1; i <= chunkIdx + recall.contextWindowChunks; i++) {
      const n = byIndex.get(i);
      if (n) {
        pieces.push(n.text);
      }
    }
    displayText = pieces.join("\n\n");
  }

  return {
    entry: row.entry,
    score: row.decayed,
    rawVectorScore: row.vec,
    rawBm25Score: row.bm,
    decayMultiplier: row.multiplier,
    displayText,
    sourceFile: src,
    chunkIndex: chunkIdx,
    chunkTotal,
    lineStart,
    lineEnd,
    fileDate,
  };
}

// ============================================================================
// Block Formatter (shared by all 3 scopes)
// ============================================================================

export interface BlockItem {
  sourceFile: string;
  lineStart: number;
  lineEnd: number;
  text: string;
}

/**
 * Format a list of recalled items into the `<from-*-memory>` sub-block body
 * content. Returns null if no items fit the budget.
 */
export function formatScopedBlock(
  items: BlockItem[],
  maxChars: number,
): { body: string; used: number } | null {
  if (items.length === 0) {
    return null;
  }
  const lines: string[] = [];
  let used = 0;
  for (const item of items) {
    const remaining = maxChars - used;
    if (remaining <= 32) {
      break;
    }
    const loc = `${item.sourceFile}#L${item.lineStart}-${item.lineEnd}`;
    const snippet = item.text
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, Math.max(32, remaining - loc.length - 6));
    const line = `- ${loc}:\n  ${snippet}`;
    lines.push(line);
    used += snippet.length + loc.length;
  }
  if (lines.length === 0) {
    return null;
  }
  return { body: lines.join("\n"), used };
}
