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
  minScore: number;
  timeDecay: {
    enabled: boolean;
    halfLifeDays: number;
    source?: "filename" | "mtime";
  };
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

export const DEFAULT_SCOPE_CHUNKING: ScopeChunkingConfig = {
  maxChunkSize: 2000,
  minChunkSize: 200,
  overlapSize: 200,
  maxLinesPerChunk: 40,
  semanticSplit: true,
};

export const DEFAULT_WORKSPACE_SCOPE: WorkspaceScopeConfig = {
  enabled: false,
  memoryDir: "memory",
  chunking: DEFAULT_SCOPE_CHUNKING,
  recall: {
    candidatePool: 20,
    maxItems: 3,
    maxChars: 600,
    contextWindowChunks: 1,
    minScore: 0.3,
    timeDecay: { enabled: false, halfLifeDays: 365 },
  },
  sync: { intervalMinutes: 10, onStartup: true },
  excludeGlobs: ["memory/.archived/**", "memory/.dreams/**"],
};

export const DEFAULT_DAILY_SCOPE: DailyScopeConfig = {
  enabled: false,
  chunking: DEFAULT_SCOPE_CHUNKING,
  recall: {
    candidatePool: 20,
    maxItems: 3,
    maxChars: 600,
    contextWindowChunks: 1,
    minScore: 0.3,
    timeDecay: { enabled: true, halfLifeDays: 30, source: "filename" },
  },
  sync: { intervalMinutes: 5, onStartup: true },
};

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

  const chunked = chunkDocument(text, chunkingConfig);
  if (chunked.chunks.length === 0) {
    return { indexed: false, chunks: 0 };
  }

  // Track approximate line ranges by searching for chunk start positions.
  // We walk the original text forward so each chunk gets a monotonic cursor.
  const lineRanges: Array<{ start: number; end: number }> = [];
  {
    let cursor = 0;
    for (const chunk of chunked.chunks) {
      const head = chunk.slice(0, Math.min(chunk.length, 60)).trimStart();
      const found = head.length > 0 ? text.indexOf(head, cursor) : cursor;
      const startOffset = found >= 0 ? found : cursor;
      const endOffset = Math.min(text.length, startOffset + chunk.length);
      lineRanges.push({
        start: countLinesUpTo(text, startOffset),
        end: countLinesUpTo(text, endOffset),
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
  const multiplier = 0.5 ** (ageDays / halfLifeDays);
  return fused * multiplier;
}

export interface ScopedRetrieveOptions {
  query: string;
  recall: ScopeRecallConfig;
  store: MemoryStore;
  embedder: Embedder;
  nowMs?: number;
  log?: (msg: string) => void;
}

/**
 * Hybrid (vector + BM25) retrieval against a single scope store, with
 * optional time decay and context-window expansion.
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

  // Fuse rankings (70/30 vector/BM25 — matches main retriever default).
  const merged = mergeRankings(vectorHits, bm25Hits, 0.7, 0.3);

  // Apply time decay if configured.
  const halfLife = recall.timeDecay.enabled ? recall.timeDecay.halfLifeDays : 0;
  const withDecay: Array<{
    entry: MemoryEntry;
    fused: number;
    vec: number;
    bm: number;
    decayed: number;
    multiplier: number;
  }> = [];
  for (const m of merged.values()) {
    const decayed =
      halfLife > 0 ? applyTimeDecay(m.fused, m.entry.timestamp, halfLife, nowMs) : m.fused;
    const multiplier = m.fused > 0 ? decayed / m.fused : 1;
    withDecay.push({ ...m, decayed, multiplier });
  }

  // Sort by decayed score, pick top candidatePool.
  const diversified = withDecay.toSorted((a, b) => b.decayed - a.decayed).slice(0, candidatePool);

  // Dedup & keep unique source_file diversity (prefer different files first).
  const seenFiles = new Map<string, number>();

  const results: ScopedRetrievalResult[] = [];
  // First pass: pick best-per-file for up to maxItems.
  for (const row of diversified) {
    if (results.length >= recall.maxItems) {
      break;
    }
    const meta = parseMetadata(row.entry);
    const src = (meta.source_file as string) || "unknown";
    if (seenFiles.has(src)) {
      continue;
    }
    seenFiles.set(src, 1);
    results.push(await expandContext(row, meta, store, recall, nowMs));
  }
  // Second pass: fill remaining slots with runners-up (may be same file).
  if (results.length < recall.maxItems) {
    for (const row of diversified) {
      if (results.length >= recall.maxItems) {
        break;
      }
      if (results.some((r) => r.entry.id === row.entry.id)) {
        continue;
      }
      const meta = parseMetadata(row.entry);
      results.push(await expandContext(row, meta, store, recall, nowMs));
    }
  }
  return results;
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
