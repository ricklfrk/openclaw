/**
 * LanceDB Store with per-agent database isolation.
 * Simplified from memory-lancedb-pro: no scope management, no smart-metadata.
 */

import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, lstatSync, realpathSync, readdirSync } from "node:fs";
import { accessSync, constants } from "node:fs";
import { dirname, join } from "node:path";

// ============================================================================
// Lazy LanceDB Loading
// ============================================================================

type LanceDB = typeof import("@lancedb/lancedb");
let _lancedb: LanceDB | null = null;

async function loadLanceDB(): Promise<LanceDB> {
  if (_lancedb) {
    return _lancedb;
  }
  _lancedb = await import("@lancedb/lancedb");
  return _lancedb;
}

let _lockfileModule: {
  lock: (path: string, opts?: unknown) => Promise<() => Promise<void>>;
} | null = null;

async function loadLockfile(): Promise<typeof _lockfileModule & {}> {
  if (_lockfileModule) {
    return _lockfileModule;
  }
  const mod = await import("proper-lockfile");
  _lockfileModule = mod.default ?? mod;
  return _lockfileModule;
}

// ============================================================================
// Types
// ============================================================================

export type StoreCategory = "preference" | "fact" | "decision" | "entity" | "other";

export interface MemoryEntry {
  id: string;
  text: string;
  vector: number[];
  category: StoreCategory;
  importance: number;
  timestamp: number;
  metadata: string;
}

type LanceRecord = Record<string, unknown>;

export interface MemorySearchResult {
  entry: MemoryEntry;
  score: number;
}

export interface StoreConfig {
  dbPath: string;
  vectorDim: number;
}

// ============================================================================
// Utility Functions
// ============================================================================

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
}

function escapeSqlLiteral(value: string): string {
  return value.replace(/'/g, "''");
}

// ============================================================================
// Storage Path Validation
// ============================================================================

export function validateStoragePath(dbPath: string): string {
  let resolvedPath = dbPath;
  try {
    const stats = lstatSync(dbPath);
    if (stats.isSymbolicLink()) {
      try {
        resolvedPath = realpathSync(dbPath);
      } catch (err: unknown) {
        const e = err as { code?: string; message?: string };
        throw new Error(
          `dbPath "${dbPath}" is a dangling symlink. ${e.code ?? ""} ${e.message ?? ""}`,
          { cause: err },
        );
      }
    }
  } catch (err: unknown) {
    const e = err as { code?: string; message?: string };
    if (
      e.code !== "ENOENT" &&
      !(typeof e.message === "string" && e.message.includes("dangling symlink"))
    ) {
      // ignore other lstat errors
    } else if (e.message?.includes("dangling symlink")) {
      throw err;
    }
  }

  if (!existsSync(resolvedPath)) {
    try {
      mkdirSync(resolvedPath, { recursive: true });
    } catch (err: unknown) {
      const e = err as { code?: string; message?: string };
      throw new Error(
        `Failed to create dbPath "${resolvedPath}": ${e.code ?? ""} ${e.message ?? ""}`,
        { cause: err },
      );
    }
  }

  try {
    accessSync(resolvedPath, constants.W_OK);
  } catch (err: unknown) {
    const e = err as { code?: string; message?: string };
    throw new Error(
      `dbPath "${resolvedPath}" is not writable: ${e.code ?? ""} ${e.message ?? ""}`,
      { cause: err },
    );
  }

  return resolvedPath;
}

// ============================================================================
// Memory Store
// ============================================================================

const TABLE_NAME = "memories";

export class MemoryStore {
  private db: import("@lancedb/lancedb").Connection | null = null;
  private table: import("@lancedb/lancedb").Table | null = null;
  private initPromise: Promise<void> | null = null;
  private ftsIndexCreated = false;
  private updateQueue: Promise<void> = Promise.resolve();

  constructor(private readonly config: StoreConfig) {}

  private async runWithFileLock<T>(fn: () => Promise<T>): Promise<T> {
    const lockfile = await loadLockfile();
    const lockPath = join(this.config.dbPath, ".memory-write.lock");
    if (!existsSync(lockPath)) {
      try {
        mkdirSync(dirname(lockPath), { recursive: true });
      } catch {}
      try {
        const { writeFileSync } = await import("node:fs");
        writeFileSync(lockPath, "", { flag: "wx" });
      } catch {}
    }
    const release = await lockfile.lock(lockPath, {
      retries: { retries: 5, factor: 2, minTimeout: 100, maxTimeout: 2000 },
      stale: 10000,
    });
    try {
      return await fn();
    } finally {
      await release();
    }
  }

  get dbPath(): string {
    return this.config.dbPath;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.table) {
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }
    this.initPromise = this.doInitialize().catch((err) => {
      this.initPromise = null;
      throw err;
    });
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    const lancedb = await loadLanceDB();
    const db = await lancedb.connect(this.config.dbPath);

    let table: import("@lancedb/lancedb").Table;
    try {
      table = await db.openTable(TABLE_NAME);
    } catch {
      const schemaEntry: MemoryEntry = {
        id: "__schema__",
        text: "",
        vector: Array.from({ length: this.config.vectorDim }).fill(0) as number[],
        category: "other",
        importance: 0,
        timestamp: 0,
        metadata: "{}",
      };
      try {
        table = await db.createTable(TABLE_NAME, [schemaEntry as unknown as LanceRecord]);
        await table.delete('id = "__schema__"');
      } catch (createErr) {
        if (String(createErr).includes("already exists")) {
          table = await db.openTable(TABLE_NAME);
        } else {
          throw createErr;
        }
      }
    }

    // Validate vector dimensions
    const sample = await table.query().limit(1).toArray();
    if (sample.length > 0 && sample[0]?.vector?.length) {
      const existingDim = sample[0].vector.length;
      if (existingDim !== this.config.vectorDim) {
        throw new Error(
          `Vector dimension mismatch: table=${existingDim}, config=${this.config.vectorDim}`,
        );
      }
    }

    // Create FTS index for BM25
    try {
      const indices = await table.listIndices();
      const hasFts = indices?.some(
        (idx) =>
          (idx as { indexType?: string }).indexType === "FTS" ||
          (Array.isArray((idx as { columns?: unknown }).columns) &&
            (idx as { columns: string[] }).columns.includes("text")),
      );
      if (!hasFts) {
        const lb = await loadLanceDB();
        await table.createIndex("text", {
          config: (
            lb as unknown as { Index: { fts: () => Record<string, unknown> } }
          ).Index.fts() as never,
        });
      }
      this.ftsIndexCreated = true;
    } catch {
      this.ftsIndexCreated = false;
    }

    this.db = db;
    this.table = table;
  }

  get hasFtsSupport(): boolean {
    return this.ftsIndexCreated;
  }

  async store(entry: Omit<MemoryEntry, "id" | "timestamp">): Promise<MemoryEntry> {
    await this.ensureInitialized();
    const fullEntry: MemoryEntry = {
      ...entry,
      id: randomUUID(),
      timestamp: Date.now(),
      metadata: entry.metadata || "{}",
    };
    return this.runWithFileLock(async () => {
      await this.table!.add([fullEntry as unknown as LanceRecord]);
      return fullEntry;
    });
  }

  async hasId(id: string): Promise<boolean> {
    await this.ensureInitialized();
    const res = await this.table!.query()
      .select(["id"])
      .where(`id = '${escapeSqlLiteral(id)}'`)
      .limit(1)
      .toArray();
    return res.length > 0;
  }

  async getById(id: string): Promise<MemoryEntry | null> {
    await this.ensureInitialized();
    const rows = await this.table!.query()
      .where(`id = '${escapeSqlLiteral(id)}'`)
      .limit(1)
      .toArray();
    if (rows.length === 0) {
      return null;
    }
    const row = rows[0];
    return {
      id: row.id as string,
      text: row.text as string,
      vector: Array.from(row.vector as Iterable<number>),
      category: row.category as MemoryEntry["category"],
      importance: Number(row.importance),
      timestamp: Number(row.timestamp),
      metadata: (row.metadata as string) || "{}",
    };
  }

  async vectorSearch(vector: number[], limit = 5, minScore = 0.3): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();
    const safeLimit = clampInt(limit, 1, 20);
    const fetchLimit = Math.min(safeLimit * 10, 200);
    const query = this.table!.vectorSearch(vector).distanceType("cosine").limit(fetchLimit);
    const results = await query.toArray();
    const mapped: MemorySearchResult[] = [];

    for (const row of results) {
      const distance = Number(row._distance ?? 0);
      const score = 1 / (1 + distance);
      if (score < minScore) {
        continue;
      }

      // Skip invalidated (superseded) entries
      try {
        const meta = JSON.parse((row.metadata as string) || "{}");
        if (meta.invalidated_at) {
          continue;
        }
      } catch {}

      mapped.push({
        entry: {
          id: row.id as string,
          text: row.text as string,
          vector: row.vector as number[],
          category: row.category as MemoryEntry["category"],
          importance: Number(row.importance),
          timestamp: Number(row.timestamp),
          metadata: (row.metadata as string) || "{}",
        },
        score,
      });
      if (mapped.length >= safeLimit) {
        break;
      }
    }
    return mapped;
  }

  async bm25Search(query: string, limit = 5): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();
    const safeLimit = clampInt(limit, 1, 20);
    if (!this.ftsIndexCreated) {
      return [];
    }

    try {
      const results = await this.table!.search(query, "fts").limit(safeLimit).toArray();
      const mapped: MemorySearchResult[] = [];
      for (const row of results) {
        const rawScore = row._score != null ? Number(row._score) : 0;
        const normalizedScore = rawScore > 0 ? 1 / (1 + Math.exp(-rawScore / 5)) : 0.5;

        // Skip invalidated entries
        try {
          const meta = JSON.parse((row.metadata as string) || "{}");
          if (meta.invalidated_at) {
            continue;
          }
        } catch {}

        mapped.push({
          entry: {
            id: row.id as string,
            text: row.text as string,
            vector: row.vector as number[],
            category: row.category as MemoryEntry["category"],
            importance: Number(row.importance),
            timestamp: Number(row.timestamp),
            metadata: (row.metadata as string) || "{}",
          },
          score: normalizedScore,
        });
      }
      return mapped;
    } catch {
      return [];
    }
  }

  async update(
    id: string,
    updates: {
      text?: string;
      vector?: number[];
      importance?: number;
      category?: MemoryEntry["category"];
      metadata?: string;
    },
  ): Promise<MemoryEntry | null> {
    await this.ensureInitialized();
    return this.runWithFileLock(() =>
      this.runSerializedUpdate(async () => {
        const rows = await this.table!.query()
          .where(`id = '${escapeSqlLiteral(id)}'`)
          .limit(1)
          .toArray();
        if (rows.length === 0) {
          return null;
        }
        const row = rows[0];
        const original: MemoryEntry = {
          id: row.id as string,
          text: row.text as string,
          vector: Array.from(row.vector as Iterable<number>),
          category: row.category as MemoryEntry["category"],
          importance: Number(row.importance),
          timestamp: Number(row.timestamp),
          metadata: (row.metadata as string) || "{}",
        };

        const updated: MemoryEntry = {
          ...original,
          text: updates.text ?? original.text,
          vector: updates.vector ?? original.vector,
          category: updates.category ?? original.category,
          importance: updates.importance ?? original.importance,
          metadata: updates.metadata ?? original.metadata,
        };

        await this.table!.delete(`id = '${escapeSqlLiteral(id)}'`);
        try {
          await this.table!.add([updated as unknown as LanceRecord]);
        } catch (addError) {
          try {
            await this.table!.add([original as unknown as LanceRecord]);
          } catch {}
          throw addError;
        }
        return updated;
      }),
    );
  }

  private async runSerializedUpdate<T>(action: () => Promise<T>): Promise<T> {
    const previous = this.updateQueue;
    let release: (() => void) | undefined;
    const lock = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.updateQueue = previous.then(() => lock);
    await previous;
    try {
      return await action();
    } finally {
      release?.();
    }
  }

  async stats(): Promise<{ totalCount: number; categoryCounts: Record<string, number> }> {
    await this.ensureInitialized();
    const results = await this.table!.query().select(["category"]).toArray();
    const categoryCounts: Record<string, number> = {};
    for (const row of results) {
      const cat = row.category as string;
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    }
    return { totalCount: results.length, categoryCounts };
  }

  async listAll(opts?: {
    offset?: number;
    limit?: number;
    category?: string;
    sort?: "newest" | "oldest";
    /** Inclusive lower bound (ms epoch) */
    timestampFrom?: number;
    /** Inclusive upper bound (ms epoch) */
    timestampTo?: number;
    /** Return invalidated entries too (default: false) */
    includeInvalidated?: boolean;
  }): Promise<MemoryEntry[]> {
    await this.ensureInitialized();
    const offset = opts?.offset ?? 0;
    const limit = clampInt(opts?.limit ?? 20, 1, 500);
    const sortOrder = opts?.sort ?? "newest";
    const includeInvalidated = opts?.includeInvalidated ?? false;

    let q = this.table!.query().select([
      "id",
      "text",
      "vector",
      "category",
      "importance",
      "timestamp",
      "metadata",
    ]);
    const conditions: string[] = [];
    if (opts?.category) {
      conditions.push(`category = '${escapeSqlLiteral(opts.category)}'`);
    }
    if (opts?.timestampFrom != null) {
      conditions.push(`timestamp >= ${opts.timestampFrom}`);
    }
    if (opts?.timestampTo != null) {
      conditions.push(`timestamp <= ${opts.timestampTo}`);
    }
    if (conditions.length > 0) {
      q = q.where(conditions.join(" AND "));
    }

    // When a timestamp range is provided the result set is bounded;
    // fetch generously so app-layer sort covers the full window.
    // Without a range we must fetch enough for offset+limit.
    const hasTimeRange = opts?.timestampFrom != null || opts?.timestampTo != null;
    const fetchLimit = hasTimeRange
      ? Math.max(offset + limit, 2000)
      : Math.max(offset + limit, 500);
    const rows = await q.limit(fetchLimit).toArray();

    let mapped = rows.map((row) => ({
      id: row.id as string,
      text: row.text as string,
      vector: Array.from(row.vector as Iterable<number>),
      category: row.category as MemoryEntry["category"],
      importance: Number(row.importance),
      timestamp: Number(row.timestamp),
      metadata: (row.metadata as string) || "{}",
    }));

    if (!includeInvalidated) {
      mapped = mapped.filter((entry) => {
        try {
          const meta = JSON.parse(entry.metadata);
          return !meta.invalidated_at;
        } catch {
          return true;
        }
      });
    }

    mapped.sort((a, b) =>
      sortOrder === "newest" ? b.timestamp - a.timestamp : a.timestamp - b.timestamp,
    );

    return mapped.slice(offset, offset + limit);
  }

  /**
   * Find entries whose id starts with `prefix`.
   * Uses SQL LIKE pushdown so it works regardless of table size.
   */
  async findByIdPrefix(prefix: string, maxMatches = 10): Promise<MemoryEntry[]> {
    await this.ensureInitialized();
    if (!prefix || prefix.length < 4) {
      return [];
    }
    const safePrefix = escapeSqlLiteral(prefix);
    const rows = await this.table!.query()
      .where(`id LIKE '${safePrefix}%'`)
      .limit(maxMatches + 1)
      .toArray();
    return rows.map((row) => ({
      id: row.id as string,
      text: row.text as string,
      vector: Array.from(row.vector as Iterable<number>),
      category: row.category as MemoryEntry["category"],
      importance: Number(row.importance),
      timestamp: Number(row.timestamp),
      metadata: (row.metadata as string) || "{}",
    }));
  }

  /**
   * Iterate over all entries in batches. Suitable for full-table scans
   * (compact, maintenance) without loading the entire table into memory.
   */
  async *iterateAll(opts?: {
    batchSize?: number;
    includeInvalidated?: boolean;
  }): AsyncGenerator<MemoryEntry[]> {
    await this.ensureInitialized();
    const batchSize = clampInt(opts?.batchSize ?? 500, 50, 5000);
    const includeInvalidated = opts?.includeInvalidated ?? false;
    let batchOffset = 0;

    for (;;) {
      const rows = await this.table!.query()
        .select(["id", "text", "vector", "category", "importance", "timestamp", "metadata"])
        .limit(batchSize)
        .offset(batchOffset)
        .toArray();
      if (rows.length === 0) {
        break;
      }

      let batch = rows.map((row) => ({
        id: row.id as string,
        text: row.text as string,
        vector: Array.from(row.vector as Iterable<number>),
        category: row.category as MemoryEntry["category"],
        importance: Number(row.importance),
        timestamp: Number(row.timestamp),
        metadata: (row.metadata as string) || "{}",
      }));

      if (!includeInvalidated) {
        batch = batch.filter((entry) => {
          try {
            const meta = JSON.parse(entry.metadata);
            return !meta.invalidated_at;
          } catch {
            return true;
          }
        });
      }

      if (batch.length > 0) {
        yield batch;
      }

      batchOffset += rows.length;
      if (rows.length < batchSize) {
        break;
      }
    }
  }

  async deleteById(id: string): Promise<boolean> {
    await this.ensureInitialized();
    return this.runWithFileLock(async () => {
      const rows = await this.table!.query()
        .select(["id"])
        .where(`id = '${escapeSqlLiteral(id)}'`)
        .limit(1)
        .toArray();
      if (rows.length === 0) {
        return false;
      }
      await this.table!.delete(`id = '${escapeSqlLiteral(id)}'`);
      return true;
    });
  }
}

// ============================================================================
// Per-Agent Store Manager
// ============================================================================

export class StoreManager {
  private stores = new Map<string, MemoryStore>();
  private _knownAgentIds = new Set<string>();

  constructor(
    private readonly basePath: string,
    private readonly vectorDim: number,
  ) {
    // Discover existing agent DBs on disk so maintenance covers all agents,
    // not just those accessed since the last restart.
    try {
      if (existsSync(basePath)) {
        for (const entry of readdirSync(basePath, { withFileTypes: true })) {
          if (entry.isDirectory() && !entry.name.startsWith(".")) {
            this._knownAgentIds.add(entry.name);
          }
        }
      }
    } catch {
      // Non-fatal: maintenance will still work for lazily-accessed agents
    }
  }

  getStore(agentId: string): MemoryStore {
    const existing = this.stores.get(agentId);
    if (existing) {
      return existing;
    }

    const dbPath = join(this.basePath, agentId);
    validateStoragePath(dbPath);
    const store = new MemoryStore({ dbPath, vectorDim: this.vectorDim });
    this.stores.set(agentId, store);
    this._knownAgentIds.add(agentId);
    return store;
  }

  /** All known agent IDs (disk-discovered + lazily accessed). */
  get agentIds(): string[] {
    return [...this._knownAgentIds];
  }
}
