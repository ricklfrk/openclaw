/**
 * Access Tracker — debounced write-back tracker for memory access events.
 *
 * recordAccess() is synchronous (Map update). Pending deltas accumulate
 * until flush() is called. On flush, each pending entry's metadata is
 * updated with new accessCount and lastAccessedAt.
 *
 * Adapted from memory-lancedb-pro.
 */

import type { MemoryStore } from "./store.js";

export interface AccessMetadata {
  readonly accessCount: number;
  readonly lastAccessedAt: number;
}

export interface AccessTrackerOptions {
  readonly store: MemoryStore;
  readonly log: (msg: string) => void;
  readonly debounceMs?: number;
}

const MIN_ACCESS_COUNT = 0;
const MAX_ACCESS_COUNT = 10_000;

function clampAccessCount(value: number): number {
  if (!Number.isFinite(value)) {
    return MIN_ACCESS_COUNT;
  }
  return Math.min(MAX_ACCESS_COUNT, Math.max(MIN_ACCESS_COUNT, Math.floor(value)));
}

export function parseAccessMetadata(metadata: string | undefined): AccessMetadata {
  if (!metadata) {
    return { accessCount: 0, lastAccessedAt: 0 };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(metadata);
  } catch {
    return { accessCount: 0, lastAccessedAt: 0 };
  }
  if (typeof parsed !== "object" || parsed === null) {
    return { accessCount: 0, lastAccessedAt: 0 };
  }
  const obj = parsed as Record<string, unknown>;
  const rawCount =
    typeof obj.accessCount === "number" ? obj.accessCount : Number(obj.accessCount ?? 0);
  const rawLast =
    typeof obj.lastAccessedAt === "number" ? obj.lastAccessedAt : Number(obj.lastAccessedAt ?? 0);
  return {
    accessCount: clampAccessCount(rawCount),
    lastAccessedAt: Number.isFinite(rawLast) && rawLast >= 0 ? rawLast : 0,
  };
}

export function buildUpdatedAccessMetadata(
  existingMetadata: string | undefined,
  accessDelta: number,
): string {
  let existing: Record<string, unknown> = {};
  if (existingMetadata) {
    try {
      const parsed = JSON.parse(existingMetadata);
      if (typeof parsed === "object" && parsed !== null) {
        existing = { ...parsed };
      }
    } catch {}
  }
  const prev = parseAccessMetadata(existingMetadata);
  const newCount = clampAccessCount(prev.accessCount + accessDelta);
  const now = Date.now();
  return JSON.stringify({ ...existing, accessCount: newCount, lastAccessedAt: now });
}

export class AccessTracker {
  private readonly pending = new Map<string, number>();
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private flushPromise: Promise<void> | null = null;
  private readonly debounceMs: number;
  private readonly store: MemoryStore;
  private readonly log: (msg: string) => void;

  constructor(options: AccessTrackerOptions) {
    this.store = options.store;
    this.log = options.log;
    this.debounceMs = options.debounceMs ?? 5_000;
  }

  recordAccess(ids: readonly string[]): void {
    for (const id of ids) {
      this.pending.set(id, (this.pending.get(id) ?? 0) + 1);
    }
    this.resetTimer();
  }

  async flush(): Promise<void> {
    this.clearTimer();
    if (this.flushPromise) {
      await this.flushPromise;
      if (this.pending.size > 0) {
        return this.flush();
      }
      return;
    }
    if (this.pending.size === 0) {
      return;
    }
    this.flushPromise = this.doFlush();
    try {
      await this.flushPromise;
    } finally {
      this.flushPromise = null;
    }
    if (this.pending.size > 0) {
      this.resetTimer();
    }
  }

  destroy(): void {
    this.clearTimer();
    if (this.pending.size > 0) {
      this.log(
        `vector-memory: access-tracker: destroying with ${this.pending.size} pending writes`,
      );
    }
    this.pending.clear();
  }

  private async doFlush(): Promise<void> {
    const batch = new Map(this.pending);
    this.pending.clear();
    for (const [id, delta] of batch) {
      try {
        const current = await this.store.getById(id);
        if (!current) {
          continue;
        }
        const updatedMeta = buildUpdatedAccessMetadata(current.metadata, delta);
        await this.store.update(id, { metadata: updatedMeta });
      } catch (err) {
        this.pending.set(id, (this.pending.get(id) ?? 0) + delta);
        this.log(`vector-memory: access-tracker: write-back failed for ${id}: ${String(err)}`);
      }
    }
  }

  private resetTimer(): void {
    this.clearTimer();
    this.debounceTimer = setTimeout(() => {
      void this.flush();
    }, this.debounceMs);
  }

  private clearTimer(): void {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }
}
