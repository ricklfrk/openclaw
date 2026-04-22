import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { MemoryStore, type MemoryEntry } from "./store.js";

const VECTOR_DIM = 8;

function makeVector(seed: number): number[] {
  return Array.from({ length: VECTOR_DIM }, (_, i) => Math.sin(seed + i));
}

describe("MemoryStore", () => {
  let store: MemoryStore;
  let tmpDir: string;

  beforeAll(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "vm-store-test-"));
    store = new MemoryStore({ dbPath: tmpDir, vectorDim: VECTOR_DIM });
  });

  afterAll(() => {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  });

  // Keep a reference to stored IDs for subsequent tests
  const storedIds: string[] = [];

  it("stores entries and retrieves by ID", async () => {
    const entry = await store.store({
      text: "User likes TypeScript",
      vector: makeVector(1),
      category: "preference",
      importance: 0.8,
      metadata: JSON.stringify({ l0_abstract: "User likes TypeScript" }),
    });
    storedIds.push(entry.id);
    expect(entry.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-/);
    expect(entry.id.length).toBe(36);

    const fetched = await store.getById(entry.id);
    expect(fetched).not.toBeNull();
    expect(fetched!.text).toBe("User likes TypeScript");
  });

  describe("findByIdPrefix", () => {
    it("finds entry by full UUID", async () => {
      const results = await store.findByIdPrefix(storedIds[0], 10);
      expect(results.length).toBe(1);
      expect(results[0].id).toBe(storedIds[0]);
    });

    it("finds entry by 8-char prefix", async () => {
      const prefix = storedIds[0].slice(0, 8);
      const results = await store.findByIdPrefix(prefix, 10);
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].id.startsWith(prefix)).toBe(true);
    });

    it("returns empty for too-short prefix", async () => {
      const results = await store.findByIdPrefix("ab", 10);
      expect(results).toEqual([]);
    });

    it("returns empty for non-matching prefix", async () => {
      const results = await store.findByIdPrefix("zzzzzzzz", 10);
      expect(results).toEqual([]);
    });
  });

  describe("listAll with includeInvalidated", () => {
    let invalidatedId: string;

    it("setup: store and invalidate an entry", async () => {
      const entry = await store.store({
        text: "To be invalidated",
        vector: makeVector(99),
        category: "fact",
        importance: 0.5,
        metadata: "{}",
      });
      invalidatedId = entry.id;
      storedIds.push(invalidatedId);

      await store.update(invalidatedId, {
        metadata: JSON.stringify({ invalidated_at: Date.now(), invalidated_by: "test" }),
      });
    });

    it("excludes invalidated by default", async () => {
      const results = await store.listAll({ limit: 500 });
      const ids = results.map((r) => r.id);
      expect(ids).not.toContain(invalidatedId);
    });

    it("includes invalidated when requested", async () => {
      const results = await store.listAll({ limit: 500, includeInvalidated: true });
      const ids = results.map((r) => r.id);
      expect(ids).toContain(invalidatedId);
    });
  });

  describe("iterateAll", () => {
    it("yields all active entries in batches", async () => {
      const allEntries: MemoryEntry[] = [];
      for await (const batch of store.iterateAll({ batchSize: 50 })) {
        allEntries.push(...batch);
      }
      // Should not include invalidated entries
      const activeFromList = await store.listAll({ limit: 500 });
      expect(allEntries.length).toBe(activeFromList.length);
    });

    it("yields all entries including invalidated when requested", async () => {
      const allEntries: MemoryEntry[] = [];
      for await (const batch of store.iterateAll({ batchSize: 50, includeInvalidated: true })) {
        allEntries.push(...batch);
      }
      const allFromList = await store.listAll({ limit: 500, includeInvalidated: true });
      expect(allEntries.length).toBe(allFromList.length);
    });
  });
});
