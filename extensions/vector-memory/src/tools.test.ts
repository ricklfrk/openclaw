import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Embedder } from "./embedder.js";
import { DEFAULT_RETRIEVAL_CONFIG } from "./retriever.js";
import { MemoryStore, StoreManager } from "./store.js";
import { createForgetTool, createDetailTool, type ToolDeps } from "./tools.js";

const VECTOR_DIM = 8;

function makeVector(seed: number): number[] {
  return Array.from({ length: VECTOR_DIM }, (_, i) => Math.sin(seed + i));
}

const fakeEmbedder = {
  embed: async () => makeVector(0),
  embedQuery: async () => makeVector(0),
  dimensions: VECTOR_DIM,
} as unknown as Embedder;

describe("tools: id resolution and full UUID display", () => {
  let tmpDir: string;
  let storeManager: StoreManager;
  let store: MemoryStore;
  let deps: ToolDeps;
  const logs: string[] = [];
  const storedEntryIds: string[] = [];

  beforeAll(async () => {
    tmpDir = mkdtempSync(join(tmpdir(), "vm-tools-test-"));
    storeManager = new StoreManager(tmpDir, VECTOR_DIM);
    store = storeManager.getStore("test-agent");
    deps = {
      storeManager,
      embedder: fakeEmbedder,
      retrievalConfig: DEFAULT_RETRIEVAL_CONFIG,
      log: (msg: string) => logs.push(msg),
    };

    // Store a few entries
    for (let i = 0; i < 3; i++) {
      const entry = await store.store({
        text: `Memory number ${i}`,
        vector: makeVector(i),
        category: "fact",
        importance: 0.5,
        metadata: JSON.stringify({ l0_abstract: `Memory number ${i}` }),
      });
      storedEntryIds.push(entry.id);
    }
  });

  afterAll(() => {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  });

  describe("forget tool", () => {
    it("invalidates memory by full UUID", async () => {
      const targetId = storedEntryIds[0];
      const tool = createForgetTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-1", { memoryId: targetId });
      const text = (result.content[0] as { text: string }).text;
      expect(text).toContain("Deleted vector memory");
      // Output must contain the full UUID
      expect(text).toContain(targetId);
    });

    it("invalidates memory by 8-char prefix", async () => {
      const targetId = storedEntryIds[1];
      const prefix = targetId.slice(0, 8);
      const tool = createForgetTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-2", { memoryId: prefix });
      const text = (result.content[0] as { text: string }).text;
      // Should succeed (prefix resolves to exactly one entry)
      expect(text).toContain("Deleted vector memory");
      // Output should contain the full UUID, not just the prefix
      expect(text).toContain(targetId);
    });

    it("returns not_found for non-matching ID", async () => {
      const tool = createForgetTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-3", {
        memoryId: "00000000-0000-0000-0000-000000000000",
      });
      const text = (result.content[0] as { text: string }).text;
      expect(text).toContain("Memory not found");
    });
  });

  describe("detail tool", () => {
    it("finds memory by full UUID", async () => {
      const targetId = storedEntryIds[2];
      const tool = createDetailTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-4", { id: targetId });
      const text = (result.content[0] as { text: string }).text;
      expect(text).toContain("Memory number 2");
      expect(text).toContain(targetId);
    });

    it("finds memory by prefix", async () => {
      const targetId = storedEntryIds[2];
      const prefix = targetId.slice(0, 12);
      const tool = createDetailTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-5", { id: prefix });
      const text = (result.content[0] as { text: string }).text;
      expect(text).toContain("Memory number 2");
    });

    it("rejects too-short ID", async () => {
      const tool = createDetailTool(deps)({ agentId: "test-agent" });
      const result = await tool.execute("call-6", { id: "ab" });
      const text = (result.content[0] as { text: string }).text;
      expect(text).toContain("too short");
    });
  });
});
