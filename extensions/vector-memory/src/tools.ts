/**
 * Agent Tools — vector memory CRUD + health operations.
 * Each agent can only operate on its own per-agent LanceDB.
 *
 * Tools:
 *   vector_memory_recall        — hybrid search
 *   vector_memory_store         — save with dedup
 *   vector_memory_store_media   — save image/audio/video/PDF with multimodal embedding
 *   vector_memory_forget        — soft-delete
 *   vector_memory_stats         — DB statistics
 *   vector_memory_update        — update existing memory
 *   vector_memory_list          — browse/paginate memories
 *   vector_memory_compact       — batch dedup compression
 *   vector_memory_promote       — promote tier (peripheral→working→core)
 *   vector_memory_archive       — demote/archive memory
 *   vector_memory_explain_rank  — explain search scoring
 */

import { existsSync } from "node:fs";
import { extname } from "node:path";
import { Type } from "@sinclair/typebox";
import {
  MEMORY_CATEGORIES,
  mapToStoreCategory,
  getDefaultImportance,
  normalizeCategory,
} from "./categories.js";
import type { Embedder } from "./embedder.js";
import { convertFileForEmbedding } from "./media-convert.js";
import { isNoise } from "./noise-filter.js";
import { createRetriever, type RetrievalConfig } from "./retriever.js";
import type { StoreManager, MemoryEntry } from "./store.js";
import type { MemoryTier } from "./tier-manager.js";

function stringEnum<T extends readonly [string, ...string[]]>(values: T) {
  return Type.Unsafe<T[number]>({ type: "string", enum: [...values] });
}

function optionalStringEnum<T extends readonly [string, ...string[]]>(values: T) {
  return Type.Optional(Type.Unsafe<T[number]>({ type: "string", enum: [...values] }));
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.floor(value)));
}

function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(1, maxChars - 1)).trimEnd()}…`;
}

function resolveAgentId(ctx: { agentId?: string }): string {
  return ctx.agentId?.trim() || "main";
}

function parseMeta(entry: MemoryEntry): Record<string, unknown> {
  try {
    return JSON.parse(entry.metadata || "{}");
  } catch {
    return {};
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// ============================================================================
// Types
// ============================================================================

export interface ToolDeps {
  storeManager: StoreManager;
  embedder: Embedder;
  retrievalConfig: RetrievalConfig;
  log: (msg: string) => void;
}

// ============================================================================
// Tool: vector_memory_recall
// ============================================================================

export function createRecallTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_recall",
    label: "Vector Memory Recall",
    description:
      "Search through supplementary vector long-term memories using hybrid retrieval (vector + keyword search). " +
      "Use when you need context about user preferences, past decisions, or previously discussed topics.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query for finding relevant memories" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default: 5, max: 20)" })),
      category: Type.Optional(stringEnum([...MEMORY_CATEGORIES])),
      entity_tags: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Entity tags to boost matches — person names, project names, technologies (lowercase)",
        }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const {
        query,
        limit = 5,
        category,
        entity_tags: rawEntityTags,
      } = params as {
        query: string;
        limit?: number;
        category?: string;
        entity_tags?: string[];
      };

      const agentId = resolveAgentId(toolCtx);

      try {
        if (!query || query.trim().length < 2) {
          return {
            content: [{ type: "text", text: "Query too short. Provide at least 2 characters." }],
            details: { error: "query_too_short" },
          };
        }

        const store = deps.storeManager.getStore(agentId);
        const retriever = createRetriever(store, deps.embedder, deps.retrievalConfig);
        const safeLimit = clampInt(limit, 1, 20);

        const entityTags = Array.isArray(rawEntityTags)
          ? rawEntityTags
              .filter((t): t is string => typeof t === "string" && t.trim().length > 0)
              .map((t) => t.trim().toLowerCase())
              .slice(0, 10)
          : undefined;
        const results = await retriever.retrieve({
          query: query.trim(),
          limit: safeLimit,
          category: category
            ? normalizeCategory(category)
              ? mapToStoreCategory(normalizeCategory(category)!)
              : undefined
            : undefined,
          entityTags,
        });

        if (results.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `No matching vector memories found for: "${truncateText(query, 60)}"`,
              },
            ],
            details: { action: "recall", matches: 0, agentId },
          };
        }

        const formatted = results.map((r, i) => {
          let displayCategory: string = r.entry.category;
          let title = r.entry.text;
          let detail = "";
          let tagsSuffix = "";
          try {
            const meta = JSON.parse(r.entry.metadata || "{}");
            displayCategory = (meta.memory_category as string) || r.entry.category;
            title = (meta.l0_abstract as string) || r.entry.text;
            const l1 = (meta.l1_overview as string) || "";
            if (
              l1.length > 0 &&
              l1 !== title &&
              !l1.startsWith(title) &&
              !title.startsWith(l1.slice(0, Math.min(l1.length, 180)))
            ) {
              detail = ` — ${truncateText(l1, 150)}`;
            }
            if (Array.isArray(meta.entity_tags) && meta.entity_tags.length > 0) {
              tagsSuffix = ` tags:[${(meta.entity_tags as string[]).join(",")}]`;
            }
          } catch {}
          const dateStr = new Date(r.entry.timestamp).toISOString().split("T")[0];
          return `${i + 1}. [${dateStr}][${displayCategory}] ${truncateText(title, 200)}${detail} (score: ${r.score.toFixed(3)}, id: ${r.entry.id.slice(0, 8)}${tagsSuffix})`;
        });

        return {
          content: [
            {
              type: "text",
              text: `Found ${results.length} vector memories:\n${formatted.join("\n")}`,
            },
          ],
          details: {
            action: "recall",
            matches: results.length,
            agentId,
            results: results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              score: r.score,
            })),
          },
        };
      } catch (err) {
        deps.log(`vector-memory: recall tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Vector memory recall failed: ${String(err)}` }],
          details: { error: "recall_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_store
// ============================================================================

export function createStoreTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_store",
    label: "Vector Memory Store",
    description:
      "Save important information to supplementary vector long-term memory. " +
      "Use for preferences, facts, decisions, entities, and other notable information worth remembering across sessions.",
    parameters: Type.Object({
      text: Type.String({ description: "Information to remember" }),
      category: Type.Optional(stringEnum([...MEMORY_CATEGORIES])),
      importance: Type.Optional(
        Type.Number({ description: "Importance score 0-1 (default: auto by category)" }),
      ),
      entity_tags: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Entity tags for precise retrieval — proper nouns like person names, project names, technologies (max 5, lowercase)",
        }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const {
        text,
        category: rawCategory,
        importance,
        entity_tags: rawEntityTags,
      } = params as {
        text: string;
        category?: string;
        importance?: number;
        entity_tags?: string[];
      };

      const agentId = resolveAgentId(toolCtx);

      try {
        if (!text || text.trim().length < 3) {
          return {
            content: [{ type: "text", text: "Text too short. Provide at least 3 characters." }],
            details: { error: "text_too_short" },
          };
        }

        if (isNoise(text)) {
          return {
            content: [
              { type: "text", text: "Content appears to be noise/boilerplate and was not stored." },
            ],
            details: { error: "noise_filtered" },
          };
        }

        const normalizedCategory = normalizeCategory(rawCategory ?? "other");
        const storeCategory = normalizedCategory ? mapToStoreCategory(normalizedCategory) : "other";
        const finalImportance =
          importance != null && Number.isFinite(importance)
            ? Math.min(1, Math.max(0, importance))
            : normalizedCategory
              ? getDefaultImportance(normalizedCategory)
              : 0.5;

        const vector = await deps.embedder.embed(text.trim());
        const store = deps.storeManager.getStore(agentId);

        // Dedup + conflict detection against existing memories
        let conflictWarning = "";
        try {
          const existing = await store.vectorSearch(vector, 3, 0.1);
          if (existing.length > 0 && existing[0].score > 0.98) {
            return {
              content: [
                {
                  type: "text",
                  text: `Similar memory already exists: "${truncateText(existing[0].entry.text, 80)}" (id: ${existing[0].entry.id.slice(0, 8)}, similarity: ${existing[0].score.toFixed(3)}). Use vector_memory_update to modify it instead.`,
                },
              ],
              details: {
                action: "duplicate",
                existingId: existing[0].entry.id,
                similarity: existing[0].score,
                agentId,
              },
            };
          }
          // Warn about potentially conflicting neighbors (high similarity but not identical).
          // The memory will still be stored, but the agent gets a hint to review.
          const conflicts = existing.filter((r) => r.score >= 0.75 && r.score <= 0.98);
          if (conflicts.length > 0) {
            const items = conflicts
              .slice(0, 2)
              .map(
                (r) =>
                  `"${truncateText(r.entry.text, 80)}" (id: ${r.entry.id.slice(0, 8)}, similarity: ${r.score.toFixed(3)})`,
              )
              .join("; ");
            conflictWarning =
              `\n⚠ Similar memories exist that may conflict or overlap: ${items}. ` +
              `Consider using vector_memory_update or vector_memory_forget if this supersedes them.`;
          }
        } catch (err) {
          deps.log(`vector-memory: dedup pre-check failed, continue store: ${String(err)}`);
        }

        const categoryLabel = normalizedCategory ?? "other";
        const entityTags = Array.isArray(rawEntityTags)
          ? rawEntityTags
              .filter((t): t is string => typeof t === "string" && t.trim().length > 0)
              .map((t) => t.trim().toLowerCase())
              .slice(0, 5)
          : undefined;
        const metaObj: Record<string, unknown> = {
          l0_abstract: text.trim().slice(0, 200),
          l1_overview: "",
          l2_content: text.trim(),
          memory_category: categoryLabel,
          source: "agent_tool",
        };
        if (entityTags && entityTags.length > 0) {
          metaObj.entity_tags = entityTags;
        }
        const metadata = JSON.stringify(metaObj);

        const entry = await store.store({
          text: text.trim().slice(0, 200),
          vector,
          category: storeCategory,
          importance: finalImportance,
          metadata,
        });

        deps.log(
          `vector-memory: store tool [${agentId}][${categoryLabel}] id=${entry.id.slice(0, 8)}`,
        );

        return {
          content: [
            {
              type: "text",
              text: `Stored vector memory: [${categoryLabel}] "${truncateText(text, 60)}" (id: ${entry.id.slice(0, 8)})${conflictWarning}`,
            },
          ],
          details: {
            action: "store",
            id: entry.id,
            category: categoryLabel,
            importance: finalImportance,
            agentId,
            ...(conflictWarning ? { hasConflictWarning: true } : {}),
          },
        };
      } catch (err) {
        deps.log(`vector-memory: store tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Vector memory store failed: ${String(err)}` }],
          details: { error: "store_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_forget
// ============================================================================

export function createForgetTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_forget",
    label: "Vector Memory Forget",
    description:
      "Delete specific vector memories by search query or direct ID. " +
      "Only affects this agent's own memory store.",
    parameters: Type.Object({
      query: Type.Optional(Type.String({ description: "Search query to find memory to delete" })),
      memoryId: Type.Optional(
        Type.String({ description: "Specific memory ID (full UUID or 8+ char prefix)" }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { query, memoryId } = params as { query?: string; memoryId?: string };
      const agentId = resolveAgentId(toolCtx);

      try {
        if (!query && !memoryId) {
          return {
            content: [{ type: "text", text: "Provide either a search query or memoryId." }],
            details: { error: "missing_params" },
          };
        }

        const store = deps.storeManager.getStore(agentId);

        // Direct ID deletion
        if (memoryId) {
          const existing = await store.getById(memoryId);
          if (!existing) {
            return {
              content: [{ type: "text", text: `Memory not found: ${memoryId}` }],
              details: { error: "not_found", agentId },
            };
          }

          // Mark as invalidated instead of hard delete (preserves history)
          let meta: Record<string, unknown> = {};
          try {
            meta = JSON.parse(existing.metadata || "{}");
          } catch {}
          meta.invalidated_at = Date.now();
          meta.invalidated_by = "agent_tool";
          await store.update(memoryId, { metadata: JSON.stringify(meta) });

          deps.log(`vector-memory: forget tool [${agentId}] invalidated ${memoryId.slice(0, 8)}`);
          return {
            content: [
              {
                type: "text",
                text: `Deleted vector memory: ${existing.text.slice(0, 60)} (id: ${memoryId.slice(0, 8)})`,
              },
            ],
            details: { action: "forget", id: memoryId, agentId },
          };
        }

        // Search-based deletion
        const queryVector = await deps.embedder.embedQuery(query!);
        const results = await store.vectorSearch(queryVector, 1, 0.5);

        if (results.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `No matching vector memory found for: "${truncateText(query!, 60)}"`,
              },
            ],
            details: { error: "no_match", agentId },
          };
        }

        const target = results[0];
        let meta: Record<string, unknown> = {};
        try {
          meta = JSON.parse(target.entry.metadata || "{}");
        } catch {}
        meta.invalidated_at = Date.now();
        meta.invalidated_by = "agent_tool";
        await store.update(target.entry.id, { metadata: JSON.stringify(meta) });

        deps.log(
          `vector-memory: forget tool [${agentId}] invalidated ${target.entry.id.slice(0, 8)} via query`,
        );
        return {
          content: [
            {
              type: "text",
              text: `Deleted vector memory: ${target.entry.text.slice(0, 60)} (id: ${target.entry.id.slice(0, 8)}, score: ${target.score.toFixed(3)})`,
            },
          ],
          details: { action: "forget", id: target.entry.id, score: target.score, agentId },
        };
      } catch (err) {
        deps.log(`vector-memory: forget tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Vector memory forget failed: ${String(err)}` }],
          details: { error: "forget_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_stats
// ============================================================================

export function createStatsTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_stats",
    label: "Vector Memory Stats",
    description: "Get statistics about this agent's vector memory store.",
    parameters: Type.Object({}),
    async execute() {
      const agentId = resolveAgentId(toolCtx);

      try {
        const store = deps.storeManager.getStore(agentId);
        const stats = await store.stats();

        const lines = [
          `Vector Memory Stats for agent: ${agentId}`,
          `Total entries: ${stats.totalCount}`,
          `DB path: ${store.dbPath}`,
          `FTS (BM25) support: ${store.hasFtsSupport ? "yes" : "no"}`,
          "",
          "By category:",
          ...Object.entries(stats.categoryCounts).map(([cat, count]) => `  ${cat}: ${count}`),
        ];

        return {
          content: [{ type: "text", text: lines.join("\n") }],
          details: { action: "stats", agentId, ...stats },
        };
      } catch (err) {
        deps.log(`vector-memory: stats tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Vector memory stats failed: ${String(err)}` }],
          details: { error: "stats_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_update
// ============================================================================

export function createUpdateTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_update",
    label: "Vector Memory Update",
    description:
      "Update an existing vector memory entry. Can change text, category, or importance. " +
      "The old version is preserved as superseded history.",
    parameters: Type.Object({
      memoryId: Type.String({ description: "Memory ID to update (full UUID or 8+ char prefix)" }),
      text: Type.Optional(Type.String({ description: "New text content" })),
      category: Type.Optional(stringEnum([...MEMORY_CATEGORIES])),
      importance: Type.Optional(Type.Number({ description: "New importance score 0-1" })),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const {
        memoryId,
        text,
        category: rawCategory,
        importance,
      } = params as {
        memoryId: string;
        text?: string;
        category?: string;
        importance?: number;
      };
      const agentId = resolveAgentId(toolCtx);

      try {
        if (!memoryId) {
          return {
            content: [{ type: "text", text: "memoryId is required." }],
            details: { error: "missing_id" },
          };
        }

        const store = deps.storeManager.getStore(agentId);
        const existing = await store.getById(memoryId);
        if (!existing) {
          return {
            content: [{ type: "text", text: `Memory not found: ${memoryId}` }],
            details: { error: "not_found", agentId },
          };
        }

        const updates: Record<string, unknown> = {};
        const meta = parseMeta(existing);

        if (text && text.trim().length >= 3) {
          const newVector = await deps.embedder.embed(text.trim());
          updates.text = text.trim().slice(0, 200);
          updates.vector = newVector;
          meta.l0_abstract = text.trim().slice(0, 200);
          meta.l2_content = text.trim();
          meta.updated_at = Date.now();
        }

        if (rawCategory) {
          const cat = normalizeCategory(rawCategory);
          if (cat) {
            updates.category = mapToStoreCategory(cat);
            meta.memory_category = cat;
          }
        }

        if (importance != null && Number.isFinite(importance)) {
          updates.importance = Math.min(1, Math.max(0, importance));
        }

        // Track supersede history
        if (!meta.history) {
          meta.history = [];
        }
        (meta.history as unknown[]).push({
          text: existing.text,
          category: existing.category,
          importance: existing.importance,
          updated_at: Date.now(),
        });

        updates.metadata = JSON.stringify(meta);
        await store.update(memoryId, updates as Parameters<typeof store.update>[1]);

        deps.log(`vector-memory: update tool [${agentId}] updated ${memoryId.slice(0, 8)}`);
        return {
          content: [
            {
              type: "text",
              text: `Updated memory ${memoryId.slice(0, 8)}: ${Object.keys(updates)
                .filter((k) => k !== "metadata")
                .join(", ")}`,
            },
          ],
          details: { action: "update", id: memoryId, agentId, fields: Object.keys(updates) },
        };
      } catch (err) {
        deps.log(`vector-memory: update tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Update failed: ${String(err)}` }],
          details: { error: "update_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_list
// ============================================================================

/**
 * Parse a YYYY-MM-DD string into epoch ms for the start of that day (UTC).
 * Returns undefined for invalid input.
 */
function parseDateToMs(dateStr: string | undefined): number | undefined {
  if (!dateStr) {
    return undefined;
  }
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr.trim());
  if (!m) {
    return undefined;
  }
  const ts = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (Number.isNaN(ts)) {
    return undefined;
  }
  return ts;
}

const MS_PER_DAY = 86_400_000;

export function createListTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_list",
    label: "Vector Memory List",
    description:
      "List and browse vector memories with optional category filter, date filter, sorting, and pagination. " +
      "Date filter: use 'date' for a single day, OR 'date_start'/'date_end' for a range. Do NOT combine 'date' with 'date_start'/'date_end'.",
    parameters: Type.Object({
      category: Type.Optional(stringEnum([...MEMORY_CATEGORIES])),
      date: Type.Optional(Type.String({ description: "Single day filter (YYYY-MM-DD)" })),
      date_start: Type.Optional(Type.String({ description: "Range start inclusive (YYYY-MM-DD)" })),
      date_end: Type.Optional(Type.String({ description: "Range end inclusive (YYYY-MM-DD)" })),
      sort: Type.Optional(stringEnum(["newest", "oldest"])),
      offset: Type.Optional(Type.Number({ description: "Skip first N entries (default: 0)" })),
      limit: Type.Optional(Type.Number({ description: "Max entries (default: 10, max: 50)" })),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const {
        category: rawCategory,
        date,
        date_start: dateStart,
        date_end: dateEnd,
        sort = "newest",
        offset = 0,
        limit = 10,
      } = params as {
        category?: string;
        date?: string;
        date_start?: string;
        date_end?: string;
        sort?: "newest" | "oldest";
        offset?: number;
        limit?: number;
      };
      const agentId = resolveAgentId(toolCtx);

      // ── Date parameter validation ──────────────────────────────
      if (date && (dateStart || dateEnd)) {
        return {
          content: [
            {
              type: "text",
              text: "Invalid date filter: use either 'date' for a single day, or 'date_start'/'date_end' for a range — not both.",
            },
          ],
          details: { error: "invalid_date_params" },
        };
      }

      let timestampFrom: number | undefined;
      let timestampTo: number | undefined;
      let dateLabel = "";

      if (date) {
        const dayMs = parseDateToMs(date);
        if (dayMs == null) {
          return {
            content: [{ type: "text", text: `Invalid date format: "${date}". Use YYYY-MM-DD.` }],
            details: { error: "invalid_date_format" },
          };
        }
        timestampFrom = dayMs;
        timestampTo = dayMs + MS_PER_DAY - 1;
        dateLabel = date.trim();
      } else if (dateStart || dateEnd) {
        if (dateStart) {
          const ms = parseDateToMs(dateStart);
          if (ms == null) {
            return {
              content: [
                {
                  type: "text",
                  text: `Invalid date_start format: "${dateStart}". Use YYYY-MM-DD.`,
                },
              ],
              details: { error: "invalid_date_format" },
            };
          }
          timestampFrom = ms;
        }
        if (dateEnd) {
          const ms = parseDateToMs(dateEnd);
          if (ms == null) {
            return {
              content: [
                { type: "text", text: `Invalid date_end format: "${dateEnd}". Use YYYY-MM-DD.` },
              ],
              details: { error: "invalid_date_format" },
            };
          }
          timestampTo = ms + MS_PER_DAY - 1;
        }
        // date_start == date_end → same as single day
        const startStr = dateStart?.trim() ?? "earliest";
        const endStr = dateEnd?.trim() ?? "latest";
        dateLabel = startStr === endStr ? startStr : `${startStr} → ${endStr}`;
      }

      try {
        const store = deps.storeManager.getStore(agentId);
        const normalizedListCategory = rawCategory ? normalizeCategory(rawCategory) : null;
        const storeCategory = rawCategory
          ? normalizedListCategory
            ? mapToStoreCategory(normalizedListCategory)
            : "other"
          : undefined;
        const safeLimit = clampInt(limit, 1, 50);
        const safeOffset = Math.max(0, Math.floor(offset));
        const safeSort = sort === "oldest" ? ("oldest" as const) : ("newest" as const);

        const entries = await store.listAll({
          offset: safeOffset,
          limit: safeLimit,
          category: storeCategory,
          sort: safeSort,
          timestampFrom,
          timestampTo,
        });

        if (entries.length === 0) {
          const filters = [
            rawCategory ? `category: ${rawCategory}` : "",
            dateLabel ? `date: ${dateLabel}` : "",
          ]
            .filter(Boolean)
            .join(", ");
          return {
            content: [
              {
                type: "text",
                text: `No memories found${filters ? ` (${filters})` : ""} (offset: ${safeOffset})`,
              },
            ],
            details: { action: "list", count: 0, agentId },
          };
        }

        const lines = entries.map((entry, i) => {
          const meta = parseMeta(entry);
          const cat = (meta.memory_category as string) || entry.category;
          const abstract = (meta.l0_abstract as string) || entry.text;
          const tier = (meta.tier as string) || "working";
          const entryDate = new Date(entry.timestamp).toISOString().split("T")[0];
          const accessCount = typeof meta.accessCount === "number" ? meta.accessCount : 0;
          return `${safeOffset + i + 1}. [${entryDate}][${cat}][${tier}] ${truncateText(abstract, 120)} (id: ${entry.id.slice(0, 8)}, imp: ${entry.importance.toFixed(2)}, access: ${accessCount})`;
        });

        const sortLabel = safeSort === "newest" ? "newest first" : "oldest first";
        const rangeNote = dateLabel ? `, ${dateLabel}` : "";
        return {
          content: [
            {
              type: "text",
              text: `Memories (${safeOffset + 1}-${safeOffset + entries.length}, ${sortLabel}${rangeNote}):\n${lines.join("\n")}`,
            },
          ],
          details: {
            action: "list",
            count: entries.length,
            offset: safeOffset,
            sort: safeSort,
            agentId,
          },
        };
      } catch (err) {
        deps.log(`vector-memory: list tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `List failed: ${String(err)}` }],
          details: { error: "list_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_compact
// ============================================================================

export function createCompactTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_compact",
    label: "Vector Memory Compact",
    description:
      "Batch deduplication and compression. Finds near-duplicate memories (cosine > threshold) " +
      "and removes redundant entries. Also purges invalidated/soft-deleted entries.",
    parameters: Type.Object({
      threshold: Type.Optional(
        Type.Number({
          description: "Similarity threshold for dedup (default: 0.95, range 0.85-0.99)",
        }),
      ),
      dryRun: Type.Optional(
        Type.Boolean({
          description: "If true, report duplicates without deleting (default: false)",
        }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { threshold = 0.95, dryRun = false } = params as {
        threshold?: number;
        dryRun?: boolean;
      };
      const agentId = resolveAgentId(toolCtx);

      try {
        const store = deps.storeManager.getStore(agentId);
        const safeThreshold = Math.min(0.99, Math.max(0.85, threshold));

        // Load all entries
        const allEntries = await store.listAll({ limit: 100 });
        if (allEntries.length === 0) {
          return {
            content: [{ type: "text", text: "No memories to compact." }],
            details: { action: "compact", removed: 0, agentId },
          };
        }

        // Phase 1: purge invalidated entries
        const invalidated: string[] = [];
        const active: MemoryEntry[] = [];
        for (const entry of allEntries) {
          const meta = parseMeta(entry);
          if (meta.invalidated_at) {
            invalidated.push(entry.id);
          } else {
            active.push(entry);
          }
        }

        // Phase 2: find near-duplicates among active entries
        const duplicatePairs: Array<{ keepId: string; removeId: string; similarity: number }> = [];
        const toRemove = new Set<string>();

        for (let i = 0; i < active.length; i++) {
          if (toRemove.has(active[i].id)) {
            continue;
          }
          for (let j = i + 1; j < active.length; j++) {
            if (toRemove.has(active[j].id)) {
              continue;
            }
            const sim = cosineSimilarity(active[i].vector, active[j].vector);
            if (sim >= safeThreshold) {
              // Keep the one with higher importance or more recent timestamp
              const keep =
                active[i].importance >= active[j].importance ||
                active[i].timestamp >= active[j].timestamp
                  ? active[i]
                  : active[j];
              const remove = keep === active[i] ? active[j] : active[i];
              duplicatePairs.push({ keepId: keep.id, removeId: remove.id, similarity: sim });
              toRemove.add(remove.id);
            }
          }
        }

        if (dryRun) {
          const lines = [
            `[DRY RUN] Compact analysis for agent: ${agentId}`,
            `Total entries: ${allEntries.length}`,
            `Invalidated (would purge): ${invalidated.length}`,
            `Duplicates (would remove): ${duplicatePairs.length}`,
          ];
          for (const pair of duplicatePairs.slice(0, 10)) {
            lines.push(
              `  remove ${pair.removeId.slice(0, 8)} (keep ${pair.keepId.slice(0, 8)}, sim: ${pair.similarity.toFixed(3)})`,
            );
          }
          return {
            content: [{ type: "text", text: lines.join("\n") }],
            details: {
              action: "compact_dry",
              invalidated: invalidated.length,
              duplicates: duplicatePairs.length,
              agentId,
            },
          };
        }

        // Execute purges
        let purged = 0;
        for (const id of invalidated) {
          if (await store.deleteById(id)) {
            purged++;
          }
        }
        let deduped = 0;
        for (const id of toRemove) {
          if (await store.deleteById(id)) {
            deduped++;
          }
        }

        deps.log(`vector-memory: compact [${agentId}] purged=${purged} deduped=${deduped}`);
        return {
          content: [
            {
              type: "text",
              text: `Compact complete: purged ${purged} invalidated, removed ${deduped} duplicates (threshold: ${safeThreshold})`,
            },
          ],
          details: { action: "compact", purged, deduped, threshold: safeThreshold, agentId },
        };
      } catch (err) {
        deps.log(`vector-memory: compact tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Compact failed: ${String(err)}` }],
          details: { error: "compact_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_promote
// ============================================================================

const TIER_ORDER: MemoryTier[] = ["peripheral", "working", "core"];

function normalizeTier(raw: string | undefined): MemoryTier {
  const tier = raw || "working";
  return TIER_ORDER.includes(tier as MemoryTier) ? (tier as MemoryTier) : "working";
}

export function createPromoteTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_promote",
    label: "Vector Memory Promote",
    description:
      "Promote a memory to a higher tier (peripheral → working → core). " +
      "Higher-tier memories decay slower and are prioritized in recall.",
    parameters: Type.Object({
      memoryId: Type.String({ description: "Memory ID to promote" }),
      targetTier: optionalStringEnum(["working", "core"] as const),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { memoryId, targetTier } = params as { memoryId: string; targetTier?: MemoryTier };
      const agentId = resolveAgentId(toolCtx);

      try {
        const store = deps.storeManager.getStore(agentId);
        const entry = await store.getById(memoryId);
        if (!entry) {
          return {
            content: [{ type: "text", text: `Memory not found: ${memoryId}` }],
            details: { error: "not_found", agentId },
          };
        }

        const meta = parseMeta(entry);
        const currentTier = normalizeTier(meta.tier as string);
        const currentIdx = TIER_ORDER.indexOf(currentTier);

        let newTier: MemoryTier;
        if (targetTier) {
          const targetIdx = TIER_ORDER.indexOf(targetTier);
          if (targetIdx <= currentIdx) {
            return {
              content: [
                {
                  type: "text",
                  text: `Memory is already at tier "${currentTier}" (target "${targetTier}" is not higher).`,
                },
              ],
              details: { error: "already_at_tier", agentId },
            };
          }
          newTier = targetTier;
        } else {
          if (currentIdx >= TIER_ORDER.length - 1) {
            return {
              content: [
                { type: "text", text: `Memory is already at highest tier: "${currentTier}"` },
              ],
              details: { error: "already_max", agentId },
            };
          }
          newTier = TIER_ORDER[currentIdx + 1];
        }

        meta.tier = newTier;
        meta.promoted_at = Date.now();
        await store.update(memoryId, { metadata: JSON.stringify(meta) });

        deps.log(
          `vector-memory: promote [${agentId}] ${memoryId.slice(0, 8)}: ${currentTier} → ${newTier}`,
        );
        return {
          content: [
            {
              type: "text",
              text: `Promoted memory ${memoryId.slice(0, 8)}: ${currentTier} → ${newTier}`,
            },
          ],
          details: { action: "promote", id: memoryId, from: currentTier, to: newTier, agentId },
        };
      } catch (err) {
        deps.log(`vector-memory: promote tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Promote failed: ${String(err)}` }],
          details: { error: "promote_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_archive
// ============================================================================

export function createArchiveTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_archive",
    label: "Vector Memory Archive",
    description:
      "Demote/archive a memory to a lower tier (core → working → peripheral). " +
      "Peripheral memories decay fastest and are deprioritized in recall.",
    parameters: Type.Object({
      memoryId: Type.String({ description: "Memory ID to archive/demote" }),
      targetTier: optionalStringEnum(["working", "peripheral"] as const),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { memoryId, targetTier } = params as { memoryId: string; targetTier?: MemoryTier };
      const agentId = resolveAgentId(toolCtx);

      try {
        const store = deps.storeManager.getStore(agentId);
        const entry = await store.getById(memoryId);
        if (!entry) {
          return {
            content: [{ type: "text", text: `Memory not found: ${memoryId}` }],
            details: { error: "not_found", agentId },
          };
        }

        const meta = parseMeta(entry);
        const currentTier = normalizeTier(meta.tier as string);
        const currentIdx = TIER_ORDER.indexOf(currentTier);

        let newTier: MemoryTier;
        if (targetTier) {
          const targetIdx = TIER_ORDER.indexOf(targetTier);
          if (targetIdx >= currentIdx) {
            return {
              content: [
                {
                  type: "text",
                  text: `Memory is already at tier "${currentTier}" (target "${targetTier}" is not lower).`,
                },
              ],
              details: { error: "already_at_tier", agentId },
            };
          }
          newTier = targetTier;
        } else {
          if (currentIdx <= 0) {
            return {
              content: [
                { type: "text", text: `Memory is already at lowest tier: "${currentTier}"` },
              ],
              details: { error: "already_min", agentId },
            };
          }
          newTier = TIER_ORDER[currentIdx - 1];
        }

        meta.tier = newTier;
        meta.archived_at = Date.now();
        await store.update(memoryId, { metadata: JSON.stringify(meta) });

        deps.log(
          `vector-memory: archive [${agentId}] ${memoryId.slice(0, 8)}: ${currentTier} → ${newTier}`,
        );
        return {
          content: [
            {
              type: "text",
              text: `Archived memory ${memoryId.slice(0, 8)}: ${currentTier} → ${newTier}`,
            },
          ],
          details: { action: "archive", id: memoryId, from: currentTier, to: newTier, agentId },
        };
      } catch (err) {
        deps.log(`vector-memory: archive tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Archive failed: ${String(err)}` }],
          details: { error: "archive_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_explain_rank
// ============================================================================

export function createExplainRankTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_explain_rank",
    label: "Vector Memory Explain Rank",
    description:
      "Explain why a search query returns specific results and their ranking. " +
      "Shows vector score, BM25 score, rerank score, recency, importance, and tier for each result.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query to explain" }),
      limit: Type.Optional(Type.Number({ description: "Max results, 1-10 (default: 5)" })),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { query, limit = 5 } = params as { query: string; limit?: number };
      const agentId = resolveAgentId(toolCtx);

      try {
        if (!query || query.trim().length < 2) {
          return {
            content: [{ type: "text", text: "Query too short." }],
            details: { error: "query_too_short" },
          };
        }

        const store = deps.storeManager.getStore(agentId);
        const retriever = createRetriever(store, deps.embedder, deps.retrievalConfig);
        const safeLimit = clampInt(limit, 1, 10);

        const results = await retriever.retrieve({ query: query.trim(), limit: safeLimit });

        if (results.length === 0) {
          return {
            content: [{ type: "text", text: `No results for: "${truncateText(query, 60)}"` }],
            details: { action: "explain_rank", matches: 0, agentId },
          };
        }

        const now = Date.now();
        const lines = results.map((r, i) => {
          const meta = parseMeta(r.entry);
          const cat = (meta.memory_category as string) || r.entry.category;
          const tier = (meta.tier as string) || "working";
          const abstract = (meta.l0_abstract as string) || r.entry.text;
          const ageDays = Math.floor((now - r.entry.timestamp) / 86_400_000);
          const accessCount = typeof meta.accessCount === "number" ? meta.accessCount : 0;

          const sourceLines: string[] = [];
          if (r.sources?.vector) {
            sourceLines.push(
              `vector=${r.sources.vector.score.toFixed(3)} rank=${r.sources.vector.rank}`,
            );
          }
          if (r.sources?.bm25) {
            sourceLines.push(`bm25=${r.sources.bm25.score.toFixed(3)} rank=${r.sources.bm25.rank}`);
          }
          if (r.sources?.fused) {
            sourceLines.push(`fused=${r.sources.fused.score.toFixed(3)}`);
          }
          if (r.sources?.reranked) {
            sourceLines.push(`reranked=${r.sources.reranked.score.toFixed(3)}`);
          }

          const tagsLine =
            Array.isArray(meta.entity_tags) && meta.entity_tags.length > 0
              ? ` tags=[${(meta.entity_tags as string[]).join(",")}]`
              : "";
          return [
            `${i + 1}. [${cat}][${tier}] ${truncateText(abstract, 100)}`,
            `   final=${r.score.toFixed(3)} | ${sourceLines.join(" | ")}`,
            `   imp=${r.entry.importance.toFixed(2)} age=${ageDays}d access=${accessCount} id=${r.entry.id.slice(0, 8)}${tagsLine}`,
          ].join("\n");
        });

        return {
          content: [
            {
              type: "text",
              text: `Rank explanation for: "${truncateText(query, 60)}"\n\n${lines.join("\n\n")}`,
            },
          ],
          details: { action: "explain_rank", matches: results.length, agentId },
        };
      } catch (err) {
        deps.log(`vector-memory: explain_rank error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Explain rank failed: ${String(err)}` }],
          details: { error: "explain_rank_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_store_media
// ============================================================================

// Natively supported + auto-convertible formats
const EXT_TO_MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".pdf": "application/pdf",
  // Auto-converted to supported formats before embedding
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".webm": "video/webm",
};

function guessMimeType(filePath: string): string | undefined {
  const ext = extname(filePath).toLowerCase();
  return EXT_TO_MIME[ext];
}

export function createStoreMediaTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_store_media",
    label: "Vector Memory Store Media",
    description:
      "Store an image, audio, video, or PDF as a multimodal vector memory. " +
      "Requires gemini-embedding-2-preview model. WebP/GIF images and WebM video are auto-converted to supported formats. " +
      "The media is embedded into the same vector space as text, so it can be recalled by text queries later.",
    parameters: Type.Object({
      filePath: Type.String({
        description: "Absolute path to the media file (image/audio/video/PDF)",
      }),
      description: Type.String({
        description:
          "Human-readable description of the media content. Used for display in recall and BM25 keyword search.",
      }),
      mimeType: Type.Optional(
        Type.String({ description: "MIME type (auto-detected from extension if omitted)" }),
      ),
      category: Type.Optional(stringEnum([...MEMORY_CATEGORIES])),
      importance: Type.Optional(
        Type.Number({ description: "Importance score 0-1 (default: 0.7)" }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const {
        filePath,
        description,
        mimeType: rawMimeType,
        category: rawCategory,
        importance,
      } = params as {
        filePath: string;
        description: string;
        mimeType?: string;
        category?: string;
        importance?: number;
      };

      const agentId = resolveAgentId(toolCtx);

      try {
        if (!filePath || !description || description.trim().length < 3) {
          return {
            content: [
              {
                type: "text",
                text: "Both filePath and description (3+ chars) are required.",
              },
            ],
            details: { error: "missing_params" },
          };
        }

        if (!existsSync(filePath)) {
          return {
            content: [{ type: "text", text: `File not found: ${filePath}` }],
            details: { error: "file_not_found" },
          };
        }

        const mimeType = rawMimeType || guessMimeType(filePath);
        if (!mimeType) {
          return {
            content: [
              {
                type: "text",
                text: `Cannot determine MIME type for: ${filePath}. Provide mimeType parameter.`,
              },
            ],
            details: { error: "unknown_mime" },
          };
        }

        const normalizedMediaCategory = normalizeCategory(rawCategory ?? "entity");
        const storeCategory = normalizedMediaCategory
          ? mapToStoreCategory(normalizedMediaCategory)
          : "entity";
        const categoryLabel = normalizedMediaCategory ?? "entities";
        const finalImportance =
          importance != null && Number.isFinite(importance)
            ? Math.min(1, Math.max(0, importance))
            : 0.7;

        let vector: number[];
        let finalMime = mimeType;
        const isMultimodal = deps.embedder.isMultimodal;

        if (isMultimodal) {
          // Multimodal: read file, convert if needed, embed actual media content
          let fileData: Buffer;
          try {
            const result = await convertFileForEmbedding(filePath, mimeType);
            fileData = result.data;
            finalMime = result.mimeType;
            if (result.converted) {
              deps.log(
                `vector-memory: store_media [${agentId}] converted ${mimeType} → ${finalMime}`,
              );
            }
          } catch (convErr) {
            return {
              content: [
                {
                  type: "text",
                  text: `Media conversion failed (${mimeType}): ${String(convErr)}. Try converting the file manually.`,
                },
              ],
              details: { error: "conversion_failed", mimeType },
            };
          }
          vector = await deps.embedder.embedMedia(fileData, finalMime, description.trim());
        } else {
          // Non-multimodal: embed the text description only.
          // The memory record still tracks media_type and file path for reference.
          vector = await deps.embedder.embed(description.trim());
          deps.log(
            `vector-memory: store_media [${agentId}] model is not multimodal, embedding description only`,
          );
        }

        const store = deps.storeManager.getStore(agentId);

        // Dedup check
        try {
          const existing = await store.vectorSearch(vector, 1, 0.1);
          if (existing.length > 0 && existing[0].score > 0.97) {
            return {
              content: [
                {
                  type: "text",
                  text: `Similar media memory already exists: "${truncateText(existing[0].entry.text, 80)}" (id: ${existing[0].entry.id.slice(0, 8)}, similarity: ${existing[0].score.toFixed(3)})`,
                },
              ],
              details: {
                action: "duplicate",
                existingId: existing[0].entry.id,
                similarity: existing[0].score,
                agentId,
              },
            };
          }
        } catch (err) {
          deps.log(`vector-memory: media dedup check failed, continuing: ${String(err)}`);
        }

        const metadata = JSON.stringify({
          l0_abstract: description.trim().slice(0, 200),
          l1_overview: "",
          l2_content: description.trim(),
          memory_category: categoryLabel,
          source: "agent_tool_media",
          media_type: finalMime,
          original_media_type: mimeType !== finalMime ? mimeType : undefined,
          media_path: filePath,
          multimodal_embedded: isMultimodal,
        });

        const entry = await store.store({
          text: description.trim().slice(0, 200),
          vector,
          category: storeCategory,
          importance: finalImportance,
          metadata,
        });

        deps.log(
          `vector-memory: store_media [${agentId}][${categoryLabel}] id=${entry.id.slice(0, 8)} mime=${finalMime}${mimeType !== finalMime ? ` (from ${mimeType})` : ""} multimodal=${isMultimodal}`,
        );

        const modeNote = isMultimodal ? "" : " (text-only embedding, media content not embedded)";
        return {
          content: [
            {
              type: "text",
              text: `Stored media memory: [${categoryLabel}] "${truncateText(description, 60)}" (id: ${entry.id.slice(0, 8)}, type: ${finalMime}${mimeType !== finalMime ? ` (converted from ${mimeType})` : ""}${modeNote})`,
            },
          ],
          details: {
            action: "store_media",
            id: entry.id,
            category: categoryLabel,
            importance: finalImportance,
            mimeType: finalMime,
            originalMimeType: mimeType !== finalMime ? mimeType : undefined,
            multimodal: isMultimodal,
            agentId,
          },
        };
      } catch (err) {
        deps.log(`vector-memory: store_media error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Media memory store failed: ${String(err)}` }],
          details: { error: "store_media_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Tool: vector_memory_detail
// ============================================================================

export function createDetailTool(deps: ToolDeps) {
  return (toolCtx: { agentId?: string }) => ({
    name: "vector_memory_detail",
    label: "Vector Memory Detail",
    description:
      "Retrieve the full original content of a specific memory by its ID. " +
      "Only use this when auto-recalled memories are clearly insufficient to answer the user's question " +
      "and you need the verbatim original text. Do NOT call this for every recalled memory.",
    parameters: Type.Object({
      id: Type.String({ description: "Memory ID (full or first 8 characters)" }),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { id } = params as { id: string };
      const agentId = resolveAgentId(toolCtx);

      try {
        if (!id || id.trim().length < 4) {
          return {
            content: [
              { type: "text", text: "Memory ID too short. Provide at least 4 characters." },
            ],
            details: { error: "id_too_short" },
          };
        }

        const store = deps.storeManager.getStore(agentId);

        // Try exact match first, then prefix match
        let entry = await store.getById(id.trim());
        if (!entry && id.trim().length < 36) {
          const all = await store.listAll({ limit: 500 });
          entry = all.find((e) => e.id.startsWith(id.trim())) ?? null;
        }

        if (!entry) {
          return {
            content: [{ type: "text", text: `Memory not found: ${id}` }],
            details: { error: "not_found", id },
          };
        }

        let l0 = entry.text;
        let l1 = "";
        let l2 = "";
        let category: string = entry.category;
        let mediaPath = "";
        try {
          const meta = JSON.parse(entry.metadata || "{}");
          l0 = (meta.l0_abstract as string) || entry.text;
          l1 = (meta.l1_overview as string) || "";
          l2 = (meta.l2_content as string) || "";
          category = (meta.memory_category as string) || entry.category;
          mediaPath = (meta.media_path as string) || "";
        } catch {}

        const dateStr = new Date(entry.timestamp).toISOString().split("T")[0];
        const sections = [`**[${dateStr}][${category}] ${l0}**`];
        if (l1.length > 0) {
          sections.push(`\n**Overview:**\n${l1}`);
        }
        if (l2.length > 0) {
          sections.push(`\n**Original Content:**\n${l2}`);
        }
        if (mediaPath.length > 0) {
          sections.push(`\n**Media file:** ${mediaPath}`);
        }
        sections.push(
          `\n_id: ${entry.id} | importance: ${entry.importance.toFixed(2)} | date: ${dateStr}_`,
        );

        return {
          content: [{ type: "text", text: sections.join("\n") }],
          details: {
            action: "detail",
            id: entry.id,
            category,
            importance: entry.importance,
            agentId,
          },
        };
      } catch (err) {
        deps.log(`vector-memory: detail tool error [${agentId}]: ${String(err)}`);
        return {
          content: [{ type: "text", text: `Memory detail lookup failed: ${String(err)}` }],
          details: { error: "detail_failed" },
        };
      }
    },
  });
}

// ============================================================================
// Register All Tools
// ============================================================================

export function registerAllVectorMemoryTools(
  api: { registerTool: (tool: unknown) => void },
  deps: ToolDeps,
): void {
  api.registerTool(createRecallTool(deps));
  api.registerTool(createStoreTool(deps));
  api.registerTool(createStoreMediaTool(deps));
  api.registerTool(createForgetTool(deps));
  api.registerTool(createStatsTool(deps));
  api.registerTool(createUpdateTool(deps));
  api.registerTool(createDetailTool(deps));
  api.registerTool(createListTool(deps));
  api.registerTool(createCompactTool(deps));
  api.registerTool(createPromoteTool(deps));
  api.registerTool(createArchiveTool(deps));
  api.registerTool(createExplainRankTool(deps));
}
