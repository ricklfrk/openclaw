/**
 * Memory Categories — 6-category classification system.
 * Directly from memory-lancedb-pro.
 */

export const MEMORY_CATEGORIES = [
  "profile",
  "preferences",
  "entities",
  "events",
  "cases",
  "patterns",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const ALWAYS_MERGE_CATEGORIES = new Set<MemoryCategory>(["profile"]);

export const MERGE_SUPPORTED_CATEGORIES = new Set<MemoryCategory>([
  "profile",
  "preferences",
  "entities",
  "patterns",
]);

export const TEMPORAL_VERSIONED_CATEGORIES = new Set<MemoryCategory>(["preferences", "entities"]);

export interface CandidateMemory {
  category: MemoryCategory;
  abstract: string;
  overview: string;
  content: string;
  entityTags?: string[];
}

export interface DedupDecision {
  decision: "create" | "merge" | "skip" | "supersede";
  reason: string;
  matchId?: string;
}

export interface ExtractionStats {
  created: number;
  merged: number;
  skipped: number;
  superseded?: number;
}

export function normalizeCategory(raw: string): MemoryCategory | null {
  const lower = raw.toLowerCase().trim();

  const directMatch = MEMORY_CATEGORIES.find((c) => c === lower);
  if (directMatch) {
    return directMatch;
  }

  const aliases: Record<string, MemoryCategory> = {
    preference: "preferences",
    entity: "entities",
    event: "events",
    case: "cases",
    pattern: "patterns",
    fact: "profile",
    identity: "profile",
    habit: "preferences",
    person: "entities",
    project: "entities",
    decision: "events",
    milestone: "events",
    solution: "cases",
    workflow: "patterns",
    process: "patterns",
  };

  return aliases[lower] ?? null;
}

/**
 * Map 6-category to store's 5-category for backward compatibility.
 */
export function mapToStoreCategory(
  category: MemoryCategory,
): "preference" | "fact" | "decision" | "entity" | "other" {
  switch (category) {
    case "profile":
      return "fact";
    case "preferences":
      return "preference";
    case "entities":
      return "entity";
    case "events":
      return "decision";
    case "cases":
      return "fact";
    case "patterns":
      return "other";
    default:
      return "other";
  }
}

export function getDefaultImportance(category: MemoryCategory): number {
  switch (category) {
    case "profile":
      return 0.9;
    case "preferences":
      return 0.8;
    case "entities":
      return 0.7;
    case "events":
      return 0.6;
    case "cases":
      return 0.8;
    case "patterns":
      return 0.85;
    default:
      return 0.5;
  }
}
