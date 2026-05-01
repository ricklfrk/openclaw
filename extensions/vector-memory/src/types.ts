/** Shared types between decay-engine and tier-manager to avoid circular imports. */

export type MemoryTier = "core" | "working" | "peripheral";

export interface DecayScore {
  memoryId: string;
  recency: number;
  frequency: number;
  intrinsic: number;
  composite: number;
}
