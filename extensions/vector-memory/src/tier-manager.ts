/**
 * Tier Manager — Three-tier memory promotion/demotion system.
 *
 * Tiers:
 * - Core (decay floor 0.9): Identity-level facts, almost never forgotten
 * - Working (decay floor 0.7): Active context, ages out without reinforcement
 * - Peripheral (decay floor 0.5): Low-priority or aging memories
 *
 * Adapted from memory-lancedb-pro.
 */

import type { DecayScore } from "./types.js";
import type { MemoryTier } from "./types.js";
export type { MemoryTier } from "./types.js";

export interface TierConfig {
  coreAccessThreshold: number;
  coreCompositeThreshold: number;
  coreImportanceThreshold: number;
  peripheralCompositeThreshold: number;
  peripheralAgeDays: number;
  workingAccessThreshold: number;
  workingCompositeThreshold: number;
}

export const DEFAULT_TIER_CONFIG: TierConfig = {
  coreAccessThreshold: 10,
  coreCompositeThreshold: 0.7,
  coreImportanceThreshold: 0.8,
  peripheralCompositeThreshold: 0.15,
  peripheralAgeDays: 60,
  workingAccessThreshold: 3,
  workingCompositeThreshold: 0.4,
};

export interface TierTransition {
  memoryId: string;
  fromTier: MemoryTier;
  toTier: MemoryTier;
  reason: string;
}

export interface TierableMemory {
  id: string;
  tier: MemoryTier;
  importance: number;
  accessCount: number;
  createdAt: number;
}

export interface TierManager {
  evaluate(memory: TierableMemory, decayScore: DecayScore, now?: number): TierTransition | null;
  evaluateAll(
    memories: TierableMemory[],
    decayScores: DecayScore[],
    now?: number,
  ): TierTransition[];
}

const MS_PER_DAY = 86_400_000;

export function createTierManager(config: TierConfig = DEFAULT_TIER_CONFIG): TierManager {
  function evaluate(
    memory: TierableMemory,
    decayScore: DecayScore,
    now: number = Date.now(),
  ): TierTransition | null {
    const ageDays = (now - memory.createdAt) / MS_PER_DAY;

    switch (memory.tier) {
      case "peripheral": {
        if (
          memory.accessCount >= config.workingAccessThreshold &&
          decayScore.composite >= config.workingCompositeThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "peripheral",
            toTier: "working",
            reason: `access=${memory.accessCount} composite=${decayScore.composite.toFixed(2)}`,
          };
        }
        break;
      }
      case "working": {
        if (
          memory.accessCount >= config.coreAccessThreshold &&
          decayScore.composite >= config.coreCompositeThreshold &&
          memory.importance >= config.coreImportanceThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "working",
            toTier: "core",
            reason: `high access/composite/importance`,
          };
        }
        if (
          decayScore.composite < config.peripheralCompositeThreshold ||
          (ageDays > config.peripheralAgeDays && memory.accessCount < config.workingAccessThreshold)
        ) {
          return {
            memoryId: memory.id,
            fromTier: "working",
            toTier: "peripheral",
            reason: `low composite or aged ${ageDays.toFixed(0)}d`,
          };
        }
        break;
      }
      case "core": {
        if (
          decayScore.composite < config.peripheralCompositeThreshold &&
          memory.accessCount < config.workingAccessThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "core",
            toTier: "working",
            reason: `severely low composite and access`,
          };
        }
        break;
      }
    }
    return null;
  }

  return {
    evaluate,
    evaluateAll(memories, decayScores, now = Date.now()) {
      const scoreMap = new Map(decayScores.map((s) => [s.memoryId, s]));
      const transitions: TierTransition[] = [];
      for (const memory of memories) {
        const score = scoreMap.get(memory.id);
        if (!score) {
          continue;
        }
        const transition = evaluate(memory, score, now);
        if (transition) {
          transitions.push(transition);
        }
      }
      return transitions;
    },
  };
}
