/**
 * Decay Engine — Weibull stretched-exponential decay model.
 *
 * Composite score = recencyWeight * recency + frequencyWeight * frequency + intrinsicWeight * intrinsic
 *
 * - Recency: Weibull decay with importance-modulated half-life and tier-specific beta
 * - Frequency: Logarithmic saturation with time-weighted access pattern bonus
 * - Intrinsic: importance * confidence
 *
 * Adapted from memory-lancedb-pro.
 */

import type { MemoryTier } from "./types.js";

const MS_PER_DAY = 86_400_000;

export interface DecayConfig {
  recencyHalfLifeDays: number;
  recencyWeight: number;
  frequencyWeight: number;
  intrinsicWeight: number;
  staleThreshold: number;
  searchBoostMin: number;
  importanceModulation: number;
  betaCore: number;
  betaWorking: number;
  betaPeripheral: number;
  coreDecayFloor: number;
  workingDecayFloor: number;
  peripheralDecayFloor: number;
}

export const DEFAULT_DECAY_CONFIG: DecayConfig = {
  recencyHalfLifeDays: 30,
  recencyWeight: 0.4,
  frequencyWeight: 0.3,
  intrinsicWeight: 0.3,
  staleThreshold: 0.3,
  searchBoostMin: 0.3,
  importanceModulation: 1.5,
  betaCore: 0.8,
  betaWorking: 1.0,
  betaPeripheral: 1.3,
  coreDecayFloor: 0.9,
  workingDecayFloor: 0.7,
  peripheralDecayFloor: 0.5,
};

import type { DecayScore } from "./types.js";
export type { DecayScore } from "./types.js";

export interface DecayableMemory {
  id: string;
  importance: number;
  confidence: number;
  tier: MemoryTier;
  accessCount: number;
  createdAt: number;
  lastAccessedAt: number;
}

export interface DecayEngine {
  score(memory: DecayableMemory, now?: number): DecayScore;
  scoreAll(memories: DecayableMemory[], now?: number): DecayScore[];
  applySearchBoost(results: Array<{ memory: DecayableMemory; score: number }>, now?: number): void;
  getStaleMemories(memories: DecayableMemory[], now?: number): DecayScore[];
}

export function createDecayEngine(config: DecayConfig = DEFAULT_DECAY_CONFIG): DecayEngine {
  const {
    recencyHalfLifeDays: halfLife,
    recencyWeight: rw,
    frequencyWeight: fw,
    intrinsicWeight: iw,
    staleThreshold,
    searchBoostMin: boostMin,
    importanceModulation: mu,
    betaCore,
    betaWorking,
    betaPeripheral,
    coreDecayFloor,
    workingDecayFloor,
    peripheralDecayFloor,
  } = config;

  function getTierBeta(tier: MemoryTier): number {
    switch (tier) {
      case "core":
        return betaCore;
      case "working":
        return betaWorking;
      case "peripheral":
        return betaPeripheral;
      default:
        return betaWorking;
    }
  }

  function getTierFloor(tier: MemoryTier): number {
    switch (tier) {
      case "core":
        return coreDecayFloor;
      case "working":
        return workingDecayFloor;
      case "peripheral":
        return peripheralDecayFloor;
      default:
        return workingDecayFloor;
    }
  }

  function recency(memory: DecayableMemory, now: number): number {
    const lastActive = memory.accessCount > 0 ? memory.lastAccessedAt : memory.createdAt;
    const daysSince = Math.max(0, (now - lastActive) / MS_PER_DAY);
    const effectiveHL = halfLife * Math.exp(mu * memory.importance);
    const lambda = Math.LN2 / effectiveHL;
    const beta = getTierBeta(memory.tier);
    return Math.exp(-lambda * Math.pow(daysSince, beta));
  }

  function frequency(memory: DecayableMemory): number {
    const base = 1 - Math.exp(-memory.accessCount / 5);
    if (memory.accessCount <= 1) {
      return base;
    }
    const lastActive = memory.accessCount > 0 ? memory.lastAccessedAt : memory.createdAt;
    const accessSpanDays = Math.max(1, (lastActive - memory.createdAt) / MS_PER_DAY);
    const avgGapDays = accessSpanDays / Math.max(memory.accessCount - 1, 1);
    const recentnessBonus = Math.exp(-avgGapDays / 30);
    return base * (0.5 + 0.5 * recentnessBonus);
  }

  function intrinsic(memory: DecayableMemory): number {
    return memory.importance * memory.confidence;
  }

  function scoreOne(memory: DecayableMemory, now: number): DecayScore {
    const r = recency(memory, now);
    const f = frequency(memory);
    const i = intrinsic(memory);
    return {
      memoryId: memory.id,
      recency: r,
      frequency: f,
      intrinsic: i,
      composite: rw * r + fw * f + iw * i,
    };
  }

  return {
    score(memory, now = Date.now()) {
      return scoreOne(memory, now);
    },
    scoreAll(memories, now = Date.now()) {
      return memories.map((m) => scoreOne(m, now));
    },
    applySearchBoost(results, now = Date.now()) {
      for (const r of results) {
        const ds = scoreOne(r.memory, now);
        const tierFloor = Math.max(getTierFloor(r.memory.tier), ds.composite);
        const multiplier = boostMin + (1 - boostMin) * tierFloor;
        r.score *= Math.min(1, Math.max(boostMin, multiplier));
      }
    },
    getStaleMemories(memories, now = Date.now()) {
      return memories
        .map((m) => scoreOne(m, now))
        .filter((s) => s.composite < staleThreshold)
        .toSorted((a, b) => a.composite - b.composite);
    },
  };
}
