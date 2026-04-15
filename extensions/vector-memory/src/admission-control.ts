/**
 * Admission Control — lightweight quality gate for memory candidates.
 *
 * Scores candidates using: typePrior, novelty, confidence, recency.
 * No LLM utility scoring (too expensive for supplementary plugin).
 *
 * Simplified from memory-lancedb-pro.
 */

import type { MemoryCategory } from "./categories.js";
import type { MemorySearchResult, MemoryStore } from "./store.js";

export interface AdmissionWeights {
  confidence: number;
  novelty: number;
  recency: number;
  typePrior: number;
}

export interface AdmissionTypePriors {
  profile: number;
  preferences: number;
  entities: number;
  events: number;
  cases: number;
  patterns: number;
}

export type AdmissionPreset = "balanced" | "conservative" | "high-recall";

export interface AdmissionControlConfig {
  preset: AdmissionPreset;
  enabled: boolean;
  weights: AdmissionWeights;
  rejectThreshold: number;
  admitThreshold: number;
  noveltyCandidatePoolSize: number;
  recencyHalfLifeDays: number;
  typePriors: AdmissionTypePriors;
}

export interface AdmissionEvaluation {
  decision: "reject" | "pass_to_dedup";
  hint?: "add" | "update_or_merge";
  score: number;
  reason: string;
  maxSimilarity: number;
}

const DEFAULT_WEIGHTS: AdmissionWeights = {
  confidence: 0.15,
  novelty: 0.15,
  recency: 0.1,
  typePrior: 0.6,
};

const DEFAULT_TYPE_PRIORS: AdmissionTypePriors = {
  profile: 0.95,
  preferences: 0.9,
  entities: 0.75,
  events: 0.45,
  cases: 0.8,
  patterns: 0.85,
};

export const ADMISSION_PRESETS: Record<AdmissionPreset, AdmissionControlConfig> = {
  balanced: {
    preset: "balanced",
    enabled: false,
    weights: DEFAULT_WEIGHTS,
    rejectThreshold: 0.4,
    admitThreshold: 0.6,
    noveltyCandidatePoolSize: 8,
    recencyHalfLifeDays: 14,
    typePriors: DEFAULT_TYPE_PRIORS,
  },
  conservative: {
    preset: "conservative",
    enabled: false,
    weights: { confidence: 0.2, novelty: 0.25, recency: 0.1, typePrior: 0.45 },
    rejectThreshold: 0.5,
    admitThreshold: 0.68,
    noveltyCandidatePoolSize: 10,
    recencyHalfLifeDays: 10,
    typePriors: {
      profile: 0.98,
      preferences: 0.94,
      entities: 0.78,
      events: 0.28,
      cases: 0.78,
      patterns: 0.8,
    },
  },
  "high-recall": {
    preset: "high-recall",
    enabled: false,
    weights: { confidence: 0.12, novelty: 0.1, recency: 0.18, typePrior: 0.6 },
    rejectThreshold: 0.32,
    admitThreshold: 0.5,
    noveltyCandidatePoolSize: 6,
    recencyHalfLifeDays: 21,
    typePriors: {
      profile: 0.96,
      preferences: 0.92,
      entities: 0.8,
      events: 0.58,
      cases: 0.84,
      patterns: 0.88,
    },
  },
};

export const DEFAULT_ADMISSION_CONFIG = ADMISSION_PRESETS.balanced;

function clamp01(value: number, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) {
    return fallback;
  }
  return Math.min(1, Math.max(0, n));
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (!a?.length || !b?.length || a.length !== b.length) {
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

function scoreTypePrior(category: MemoryCategory, typePriors: AdmissionTypePriors): number {
  return clamp01(typePriors[category], DEFAULT_TYPE_PRIORS[category]);
}

function scoreNovelty(
  candidateVector: number[],
  matches: MemorySearchResult[],
): { score: number; maxSimilarity: number } {
  if (!candidateVector?.length || matches.length === 0) {
    return { score: 1, maxSimilarity: 0 };
  }
  let maxSimilarity = 0;
  for (const match of matches) {
    const sim = Math.max(0, cosineSimilarity(candidateVector, match.entry.vector));
    if (sim > maxSimilarity) {
      maxSimilarity = sim;
    }
  }
  return { score: clamp01(1 - maxSimilarity, 1), maxSimilarity };
}

function scoreRecencyGap(now: number, matches: MemorySearchResult[], halfLifeDays: number): number {
  if (matches.length === 0 || halfLifeDays <= 0) {
    return 1;
  }
  const latestTs = Math.max(
    ...matches.map((m) => (Number.isFinite(m.entry.timestamp) ? m.entry.timestamp : 0)),
  );
  if (latestTs <= 0) {
    return 1;
  }
  const gapDays = Math.max(0, (now - latestTs) / 86_400_000);
  if (gapDays === 0) {
    return 0;
  }
  return clamp01(1 - Math.exp(-(Math.LN2 / halfLifeDays) * gapDays), 1);
}

/** Lightweight token-overlap confidence: how much candidate text is grounded in conversation. */
function scoreConfidence(candidateText: string, conversationText: string): number {
  const tokenize = (s: string) =>
    s
      .toLowerCase()
      .split(/[\s\p{P}]+/u)
      .filter((t) => t.length >= 2);
  const candidateTokens = tokenize(candidateText);
  if (candidateTokens.length === 0) {
    return 0;
  }
  const conversationSet = new Set(tokenize(conversationText));
  const overlap = candidateTokens.filter((t) => conversationSet.has(t)).length;
  return clamp01(overlap / candidateTokens.length, 0);
}

export function normalizeAdmissionConfig(raw: unknown): AdmissionControlConfig {
  if (!raw || typeof raw !== "object") {
    return { ...DEFAULT_ADMISSION_CONFIG };
  }
  const obj = raw as Record<string, unknown>;
  const presetName =
    obj.preset === "conservative" || obj.preset === "high-recall" ? obj.preset : "balanced";
  const base = ADMISSION_PRESETS[presetName as AdmissionPreset];
  return {
    ...base,
    enabled: obj.enabled === true,
    rejectThreshold:
      typeof obj.rejectThreshold === "number"
        ? clamp01(obj.rejectThreshold, base.rejectThreshold)
        : base.rejectThreshold,
    admitThreshold:
      typeof obj.admitThreshold === "number"
        ? clamp01(obj.admitThreshold, base.admitThreshold)
        : base.admitThreshold,
  };
}

export class AdmissionController {
  constructor(
    private readonly store: MemoryStore,
    private readonly config: AdmissionControlConfig,
    private readonly log: (msg: string) => void = () => {},
  ) {}

  async evaluate(params: {
    candidateCategory: MemoryCategory;
    candidateText: string;
    candidateVector: number[];
    conversationText: string;
    now?: number;
  }): Promise<AdmissionEvaluation> {
    const now = params.now ?? Date.now();

    const matches =
      params.candidateVector.length > 0
        ? await this.store.vectorSearch(
            params.candidateVector,
            this.config.noveltyCandidatePoolSize,
            0,
          )
        : [];

    const noveltyResult = scoreNovelty(params.candidateVector, matches);
    const recency = scoreRecencyGap(now, matches, this.config.recencyHalfLifeDays);
    const confidence = scoreConfidence(params.candidateText, params.conversationText);
    const typePrior = scoreTypePrior(params.candidateCategory, this.config.typePriors);

    const score =
      confidence * this.config.weights.confidence +
      noveltyResult.score * this.config.weights.novelty +
      recency * this.config.weights.recency +
      typePrior * this.config.weights.typePrior;

    const decision = score < this.config.rejectThreshold ? "reject" : "pass_to_dedup";
    const hint =
      decision === "reject"
        ? undefined
        : score >= this.config.admitThreshold && noveltyResult.maxSimilarity < 0.55
          ? "add"
          : "update_or_merge";

    const reason =
      decision === "reject"
        ? `Rejected (${score.toFixed(3)} < ${this.config.rejectThreshold})`
        : `Passed (${score.toFixed(3)})${hint ? ` hint=${hint}` : ""}`;

    this.log(
      `vector-memory: admission: ${reason} category=${params.candidateCategory} maxSim=${noveltyResult.maxSimilarity.toFixed(3)}`,
    );

    return { decision, hint, score, reason, maxSimilarity: noveltyResult.maxSimilarity };
  }
}
