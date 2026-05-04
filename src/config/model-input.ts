import { normalizeOptionalString, resolvePrimaryStringValue } from "../shared/string-coerce.js";
import type { AgentModelConfig } from "./types.agents-shared.js";

type AgentModelListLike = {
  primary?: string;
  fallbacks?: string[];
  timeoutMs?: number;
};

export function resolveAgentModelPrimaryValue(model?: AgentModelConfig): string | undefined {
  return resolvePrimaryStringValue(model);
}

export function resolveAgentModelFallbackValues(model?: AgentModelConfig): string[] {
  if (Array.isArray(model)) {
    return normalizeModelListValues(model).slice(1);
  }
  if (!model || typeof model !== "object") {
    return [];
  }
  return Array.isArray(model.fallbacks) ? model.fallbacks : [];
}

export function resolveAgentModelTimeoutMsValue(model?: AgentModelConfig): number | undefined {
  if (!model || typeof model !== "object") {
    return undefined;
  }
  return typeof model.timeoutMs === "number" &&
    Number.isFinite(model.timeoutMs) &&
    model.timeoutMs > 0
    ? Math.floor(model.timeoutMs)
    : undefined;
}

export function toAgentModelListLike(model?: AgentModelConfig): AgentModelListLike | undefined {
  if (typeof model === "string") {
    const primary = normalizeOptionalString(model);
    return primary ? { primary } : undefined;
  }
  if (Array.isArray(model)) {
    const [primary, ...fallbacks] = normalizeModelListValues(model);
    return primary ? { primary, ...(fallbacks.length > 0 ? { fallbacks } : {}) } : undefined;
  }
  if (!model || typeof model !== "object") {
    return undefined;
  }
  return model;
}

export function normalizeModelListValues(model?: AgentModelConfig): string[] {
  if (typeof model === "string") {
    const primary = normalizeOptionalString(model);
    return primary ? [primary] : [];
  }
  if (Array.isArray(model)) {
    return model.flatMap((entry) => {
      const normalized = normalizeOptionalString(entry);
      return normalized ? [normalized] : [];
    });
  }
  if (!model || typeof model !== "object") {
    return [];
  }
  const primary = normalizeOptionalString(model.primary);
  const fallbacks = Array.isArray(model.fallbacks)
    ? model.fallbacks.flatMap((entry) => {
        const normalized = normalizeOptionalString(entry);
        return normalized ? [normalized] : [];
      })
    : [];
  return primary ? [primary, ...fallbacks] : fallbacks;
}
