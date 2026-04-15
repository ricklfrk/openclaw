import type { AgentRetryConfig } from "../config/types.agent-defaults.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import type { ContextEngineInfo } from "../context-engine/types.js";
import { MIN_PROMPT_BUDGET_RATIO, MIN_PROMPT_BUDGET_TOKENS } from "./pi-compaction-constants.js";

export const DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR = 20_000;

export type PiRetryOverrides = {
  enabled?: boolean;
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
};

type PiSettingsManagerLike = {
  getCompactionReserveTokens: () => number;
  getCompactionKeepRecentTokens: () => number;
  applyOverrides: (overrides: {
    compaction?: {
      reserveTokens?: number;
      keepRecentTokens?: number;
    };
    retry?: PiRetryOverrides;
  }) => void;
  setCompactionEnabled?: (enabled: boolean) => void;
};

/**
 * Ensures the compaction reserve tokens are at least the specified minimum.
 * Note: This function is not context-aware and uses an uncapped floor.
 * If called for small-context models without threading `contextTokenBudget`,
 * it may re-introduce context overflow issues.
 */
export function ensurePiCompactionReserveTokens(params: {
  settingsManager: PiSettingsManagerLike;
  minReserveTokens?: number;
}): { didOverride: boolean; reserveTokens: number } {
  const minReserveTokens = params.minReserveTokens ?? DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR;
  const current = params.settingsManager.getCompactionReserveTokens();

  if (current >= minReserveTokens) {
    return { didOverride: false, reserveTokens: current };
  }

  params.settingsManager.applyOverrides({
    compaction: { reserveTokens: minReserveTokens },
  });

  return { didOverride: true, reserveTokens: minReserveTokens };
}

export function resolveCompactionReserveTokensFloor(cfg?: OpenClawConfig): number {
  const raw = cfg?.agents?.defaults?.compaction?.reserveTokensFloor;
  if (typeof raw === "number" && Number.isFinite(raw) && raw >= 0) {
    return Math.floor(raw);
  }
  return DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR;
}

function toNonNegativeInt(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return Math.floor(value);
}

function toPositiveInt(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return undefined;
  }
  return Math.floor(value);
}

function findAgentEntry(cfg: OpenClawConfig | undefined, agentId: string | undefined) {
  if (!agentId || !cfg?.agents?.list) {
    return undefined;
  }
  return cfg.agents.list.find((a) => a.id === agentId);
}

export function applyPiCompactionSettingsFromConfig(params: {
  settingsManager: PiSettingsManagerLike;
  cfg?: OpenClawConfig;
  /** When known, the resolved context window budget for the current model. */
  contextTokenBudget?: number;
  agentId?: string;
}): {
  didOverride: boolean;
  compaction: { reserveTokens: number; keepRecentTokens: number };
} {
  const currentReserveTokens = params.settingsManager.getCompactionReserveTokens();
  const currentKeepRecentTokens = params.settingsManager.getCompactionKeepRecentTokens();
  const defaultsCfg = params.cfg?.agents?.defaults?.compaction;
  const perAgentCfg = findAgentEntry(params.cfg, params.agentId)?.compaction;
  const compactionCfg = perAgentCfg ? { ...defaultsCfg, ...perAgentCfg } : defaultsCfg;

  const configuredReserveTokens = toNonNegativeInt(compactionCfg?.reserveTokens);
  const configuredKeepRecentTokens = toPositiveInt(compactionCfg?.keepRecentTokens);
  let reserveTokensFloor = resolveCompactionReserveTokensFloor(params.cfg);

  // Cap the floor to a safe fraction of the context window so that
  // small-context models (e.g. Ollama with 16 K tokens) are not starved of
  // prompt budget.  Without this cap the default floor of 20 000 can exceed
  // the entire context window, causing every prompt to be classified as an
  // overflow and triggering an infinite compaction loop.
  const ctxBudget = params.contextTokenBudget;
  if (typeof ctxBudget === "number" && Number.isFinite(ctxBudget) && ctxBudget > 0) {
    const minPromptBudget = Math.min(
      MIN_PROMPT_BUDGET_TOKENS,
      Math.max(1, Math.floor(ctxBudget * MIN_PROMPT_BUDGET_RATIO)),
    );
    const maxReserve = Math.max(0, ctxBudget - minPromptBudget);
    reserveTokensFloor = Math.min(reserveTokensFloor, maxReserve);
  }

  const targetReserveTokens = Math.max(
    configuredReserveTokens ?? currentReserveTokens,
    reserveTokensFloor,
  );
  const targetKeepRecentTokens = configuredKeepRecentTokens ?? currentKeepRecentTokens;

  const overrides: { reserveTokens?: number; keepRecentTokens?: number } = {};
  if (targetReserveTokens !== currentReserveTokens) {
    overrides.reserveTokens = targetReserveTokens;
  }
  if (targetKeepRecentTokens !== currentKeepRecentTokens) {
    overrides.keepRecentTokens = targetKeepRecentTokens;
  }

  const didOverride = Object.keys(overrides).length > 0;
  if (didOverride) {
    params.settingsManager.applyOverrides({ compaction: overrides });
  }

  return {
    didOverride,
    compaction: {
      reserveTokens: targetReserveTokens,
      keepRecentTokens: targetKeepRecentTokens,
    },
  };
}

/** Merge defaults.retry with per-agent retry overrides. */
export function resolveAgentRetryConfig(
  cfg: OpenClawConfig | undefined,
  agentId: string | undefined,
): AgentRetryConfig | undefined {
  const defaults = cfg?.agents?.defaults?.retry;
  const perAgent = findAgentEntry(cfg, agentId)?.retry;
  if (!defaults && !perAgent) {
    return undefined;
  }
  return { ...defaults, ...perAgent };
}

/** Apply resolved retry config to the Pi settings manager. */
export function applyPiRetrySettingsFromConfig(params: {
  settingsManager: PiSettingsManagerLike;
  cfg?: OpenClawConfig;
  agentId?: string;
}): { didOverride: boolean } {
  const retry = resolveAgentRetryConfig(params.cfg, params.agentId);
  if (!retry) {
    return { didOverride: false };
  }

  const overrides: PiRetryOverrides = {};
  if (retry.enabled !== undefined) {
    overrides.enabled = retry.enabled;
  }
  if (retry.maxRetries !== undefined) {
    overrides.maxRetries = retry.maxRetries;
  }
  if (retry.baseDelayMs !== undefined) {
    overrides.baseDelayMs = retry.baseDelayMs;
  }
  if (retry.maxDelayMs !== undefined) {
    overrides.maxDelayMs = retry.maxDelayMs;
  }

  if (Object.keys(overrides).length === 0) {
    return { didOverride: false };
  }
  params.settingsManager.applyOverrides({ retry: overrides });
  return { didOverride: true };
}

/** Decide whether Pi's internal auto-compaction should be disabled for this run. */
export function shouldDisablePiAutoCompaction(params: {
  contextEngineInfo?: ContextEngineInfo;
}): boolean {
  return params.contextEngineInfo?.ownsCompaction === true;
}

/** Disable Pi auto-compaction via settings when a context engine owns compaction. */
export function applyPiAutoCompactionGuard(params: {
  settingsManager: PiSettingsManagerLike;
  contextEngineInfo?: ContextEngineInfo;
}): { supported: boolean; disabled: boolean } {
  const disable = shouldDisablePiAutoCompaction({
    contextEngineInfo: params.contextEngineInfo,
  });
  const hasMethod = typeof params.settingsManager.setCompactionEnabled === "function";
  if (!disable || !hasMethod) {
    return { supported: hasMethod, disabled: false };
  }
  params.settingsManager.setCompactionEnabled!(false);
  return { supported: true, disabled: true };
}
