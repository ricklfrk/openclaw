import type { OpenClawConfig } from "../config/config.js";
import type { AgentRetryConfig } from "../config/types.agent-defaults.js";

export const DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR = 20_000;

type PiRetryOverrides = {
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
};

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

export function applyPiCompactionSettingsFromConfig(params: {
  settingsManager: PiSettingsManagerLike;
  cfg?: OpenClawConfig;
}): {
  didOverride: boolean;
  compaction: { reserveTokens: number; keepRecentTokens: number };
} {
  const currentReserveTokens = params.settingsManager.getCompactionReserveTokens();
  const currentKeepRecentTokens = params.settingsManager.getCompactionKeepRecentTokens();
  const compactionCfg = params.cfg?.agents?.defaults?.compaction;

  const configuredReserveTokens = toNonNegativeInt(compactionCfg?.reserveTokens);
  const configuredKeepRecentTokens = toPositiveInt(compactionCfg?.keepRecentTokens);
  const reserveTokensFloor = resolveCompactionReserveTokensFloor(params.cfg);

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

function resolveAgentRetryConfig(
  cfg?: OpenClawConfig,
  agentId?: string,
): AgentRetryConfig | undefined {
  const defaults = cfg?.agents?.defaults?.retry;
  const agentEntry = agentId ? cfg?.agents?.list?.find((a) => a.id === agentId) : undefined;
  const perAgent = agentEntry?.retry;
  if (!defaults && !perAgent) {
    return undefined;
  }
  return { ...defaults, ...perAgent };
}

export function applyPiRetrySettingsFromConfig(params: {
  settingsManager: PiSettingsManagerLike;
  cfg?: OpenClawConfig;
  agentId?: string;
}): boolean {
  const retryCfg = resolveAgentRetryConfig(params.cfg, params.agentId);
  if (!retryCfg) {
    return false;
  }
  const overrides: PiRetryOverrides = {};
  if (typeof retryCfg.enabled === "boolean") {
    overrides.enabled = retryCfg.enabled;
  }
  if (typeof retryCfg.maxRetries === "number" && Number.isFinite(retryCfg.maxRetries)) {
    overrides.maxRetries = Math.floor(retryCfg.maxRetries);
  }
  if (typeof retryCfg.baseDelayMs === "number" && Number.isFinite(retryCfg.baseDelayMs)) {
    overrides.baseDelayMs = Math.floor(retryCfg.baseDelayMs);
  }
  if (typeof retryCfg.maxDelayMs === "number" && Number.isFinite(retryCfg.maxDelayMs)) {
    overrides.maxDelayMs = Math.floor(retryCfg.maxDelayMs);
  }
  if (Object.keys(overrides).length === 0) {
    return false;
  }
  params.settingsManager.applyOverrides({ retry: overrides });
  return true;
}
