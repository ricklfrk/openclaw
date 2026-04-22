/**
 * Pure helpers for the sleep-reset policy. Kept side-effect-free so they can
 * be tested without touching the filesystem, system clock, or session store.
 */

export type SleepResetConfig = {
  timezone: string;
  windowStartHour: number;
  windowEndHour: number;
  minIdleHours: number;
  checkIntervalMinutes: number;
  agentIds: string[];
};

export const DEFAULT_SLEEP_RESET_CONFIG: SleepResetConfig = {
  timezone: "Asia/Hong_Kong",
  windowStartHour: 5,
  windowEndHour: 8,
  minIdleHours: 4,
  checkIntervalMinutes: 30,
  agentIds: ["main"],
};

/** Normalize an untrusted raw config (from plugins.entries.sleep-reset.config). */
export function normalizeSleepResetConfig(raw: unknown): SleepResetConfig {
  const obj = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const base = DEFAULT_SLEEP_RESET_CONFIG;

  const timezone =
    typeof obj.timezone === "string" && obj.timezone.trim() ? obj.timezone.trim() : base.timezone;
  const windowStartHour = clampHour(obj.windowStartHour, base.windowStartHour);
  const windowEndHour = clampHour(obj.windowEndHour, base.windowEndHour);
  const minIdleHours = clampPositive(obj.minIdleHours, base.minIdleHours);
  const checkIntervalMinutes = clampPositive(obj.checkIntervalMinutes, base.checkIntervalMinutes, {
    min: 1,
  });
  const agentIds = Array.isArray(obj.agentIds)
    ? obj.agentIds
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .filter((value): value is string => value.length > 0)
    : base.agentIds;

  return {
    timezone,
    windowStartHour,
    windowEndHour,
    minIdleHours,
    checkIntervalMinutes,
    agentIds: agentIds.length > 0 ? agentIds : base.agentIds,
  };
}

function clampHour(raw: unknown, fallback: number): number {
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    return fallback;
  }
  const n = Math.floor(raw);
  if (n < 0) {
    return 0;
  }
  if (n > 23) {
    return 23;
  }
  return n;
}

function clampPositive(raw: unknown, fallback: number, opts?: { min?: number }): number {
  const min = opts?.min ?? 0;
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    return fallback;
  }
  return Math.max(min, raw);
}

/**
 * Return the hour-of-day in the configured timezone for the given moment.
 *
 * Uses Intl.DateTimeFormat so DST and IANA zone names work without a 3rd-party
 * tz library. If the timezone is invalid, falls back to the host's local hour.
 */
export function hourInTimezone(nowMs: number, timezone: string): number {
  try {
    const formatter = new Intl.DateTimeFormat("en-US", {
      hour: "numeric",
      hour12: false,
      timeZone: timezone,
    });
    const parts = formatter.formatToParts(new Date(nowMs));
    const hourPart = parts.find((p) => p.type === "hour");
    if (!hourPart) {
      return new Date(nowMs).getHours();
    }
    const parsed = Number.parseInt(hourPart.value, 10);
    // "24" can appear on some ICU versions at midnight; normalize.
    if (parsed === 24) {
      return 0;
    }
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  } catch {
    // fall through to local
  }
  return new Date(nowMs).getHours();
}

/**
 * True when `hour` falls inside the `[startHour, endHour)` window.
 *
 * Supports wrap-around (e.g. windowStart=22, windowEnd=6 for a 22:00–06:00
 * window) so the plugin can be repurposed for non-morning schedules.
 */
export function isWithinWindow(hour: number, startHour: number, endHour: number): boolean {
  if (startHour === endHour) {
    return false;
  }
  if (startHour < endHour) {
    return hour >= startHour && hour < endHour;
  }
  // Wrap-around window (e.g. 22..6): inside when hour >= start OR hour < end.
  return hour >= startHour || hour < endHour;
}

export type SessionActivitySnapshot = {
  sessionKey: string;
  lastUserUpdatedAt?: number;
  updatedAt: number;
};

export type SessionFlagDecision = {
  sessionKey: string;
  flag: boolean;
  idleMs: number;
  reason: "no-activity" | "below-threshold" | "flag";
};

/**
 * Decide which sessions should be flagged for reset given the sleep policy.
 *
 * Rules:
 *  • Only runs when `nowMs` is inside the configured window.
 *  • A session is flagged when `nowMs - lastUserUpdatedAt >= minIdleHours`.
 *  • Sessions without a `lastUserUpdatedAt` (legacy entries or freshly created
 *    by a system event) fall back to `updatedAt`; this is conservative
 *    because heartbeats advance `updatedAt`, so older heartbeat-only activity
 *    looks recent and won't be flagged. That is fine for a first rollout —
 *    once real user turns have written `lastUserUpdatedAt` the plugin works
 *    as designed.
 */
export function decideSessionsToFlag(params: {
  nowMs: number;
  sessions: SessionActivitySnapshot[];
  config: SleepResetConfig;
}): SessionFlagDecision[] {
  const { nowMs, sessions, config } = params;
  const hour = hourInTimezone(nowMs, config.timezone);
  if (!isWithinWindow(hour, config.windowStartHour, config.windowEndHour)) {
    return sessions.map((s) => ({
      sessionKey: s.sessionKey,
      flag: false,
      idleMs: 0,
      reason: "below-threshold",
    }));
  }
  const thresholdMs = config.minIdleHours * 60 * 60 * 1000;
  return sessions.map((s) => {
    const reference = s.lastUserUpdatedAt ?? s.updatedAt;
    if (!Number.isFinite(reference) || reference <= 0) {
      return {
        sessionKey: s.sessionKey,
        flag: false,
        idleMs: 0,
        reason: "no-activity",
      };
    }
    const idleMs = Math.max(0, nowMs - reference);
    if (idleMs < thresholdMs) {
      return {
        sessionKey: s.sessionKey,
        flag: false,
        idleMs,
        reason: "below-threshold",
      };
    }
    return { sessionKey: s.sessionKey, flag: true, idleMs, reason: "flag" };
  });
}
