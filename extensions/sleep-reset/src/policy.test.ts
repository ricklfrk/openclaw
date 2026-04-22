import { describe, expect, it } from "vitest";
import {
  DEFAULT_SLEEP_RESET_CONFIG,
  decideSessionsToFlag,
  hourInTimezone,
  isWithinWindow,
  normalizeSleepResetConfig,
  type SleepResetConfig,
} from "./policy.js";

describe("sleep-reset policy", () => {
  describe("normalizeSleepResetConfig", () => {
    it("returns defaults for empty input", () => {
      expect(normalizeSleepResetConfig(undefined)).toEqual(DEFAULT_SLEEP_RESET_CONFIG);
      expect(normalizeSleepResetConfig({})).toEqual(DEFAULT_SLEEP_RESET_CONFIG);
    });

    it("honors valid overrides", () => {
      const cfg = normalizeSleepResetConfig({
        timezone: "America/Los_Angeles",
        windowStartHour: 6,
        windowEndHour: 9,
        minIdleHours: 3.5,
        checkIntervalMinutes: 15,
        agentIds: ["main", "foo"],
      });
      expect(cfg).toEqual({
        timezone: "America/Los_Angeles",
        windowStartHour: 6,
        windowEndHour: 9,
        minIdleHours: 3.5,
        checkIntervalMinutes: 15,
        agentIds: ["main", "foo"],
      });
    });

    it("clamps invalid values to defaults or safe bounds", () => {
      const cfg = normalizeSleepResetConfig({
        timezone: 42,
        windowStartHour: -5,
        windowEndHour: 42,
        minIdleHours: -2,
        checkIntervalMinutes: 0,
        agentIds: "not-an-array",
      });
      expect(cfg.timezone).toBe(DEFAULT_SLEEP_RESET_CONFIG.timezone);
      expect(cfg.windowStartHour).toBe(0);
      expect(cfg.windowEndHour).toBe(23);
      expect(cfg.minIdleHours).toBe(0);
      expect(cfg.checkIntervalMinutes).toBeGreaterThanOrEqual(1);
      expect(cfg.agentIds).toEqual(DEFAULT_SLEEP_RESET_CONFIG.agentIds);
    });

    it("falls back to default agentIds when the array is empty or all strings are blank", () => {
      expect(normalizeSleepResetConfig({ agentIds: [] }).agentIds).toEqual(["main"]);
      expect(normalizeSleepResetConfig({ agentIds: ["", "  "] }).agentIds).toEqual(["main"]);
    });
  });

  describe("hourInTimezone", () => {
    it("returns a valid hour in [0..23] for a known timezone", () => {
      const nowMs = Date.now();
      const hour = hourInTimezone(nowMs, "Asia/Hong_Kong");
      expect(hour).toBeGreaterThanOrEqual(0);
      expect(hour).toBeLessThanOrEqual(23);
    });

    it("falls back to local host hour when timezone is invalid", () => {
      const nowMs = Date.now();
      const hour = hourInTimezone(nowMs, "Not/AZone");
      expect(hour).toBeGreaterThanOrEqual(0);
      expect(hour).toBeLessThanOrEqual(23);
    });
  });

  describe("isWithinWindow", () => {
    it("handles standard forward windows", () => {
      expect(isWithinWindow(5, 5, 8)).toBe(true);
      expect(isWithinWindow(7, 5, 8)).toBe(true);
      expect(isWithinWindow(8, 5, 8)).toBe(false);
      expect(isWithinWindow(4, 5, 8)).toBe(false);
    });

    it("handles wrap-around windows", () => {
      // 22:00 to 06:00
      expect(isWithinWindow(23, 22, 6)).toBe(true);
      expect(isWithinWindow(2, 22, 6)).toBe(true);
      expect(isWithinWindow(6, 22, 6)).toBe(false);
      expect(isWithinWindow(12, 22, 6)).toBe(false);
    });

    it("treats start==end as always-false", () => {
      expect(isWithinWindow(5, 5, 5)).toBe(false);
    });
  });

  describe("decideSessionsToFlag", () => {
    const now = Date.UTC(2026, 3, 22, 6, 0, 0); // arbitrary UTC moment
    // Use a tz we can predict relative to the fixed `now`.
    const base: SleepResetConfig = {
      timezone: "UTC",
      windowStartHour: 5,
      windowEndHour: 8,
      minIdleHours: 4,
      checkIntervalMinutes: 30,
      agentIds: ["main"],
    };

    it("flags sessions with idle time above threshold during the window", () => {
      const sessions = [
        {
          sessionKey: "idle",
          lastUserUpdatedAt: now - 5 * 60 * 60 * 1000,
          updatedAt: now - 60_000,
        },
        {
          sessionKey: "recent",
          lastUserUpdatedAt: now - 60 * 60 * 1000,
          updatedAt: now - 60_000,
        },
      ];
      const decisions = decideSessionsToFlag({ nowMs: now, sessions, config: base });
      expect(decisions.find((d) => d.sessionKey === "idle")?.flag).toBe(true);
      expect(decisions.find((d) => d.sessionKey === "recent")?.flag).toBe(false);
    });

    it("does nothing when outside the window", () => {
      const outsideNow = Date.UTC(2026, 3, 22, 12, 0, 0); // noon UTC, not in 05-08
      const sessions = [
        {
          sessionKey: "very-idle",
          lastUserUpdatedAt: outsideNow - 24 * 60 * 60 * 1000,
          updatedAt: outsideNow - 60_000,
        },
      ];
      const decisions = decideSessionsToFlag({
        nowMs: outsideNow,
        sessions,
        config: base,
      });
      expect(decisions.every((d) => !d.flag)).toBe(true);
    });

    it("falls back to updatedAt when lastUserUpdatedAt is missing", () => {
      const sessions = [
        {
          sessionKey: "legacy",
          lastUserUpdatedAt: undefined,
          updatedAt: now - 5 * 60 * 60 * 1000,
        },
      ];
      const decisions = decideSessionsToFlag({ nowMs: now, sessions, config: base });
      expect(decisions[0].flag).toBe(true);
    });

    it("never flags when neither timestamp is valid", () => {
      const sessions = [
        {
          sessionKey: "bad",
          lastUserUpdatedAt: undefined,
          updatedAt: 0,
        },
      ];
      const decisions = decideSessionsToFlag({ nowMs: now, sessions, config: base });
      expect(decisions[0].flag).toBe(false);
      expect(decisions[0].reason).toBe("no-activity");
    });

    it("reports idleMs even when below threshold", () => {
      const sessions = [
        {
          sessionKey: "mild",
          lastUserUpdatedAt: now - 60 * 60 * 1000,
          updatedAt: now - 60_000,
        },
      ];
      const decisions = decideSessionsToFlag({ nowMs: now, sessions, config: base });
      expect(decisions[0].flag).toBe(false);
      expect(decisions[0].reason).toBe("below-threshold");
      expect(decisions[0].idleMs).toBeGreaterThan(0);
    });
  });
});
