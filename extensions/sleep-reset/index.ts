/**
 * Sleep Reset Plugin
 *
 * Flags sessions for a `/new`-style reset once the user has been idle long
 * enough during their configured sleep window. Uses the core pending-reset
 * flag contract (see `src/config/sessions/pending-reset-flag.ts`) so the
 * reset only triggers on the *next non-system* user turn — heartbeats cannot
 * consume the flag and keep a stale overnight session alive.
 *
 * Hooks:
 *   gateway_start -> start the periodic scan loop
 *   gateway_stop  -> clear the interval on shutdown
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { hasPendingResetFlag, writePendingResetFlag } from "./src/flag.js";
import {
  type SessionActivitySnapshot,
  type SleepResetConfig,
  decideSessionsToFlag,
  hourInTimezone,
  isWithinWindow,
  normalizeSleepResetConfig,
} from "./src/policy.js";

let registered = false;

export default definePluginEntry({
  id: "sleep-reset",
  name: "Sleep Reset",
  description: "Flag idle sessions for a /new-style reset during the configured sleep window.",

  register(api) {
    if (registered) {
      return;
    }
    registered = true;

    const log = api.logger;
    const { runtime } = api;
    const config = normalizeSleepResetConfig(api.pluginConfig);

    log.info(
      `sleep-reset: registered tz=${config.timezone} window=${config.windowStartHour}-${config.windowEndHour} idle>=${config.minIdleHours}h every ${config.checkIntervalMinutes}m agents=${config.agentIds.join(",")}`,
    );

    let intervalHandle: NodeJS.Timeout | undefined;

    const loadSnapshot = (agentId: string): SessionActivitySnapshot[] => {
      try {
        const storePath = runtime.agent.session.resolveStorePath(undefined, { agentId });
        const store = runtime.agent.session.loadSessionStore(storePath);
        const out: SessionActivitySnapshot[] = [];
        for (const [sessionKey, entry] of Object.entries(store)) {
          if (!entry || typeof entry !== "object") {
            continue;
          }
          const e = entry as {
            updatedAt?: number;
            lastUserUpdatedAt?: number;
          };
          if (typeof e.updatedAt !== "number") {
            continue;
          }
          out.push({
            sessionKey,
            updatedAt: e.updatedAt,
            lastUserUpdatedAt:
              typeof e.lastUserUpdatedAt === "number" ? e.lastUserUpdatedAt : undefined,
          });
        }
        return out;
      } catch (err) {
        log.warn(`sleep-reset: failed to load session store for agent=${agentId}: ${String(err)}`);
        return [];
      }
    };

    const runCheck = (reason: "tick" | "startup"): void => {
      const now = Date.now();
      const hour = hourInTimezone(now, config.timezone);
      const inWindow = isWithinWindow(hour, config.windowStartHour, config.windowEndHour);
      log.info(
        `sleep-reset: ${reason} check hour=${hour} (${config.timezone}) inWindow=${inWindow}`,
      );
      if (!inWindow) {
        return;
      }

      let flagged = 0;
      let skippedAlready = 0;
      for (const agentId of config.agentIds) {
        const snapshots = loadSnapshot(agentId);
        if (snapshots.length === 0) {
          continue;
        }
        const decisions = decideSessionsToFlag({
          nowMs: now,
          sessions: snapshots,
          config,
        });
        for (const decision of decisions) {
          if (!decision.flag) {
            continue;
          }
          if (hasPendingResetFlag(decision.sessionKey)) {
            skippedAlready += 1;
            continue;
          }
          try {
            writePendingResetFlag(decision.sessionKey, {
              reason: "sleep",
              source: "sleep-reset",
            });
            flagged += 1;
            const idleHours = Math.round((decision.idleMs / 3_600_000) * 10) / 10;
            log.info(
              `sleep-reset: flagged session=${decision.sessionKey} agent=${agentId} idle=${idleHours}h`,
            );
          } catch (err) {
            log.warn(
              `sleep-reset: failed to write flag for session=${decision.sessionKey}: ${String(err)}`,
            );
          }
        }
      }
      if (flagged > 0 || skippedAlready > 0) {
        log.info(`sleep-reset: scan summary flagged=${flagged} alreadyFlagged=${skippedAlready}`);
      }
    };

    api.on("gateway_start", () => {
      // Kick off an immediate check in case the gateway restarted inside the
      // window (so we do not wait up to checkIntervalMinutes for the first
      // scan). Wrap in setTimeout(0) so plugin registration cannot synchronously
      // block other plugins behind an I/O-heavy scan.
      setTimeout(() => runCheck("startup"), 0);
      const periodMs = Math.max(1, config.checkIntervalMinutes) * 60 * 1000;
      intervalHandle = setInterval(() => runCheck("tick"), periodMs);
      // Don't keep the event loop alive just because of this timer.
      intervalHandle.unref?.();
    });

    api.on("gateway_stop", () => {
      if (intervalHandle) {
        clearInterval(intervalHandle);
        intervalHandle = undefined;
        log.info("sleep-reset: cleared scan interval on gateway stop");
      }
    });
  },
});

export type { SleepResetConfig };
export { decideSessionsToFlag, hourInTimezone, isWithinWindow, normalizeSleepResetConfig };
