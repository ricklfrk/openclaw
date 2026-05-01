/**
 * Plugin-local writer for the core pending-reset flag.
 *
 * Mirrors the on-disk contract defined by
 * `src/config/sessions/pending-reset-flag.ts`. Duplicating the path/payload
 * logic avoids crossing the plugin boundary back into `src/**` while still
 * keeping the extension a single read away from the core contract: if core
 * ever changes the path layout, this file and the matching core helper must
 * change together (see the existing tests in both locations).
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type PendingResetPayload = {
  reason?: string;
  requestedAt?: number;
  source?: string;
};

export function resolvePendingResetDir(homeOverride?: string): string {
  const home = homeOverride?.trim() || process.env.OPENCLAW_HOME_OVERRIDE?.trim() || os.homedir();
  return path.join(home, ".openclaw", "state", "pending-reset");
}

export function resolvePendingResetFlagPath(sessionKey: string, homeOverride?: string): string {
  return path.join(resolvePendingResetDir(homeOverride), `${encodeURIComponent(sessionKey)}.json`);
}

export function writePendingResetFlag(
  sessionKey: string,
  payload: PendingResetPayload = {},
  opts?: { homeOverride?: string },
): string {
  const flagPath = resolvePendingResetFlagPath(sessionKey, opts?.homeOverride);
  fs.mkdirSync(path.dirname(flagPath), { recursive: true });
  const body: PendingResetPayload = {
    requestedAt: Date.now(),
    ...payload,
  };
  fs.writeFileSync(flagPath, JSON.stringify(body));
  return flagPath;
}

export function hasPendingResetFlag(sessionKey: string, homeOverride?: string): boolean {
  try {
    return fs.existsSync(resolvePendingResetFlagPath(sessionKey, homeOverride));
  } catch {
    return false;
  }
}
