/**
 * File-based pending-reset flag.
 *
 * Any plugin or tool can write a flag file at
 *   ~/.openclaw/state/pending-reset/<encodedSessionKey>.json
 * to request that the *next non-system* turn for `sessionKey` forces a session
 * reset (as if the user had typed `/new`). Core consumes and deletes the flag
 * inside `initSessionState` so the file is a one-shot signal.
 *
 * The contract here is the on-disk path layout; the helper functions are just
 * a convenience wrapper. Plugins that cannot import this module are free to
 * compute the same path and write the same payload directly — the path layout
 * is the authoritative contract, not the TypeScript surface.
 *
 * This helper lives in `src/config/sessions` (not in `src/plugin-sdk/*`) on
 * purpose: core is the only always-present consumer of the flag, and adding a
 * new public plugin-SDK subpath just to re-export three tiny helpers would
 * force an API baseline bump. Bundled plugins that want a typed seam can
 * duplicate the path logic; the sleep-reset plugin does exactly that in
 * `extensions/sleep-reset/src/flag.ts`.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/** Payload stored in each pending-reset flag file. */
export type PendingResetFlagPayload = {
  /**
   * Optional short machine-readable reason (e.g. `"sleep"`). Propagated to
   * `previousSessionEndReason` when core consumes the flag, which then surfaces
   * through the `session_end` plugin hook. Keep it stable because the value
   * appears in `PluginHookSessionEndReason`.
   */
  reason?: string;
  /** Epoch-ms when the flag was written. Diagnostic only. */
  requestedAt?: number;
  /** Free-form plugin or tool id that wrote the flag. Diagnostic only. */
  source?: string;
};

/** Directory override used only by tests to redirect flag files. */
let testDirOverride: string | undefined;

/**
 * Test-only: redirect the state directory. Passing `undefined` restores the
 * default (`~/.openclaw/state/pending-reset/`). Production callers must not
 * use this.
 */
export function __setPendingResetDirForTest(dir: string | undefined): void {
  testDirOverride = dir;
}

/** Resolve the directory where pending-reset flags are stored. */
export function pendingResetDir(): string {
  if (testDirOverride) {
    return testDirOverride;
  }
  const home = process.env.OPENCLAW_HOME_OVERRIDE?.trim() || os.homedir();
  return path.join(home, ".openclaw", "state", "pending-reset");
}

/**
 * Map a `sessionKey` to its flag file path. Session keys can contain any
 * ASCII character (colons, slashes, dots), so we URI-encode to produce a
 * safe single-segment filename. Keep this synced with
 * `extensions/sleep-reset/src/flag.ts`.
 */
export function pendingResetFlagPath(sessionKey: string): string {
  return path.join(pendingResetDir(), `${encodeURIComponent(sessionKey)}.json`);
}

/**
 * Consume the flag (read + delete) for the given session key. Returns the
 * parsed payload if one existed, or `undefined` when there was no flag.
 * Errors during read or delete are swallowed and logged at the call site —
 * a malformed flag must not block a user turn.
 */
export function consumePendingResetFlag(sessionKey: string): PendingResetFlagPayload | undefined {
  const flagPath = pendingResetFlagPath(sessionKey);
  let raw: string;
  try {
    raw = fs.readFileSync(flagPath, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException | undefined)?.code;
    if (code === "ENOENT") {
      return undefined;
    }
    // Any other read error: best-effort delete so the stale flag does not
    // re-trigger on every turn.
    try {
      fs.unlinkSync(flagPath);
    } catch {
      // ignore
    }
    return undefined;
  }
  // Always delete after a successful read, even if parsing fails later.
  try {
    fs.unlinkSync(flagPath);
  } catch {
    // ignore
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return {};
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as PendingResetFlagPayload;
    }
    return {};
  } catch {
    return {};
  }
}

/**
 * Write (or overwrite) the flag file for the given session key. Intended for
 * tests and core-owned callers. Plugins living in `extensions/**` should
 * duplicate this with the path convention documented above rather than
 * importing from `src/**`.
 */
export function writePendingResetFlag(
  sessionKey: string,
  payload: PendingResetFlagPayload = {},
): void {
  const flagPath = pendingResetFlagPath(sessionKey);
  fs.mkdirSync(path.dirname(flagPath), { recursive: true });
  const body: PendingResetFlagPayload = {
    requestedAt: Date.now(),
    ...payload,
  };
  fs.writeFileSync(flagPath, JSON.stringify(body));
}

/** Delete the flag without consuming it. Mostly for tests / plugin teardown. */
export function clearPendingResetFlag(sessionKey: string): void {
  try {
    fs.unlinkSync(pendingResetFlagPath(sessionKey));
  } catch {
    // ignore
  }
}
