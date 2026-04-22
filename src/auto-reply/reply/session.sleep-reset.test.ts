import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import {
  __setPendingResetDirForTest,
  pendingResetFlagPath,
  writePendingResetFlag,
} from "../../config/sessions/pending-reset-flag.js";
import { saveSessionStore } from "../../config/sessions/store.js";
import type { SessionEntry } from "../../config/sessions/types.js";
import type { MsgContext } from "../templating.js";
import { initSessionState } from "./session.js";

describe("initSessionState — lastUserUpdatedAt + pending-reset flag", () => {
  let tempDir: string;
  let storePath: string;
  let flagDir: string;

  beforeEach(async () => {
    tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "openclaw-sleep-reset-"));
    storePath = path.join(tempDir, "sessions.json");
    flagDir = fs.mkdtempSync(path.join(os.tmpdir(), "pending-reset-"));
    __setPendingResetDirForTest(flagDir);
  });

  afterEach(async () => {
    __setPendingResetDirForTest(undefined);
    await fsp.rm(tempDir, { recursive: true, force: true });
    await fsp.rm(flagDir, { recursive: true, force: true });
  });

  const createBaseConfig = (): OpenClawConfig => ({
    agents: {
      defaults: { workspace: tempDir },
      list: [{ id: "main", workspace: tempDir }],
    },
    session: {
      store: storePath,
      // Use idle reset so tests can choose staleness independent of local time.
      reset: { mode: "idle", idleMinutes: 60 },
    },
    channels: {},
    gateway: {
      port: 18789,
      mode: "local",
      bind: "loopback",
      auth: { mode: "token", token: "test" },
    },
    plugins: { entries: {} },
  });

  const createBaseCtx = (overrides?: Partial<MsgContext>): MsgContext => ({
    Body: "hello",
    From: "user123",
    To: "bot123",
    SessionKey: "main:user123",
    Provider: "telegram",
    Surface: "telegram",
    ChatType: "direct",
    CommandAuthorized: true,
    ...overrides,
  });

  it("updates lastUserUpdatedAt on real user turns", async () => {
    await saveSessionStore(storePath, {
      "main:user123": {
        sessionId: "sid-A",
        updatedAt: Date.now() - 1_000,
        lastUserUpdatedAt: Date.now() - 60_000,
        systemSent: true,
      },
    });

    const before = Date.now();
    const result = await initSessionState({
      ctx: createBaseCtx(),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });
    const after = Date.now();

    expect(result.sessionEntry.lastUserUpdatedAt).toBeDefined();
    expect(result.sessionEntry.lastUserUpdatedAt!).toBeGreaterThanOrEqual(before);
    expect(result.sessionEntry.lastUserUpdatedAt!).toBeLessThanOrEqual(after);
  });

  it("does NOT update lastUserUpdatedAt on heartbeat (system event)", async () => {
    const preservedUserAt = Date.now() - 60_000;
    await saveSessionStore(storePath, {
      "main:user123": {
        sessionId: "sid-B",
        updatedAt: Date.now() - 1_000,
        lastUserUpdatedAt: preservedUserAt,
        systemSent: true,
      },
    });

    const result = await initSessionState({
      ctx: createBaseCtx({ Provider: "heartbeat", Body: "HEARTBEAT_OK" }),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });

    expect(result.sessionEntry.lastUserUpdatedAt).toBe(preservedUserAt);
    // updatedAt should still advance on heartbeat.
    expect(result.sessionEntry.updatedAt).toBeGreaterThanOrEqual(preservedUserAt);
  });

  it("leaves lastUserUpdatedAt unset on a new session created by a system event", async () => {
    // No baseEntry; cron-event ticks first (e.g. plugin cron running before
    // the user ever talks). lastUserUpdatedAt should stay undefined so the
    // sleep-reset plugin does not mistake the cron turn for real user
    // activity.
    await saveSessionStore(storePath, {});

    const result = await initSessionState({
      ctx: createBaseCtx({ Provider: "cron-event", Body: "cron job" }),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });

    expect(result.sessionEntry.lastUserUpdatedAt).toBeUndefined();
  });

  it("pending-reset flag forces session reset on the next real user turn", async () => {
    const freshSessionEntry: SessionEntry = {
      sessionId: "sid-will-roll",
      updatedAt: Date.now(),
      lastUserUpdatedAt: Date.now(),
      systemSent: true,
    };
    await saveSessionStore(storePath, { "main:user123": freshSessionEntry });

    writePendingResetFlag("main:user123", { reason: "sleep", source: "sleep-reset" });

    const result = await initSessionState({
      ctx: createBaseCtx(),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });

    expect(result.isNewSession).toBe(true);
    expect(result.sessionId).not.toBe("sid-will-roll");
    // Flag must be consumed (one-shot).
    expect(fs.existsSync(pendingResetFlagPath("main:user123"))).toBe(false);
  });

  it("heartbeat does NOT consume the pending-reset flag", async () => {
    const freshSessionEntry: SessionEntry = {
      sessionId: "sid-stay",
      updatedAt: Date.now(),
      lastUserUpdatedAt: Date.now(),
      systemSent: true,
    };
    await saveSessionStore(storePath, { "main:user123": freshSessionEntry });

    writePendingResetFlag("main:user123", { reason: "sleep" });

    const result = await initSessionState({
      ctx: createBaseCtx({ Provider: "heartbeat", Body: "HEARTBEAT_OK" }),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });

    expect(result.isNewSession).toBe(false);
    expect(result.sessionId).toBe("sid-stay");
    // Flag must still be present so the next real user turn can consume it.
    expect(fs.existsSync(pendingResetFlagPath("main:user123"))).toBe(true);
  });

  it("no pending flag means freshness still decides (fresh session stays)", async () => {
    const sessionEntry: SessionEntry = {
      sessionId: "sid-fresh",
      updatedAt: Date.now(),
      lastUserUpdatedAt: Date.now(),
      systemSent: true,
    };
    await saveSessionStore(storePath, { "main:user123": sessionEntry });

    const result = await initSessionState({
      ctx: createBaseCtx(),
      cfg: createBaseConfig(),
      commandAuthorized: true,
    });

    expect(result.isNewSession).toBe(false);
    expect(result.sessionId).toBe("sid-fresh");
  });
});
