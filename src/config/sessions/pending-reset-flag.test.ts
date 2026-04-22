import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  __setPendingResetDirForTest,
  clearPendingResetFlag,
  consumePendingResetFlag,
  pendingResetFlagPath,
  writePendingResetFlag,
} from "./pending-reset-flag.js";

describe("pending-reset-flag", () => {
  let dir: string;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "pending-reset-flag-"));
    __setPendingResetDirForTest(dir);
  });

  afterEach(() => {
    __setPendingResetDirForTest(undefined);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("encodes session keys into a safe single-segment filename", () => {
    const key = "agent:main:main";
    const p = pendingResetFlagPath(key);
    expect(path.dirname(p)).toBe(dir);
    expect(path.basename(p)).toBe(`${encodeURIComponent(key)}.json`);
  });

  it("returns undefined when no flag exists", () => {
    expect(consumePendingResetFlag("agent:main:main")).toBeUndefined();
  });

  it("consumes and removes the flag, returning parsed payload", () => {
    const key = "agent:main:main";
    writePendingResetFlag(key, { reason: "sleep", source: "sleep-reset" });
    expect(fs.existsSync(pendingResetFlagPath(key))).toBe(true);

    const consumed = consumePendingResetFlag(key);
    expect(consumed?.reason).toBe("sleep");
    expect(consumed?.source).toBe("sleep-reset");
    expect(typeof consumed?.requestedAt).toBe("number");
    expect(fs.existsSync(pendingResetFlagPath(key))).toBe(false);

    // Second consume should see no flag.
    expect(consumePendingResetFlag(key)).toBeUndefined();
  });

  it("deletes malformed flag files so they do not re-fire every turn", () => {
    const key = "agent:main:main";
    const flagPath = pendingResetFlagPath(key);
    fs.mkdirSync(path.dirname(flagPath), { recursive: true });
    fs.writeFileSync(flagPath, "not json ");

    const consumed = consumePendingResetFlag(key);
    expect(consumed).toEqual({});
    expect(fs.existsSync(flagPath)).toBe(false);
  });

  it("clearPendingResetFlag tolerates a missing file", () => {
    expect(() => clearPendingResetFlag("nope")).not.toThrow();
  });

  it("isolates flag files by session key", () => {
    writePendingResetFlag("key-a", { reason: "sleep" });
    writePendingResetFlag("key-b", { reason: "sleep" });

    expect(consumePendingResetFlag("key-a")?.reason).toBe("sleep");
    expect(fs.existsSync(pendingResetFlagPath("key-b"))).toBe(true);

    expect(consumePendingResetFlag("key-b")?.reason).toBe("sleep");
    expect(fs.existsSync(pendingResetFlagPath("key-b"))).toBe(false);
  });
});
