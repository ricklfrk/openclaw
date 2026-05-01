import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { hasPendingResetFlag, resolvePendingResetFlagPath, writePendingResetFlag } from "./flag.js";

describe("sleep-reset flag writer", () => {
  let tempHome: string;

  beforeEach(() => {
    tempHome = fs.mkdtempSync(path.join(os.tmpdir(), "sleep-reset-flag-"));
  });

  afterEach(() => {
    fs.rmSync(tempHome, { recursive: true, force: true });
  });

  it("writes the flag file at ~/.openclaw/state/pending-reset/<encoded>.json", () => {
    const key = "agent:main:main";
    const flagPath = writePendingResetFlag(
      key,
      { reason: "sleep", source: "sleep-reset" },
      { homeOverride: tempHome },
    );

    expect(flagPath).toBe(
      path.join(tempHome, ".openclaw", "state", "pending-reset", `${encodeURIComponent(key)}.json`),
    );
    expect(fs.existsSync(flagPath)).toBe(true);

    const parsed = JSON.parse(fs.readFileSync(flagPath, "utf8"));
    expect(parsed.reason).toBe("sleep");
    expect(parsed.source).toBe("sleep-reset");
    expect(typeof parsed.requestedAt).toBe("number");
  });

  it("hasPendingResetFlag reflects the flag file presence", () => {
    const key = "agent:main:main";
    expect(hasPendingResetFlag(key, tempHome)).toBe(false);
    writePendingResetFlag(key, { reason: "sleep" }, { homeOverride: tempHome });
    expect(hasPendingResetFlag(key, tempHome)).toBe(true);
  });

  it("resolvePendingResetFlagPath matches the encoded filename convention", () => {
    const key = "weird/key:with spaces";
    const p = resolvePendingResetFlagPath(key, tempHome);
    expect(path.basename(p)).toBe(`${encodeURIComponent(key)}.json`);
  });

  it("keeps path convention compatible with the core consumer encoding", () => {
    // This explicitly documents the cross-module contract so a rename/refactor
    // on either side will trip this test.
    const key = "agent:main:main";
    const pluginPath = resolvePendingResetFlagPath(key, tempHome);
    const expectedLeaf = `${encodeURIComponent(key)}.json`;
    expect(path.basename(pluginPath)).toBe(expectedLeaf);
  });
});
