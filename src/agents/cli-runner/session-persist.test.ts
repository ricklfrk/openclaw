import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { persistCliTurn } from "./session-persist.js";

async function readJsonl(file: string): Promise<Array<Record<string, unknown>>> {
  const raw = await fs.readFile(file, "utf8");
  return raw
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe("persistCliTurn", () => {
  let tmpDir: string;
  let sessionFile: string;

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-cli-persist-"));
    sessionFile = path.join(tmpDir, "session.jsonl");
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("appends user + assistant message pair to session jsonl", async () => {
    await persistCliTurn({
      sessionFile,
      prompt: "hello world",
      assistantText: "hi there",
      provider: "google-gemini-cli",
      model: "gemini-3.1-pro-preview",
      api: "google-gemini-cli",
      startedAt: 1_700_000_000_000,
      finishedAt: 1_700_000_001_000,
    });

    const entries = await readJsonl(sessionFile);
    const messageEntries = entries.filter(
      (entry) => (entry as { type?: string }).type === "message",
    );
    expect(messageEntries).toHaveLength(2);

    const first = messageEntries[0] as { message?: Record<string, unknown> };
    const second = messageEntries[1] as { message?: Record<string, unknown> };
    expect(first.message).toMatchObject({
      role: "user",
      content: "hello world",
      timestamp: 1_700_000_000_000,
    });
    expect(second.message).toMatchObject({
      role: "assistant",
      model: "gemini-3.1-pro-preview",
      provider: "google-gemini-cli",
      api: "google-gemini-cli",
      timestamp: 1_700_000_001_000,
    });
    const assistantContent = (second.message as { content?: unknown }).content;
    expect(assistantContent).toEqual([{ type: "text", text: "hi there" }]);
  });

  it("includes images alongside prompt when provided", async () => {
    await persistCliTurn({
      sessionFile,
      prompt: "caption this",
      images: [{ type: "image", data: "base64data", mimeType: "image/png" }],
      assistantText: "A sunset.",
      provider: "google-gemini-cli",
      model: "gemini-3.1-pro-preview",
      startedAt: 1_700_000_000_000,
    });

    const entries = await readJsonl(sessionFile);
    const user = entries.find(
      (entry) =>
        (entry as { type?: string }).type === "message" &&
        ((entry as { message?: { role?: string } }).message?.role ?? "") === "user",
    ) as { message?: { content?: unknown } } | undefined;
    expect(user?.message?.content).toEqual([
      { type: "text", text: "caption this" },
      { type: "image", data: "base64data", mimeType: "image/png" },
    ]);
  });

  it("skips persistence when assistant text is empty", async () => {
    await persistCliTurn({
      sessionFile,
      prompt: "whatever",
      assistantText: "   ",
      provider: "google-gemini-cli",
      model: "gemini-3.1-pro-preview",
      startedAt: 1_700_000_000_000,
    });

    await expect(fs.stat(sessionFile)).rejects.toThrow();
  });

  it("skips persistence when sessionFile is missing or blank", async () => {
    await persistCliTurn({
      sessionFile: "   ",
      prompt: "hi",
      assistantText: "hello",
      provider: "google-gemini-cli",
      model: "gemini-3.1-pro-preview",
      startedAt: 1_700_000_000_000,
    });
    await persistCliTurn({
      sessionFile: undefined,
      prompt: "hi",
      assistantText: "hello",
      provider: "google-gemini-cli",
      model: "gemini-3.1-pro-preview",
      startedAt: 1_700_000_000_000,
    });
    const dirEntries = await fs.readdir(tmpDir);
    expect(dirEntries).toEqual([]);
  });

  it("swallows errors so a CLI run is never failed by persistence", async () => {
    const badPath = path.join(tmpDir, "no-such-dir", "nested", "session.jsonl");
    await expect(
      persistCliTurn({
        sessionFile: badPath,
        prompt: "hi",
        assistantText: "hello",
        provider: "google-gemini-cli",
        model: "gemini-3.1-pro-preview",
        startedAt: 1_700_000_000_000,
      }),
    ).resolves.toBeUndefined();
  });
});
