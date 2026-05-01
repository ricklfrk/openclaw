import { describe, expect, it } from "vitest";
import { buildExtractionPrompt } from "./extraction-prompts.js";

describe("buildExtractionPrompt — solo mode", () => {
  const prompt = buildExtractionPrompt("user: 你好\nassistant: 你好！", "Alice", "Claw");

  it("uses actual names in conversation roles, not generic pronouns", () => {
    expect(prompt).toContain("## Conversation Roles");
    expect(prompt).toContain("`user:` messages are from **Alice**");
    expect(prompt).toContain("`assistant:` messages are from **Claw**");
  });

  it("requires correct attribution using actual names", () => {
    expect(prompt).toContain("correct attribution");
    expect(prompt).toContain('What Alice said/did/prefers → write "Alice ..."');
    expect(prompt).toContain('What Claw said/promised/decided → write "Claw ..."');
  });

  it("explicitly forbids generic role pronouns", () => {
    expect(prompt).toContain("Never use generic pronouns");
    expect(prompt).toContain("用戶");
    expect(prompt).toContain("助手");
  });

  it("forbids third-person narration", () => {
    expect(prompt).toContain("Never write third-person narration");
  });

  it("contains a language rule requiring same-language memories", () => {
    expect(prompt).toContain("## Language Rule");
    expect(prompt).toContain("same language the conversation actually uses");
  });

  it("includes the conversation text", () => {
    expect(prompt).toContain("user: 你好");
    expect(prompt).toContain("assistant: 你好！");
  });

  it("falls back to 'Assistant' when no assistantName is provided", () => {
    const fallback = buildExtractionPrompt("user: hi", "Bob");
    expect(fallback).toContain("`assistant:` messages are from **Assistant**");
  });

  it("uses user name in abstract example", () => {
    expect(prompt).toContain("Alice went to");
  });

  it("does NOT include group-chat mode marker in solo mode", () => {
    expect(prompt).not.toContain("Mode: Group Chat");
    expect(prompt).not.toContain("Conversation Roles (Group Chat)");
  });
});

describe("buildExtractionPrompt — multi-user group chat mode", () => {
  const prompt = buildExtractionPrompt("user: 大家好\nassistant: 你好！", "User", "Mea", {
    multiUser: true,
  });

  it("marks the prompt as group chat mode", () => {
    expect(prompt).toContain("Mode: Group Chat");
    expect(prompt).toContain("## Conversation Roles (Group Chat)");
  });

  it("tells the LLM user messages may come from multiple humans", () => {
    expect(prompt).toContain("**multiple different humans**");
    expect(prompt).toContain("identify the actual speaker");
  });

  it("keeps a single assistant identity", () => {
    expect(prompt).toContain("all from **Mea**");
  });

  it("forbids collapsing speakers and generic pronouns", () => {
    expect(prompt).toContain("do NOT collapse them into one person");
    expect(prompt).toContain("Never use generic pronouns");
  });

  it("requires skipping unattributable lines instead of guessing", () => {
    expect(prompt).toContain("SKIP it rather than guessing");
  });

  it("requires putting speaker names in entity_tags", () => {
    expect(prompt).toContain("entity_tags");
  });

  it("does not hard-code a single user name in the abstract example", () => {
    // Solo-mode would have "User went to" or "ricklf went to"; multi-user
    // uses a placeholder name like "Alice" to illustrate attribution.
    expect(prompt).toContain("2026-");
    expect(prompt).not.toMatch(/abstract.*\bUser went to/);
  });

  it("includes the conversation text", () => {
    expect(prompt).toContain("user: 大家好");
    expect(prompt).toContain("assistant: 你好！");
  });
});
