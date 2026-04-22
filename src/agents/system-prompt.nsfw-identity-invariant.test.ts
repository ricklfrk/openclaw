import { describe, expect, it } from "vitest";
import { loadNsfwIdentityLines } from "../../test/helpers/nsfw-identity.js";
import { buildAgentSystemPrompt } from "./system-prompt.js";

/**
 * Invariant: the NSFW plugin inline-replaces the core identity line with its
 * Part 1 jailbreak block. If the core identity line drifts without updating
 * `NSFW_IDENTITY_LINES`, the plugin falls back to prepending Part 1 at the top
 * of the system prompt, which materially changes how Gemini's safety detectors
 * see the prompt and tends to produce silent empty responses or PROHIBITED_CONTENT
 * blocks. A regression of this kind shipped in 2026-04-11 (commit cc5c691f00,
 * "operating" → "running"), which motivates this test.
 *
 * Keep this test in `src/agents/` (not in the nsfw extension) because it
 * asserts a core system-prompt property; the plugin's expected strings are
 * consumed via the shared bundled-plugin public surface helper so this test
 * does not reach into extension internals.
 */
describe("NSFW plugin identity-line invariant", () => {
  const buildWithMode = (promptMode: "full" | "minimal" | "none") =>
    buildAgentSystemPrompt({
      workspaceDir: "/tmp/openclaw",
      promptMode,
    });

  it("core system prompt exposes at least one identity line NSFW recognizes (full mode)", () => {
    const identityLines = loadNsfwIdentityLines();
    expect(identityLines.length).toBeGreaterThan(0);
    const prompt = buildWithMode("full");
    const matched = identityLines.find((line) => prompt.includes(line));
    expect(
      matched,
      `Expected buildAgentSystemPrompt (promptMode=full) to contain one of ` +
        `NSFW_IDENTITY_LINES (from extensions/nsfw). If the core identity line ` +
        `was intentionally changed, update extensions/nsfw/index.ts so its ` +
        `NSFW_IDENTITY_LINES list keeps at least one current match.`,
    ).toBeDefined();
  });

  it("core system prompt exposes at least one identity line NSFW recognizes (none mode)", () => {
    const identityLines = loadNsfwIdentityLines();
    const prompt = buildWithMode("none");
    const matched = identityLines.find((line) => prompt.includes(line));
    expect(
      matched,
      `Expected buildAgentSystemPrompt (promptMode=none) to contain one of ` +
        `NSFW_IDENTITY_LINES. See extensions/nsfw/index.ts.`,
    ).toBeDefined();
  });

  it("core system prompt exposes at least one identity line NSFW recognizes (minimal mode)", () => {
    const identityLines = loadNsfwIdentityLines();
    const prompt = buildWithMode("minimal");
    const matched = identityLines.find((line) => prompt.includes(line));
    expect(
      matched,
      `Expected buildAgentSystemPrompt (promptMode=minimal) to contain one of ` +
        `NSFW_IDENTITY_LINES. See extensions/nsfw/index.ts.`,
    ).toBeDefined();
  });
});
