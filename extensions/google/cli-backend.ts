import type { CliBackendPlugin } from "openclaw/plugin-sdk/cli-backend";
import {
  CLI_FRESH_WATCHDOG_DEFAULTS,
  CLI_RESUME_WATCHDOG_DEFAULTS,
} from "openclaw/plugin-sdk/cli-backend";

const GEMINI_MODEL_ALIASES: Record<string, string> = {
  pro: "gemini-3.1-pro-preview",
  flash: "gemini-3.1-flash-preview",
  "flash-lite": "gemini-3.1-flash-lite-preview",
};
const GEMINI_CLI_DEFAULT_MODEL_REF = "google-gemini-cli/gemini-3-flash-preview";

export function buildGoogleGeminiCliBackend(): CliBackendPlugin {
  return {
    id: "google-gemini-cli",
    liveTest: {
      defaultModelRef: GEMINI_CLI_DEFAULT_MODEL_REF,
      defaultImageProbe: true,
      defaultMcpProbe: true,
      docker: {
        npmPackage: "@google/gemini-cli",
        binaryName: "gemini",
      },
    },
    bundleMcp: true,
    bundleMcpMode: "gemini-system-settings",
    nativeToolMode: "always-on",
    config: {
      command: "gemini",
      // `--skip-trust` was removed in @google/gemini-cli ≥ 0.38.x. Yargs
      // rejects the flag with `Unknown arguments: skip-trust, skipTrust` and
      // exits non-zero, so every call would otherwise fail. Trust prompts are
      // now handled via `~/.gemini/trustedFolders.json` / Policy Engine.
      // See `https://github.com/openclaw/openclaw/issues/74749`.
      args: ["--output-format", "json", "--prompt", "{prompt}"],
      resumeArgs: ["--resume", "{sessionId}", "--output-format", "json", "--prompt", "{prompt}"],
      output: "json",
      input: "arg",
      imageArg: "@",
      imagePathScope: "workspace",
      modelArg: "--model",
      modelAliases: GEMINI_MODEL_ALIASES,
      sessionMode: "existing",
      sessionIdFields: ["session_id", "sessionId"],
      // `gemini` reads its system prompt from the file pointed to by
      // `GEMINI_SYSTEM_MD` (CLI ≥ 0.34.x). The CLI exposes no
      // `--system-prompt` arg, so without this env hook any
      // `before_prompt_build` hook returning `{ systemPrompt }` would be
      // silently dropped by `cli-runner/helpers.ts:resolveSystemPromptUsage`.
      systemPromptEnvVar: "GEMINI_SYSTEM_MD",
      reliability: {
        watchdog: {
          fresh: { ...CLI_FRESH_WATCHDOG_DEFAULTS },
          resume: { ...CLI_RESUME_WATCHDOG_DEFAULTS },
        },
      },
      serialize: true,
    },
  };
}
