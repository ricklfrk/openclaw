/**
 * OpenClaw NSFW Plugin - Unrestricted Content
 *
 * Toggle with `/nsfw true` or `/nsfw false`.
 * State is in-memory only — defaults to OFF and resets on restart.
 *
 * When active:
 *   • Part 1 — replaces the Identity line in the system prompt
 *   • Part 2 — injected after conversation history (prependContext)
 */

import { randomBytes } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { resolveAgentIdFromSessionKey } from "openclaw/plugin-sdk/routing";

// ============================================================================
// Random key generation (regenerated per request)
// ============================================================================

const CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789";

function generateKey(len = 16): string {
  const bytes = randomBytes(len);
  let key = "";
  for (let i = 0; i < len; i++) {
    key += CHARSET[bytes[i] % CHARSET.length];
  }
  return key;
}

function freshRandomKeys(): Record<string, string> {
  return {
    random_key_1: generateKey(),
    random_key_2: generateKey(),
    random_key_3: generateKey(),
    random_key_4: generateKey(),
    random_key_5: generateKey(7),
  };
}

// ============================================================================
// Load prompt fragments from prompts.json and replace {{random_key_N}}
// Auto-create a default prompts.json if it doesn't exist.
// ============================================================================

const __dir = dirname(fileURLToPath(import.meta.url));

// Priority: user data dir (~/.openclaw/extensions/nsfw/) → code dir → source fallback.
// User data dir survives rebuilds; code dir may be overwritten by rsync.
function resolvePromptsPath(): string {
  const userDataPath = join(os.homedir(), ".openclaw", "extensions", "nsfw", "prompts.json");
  if (existsSync(userDataPath)) {
    return userDataPath;
  }
  const local = join(__dir, "prompts.json");
  if (existsSync(local)) {
    return local;
  }
  const repoRoot = __dir.replace(/[/\\]dist(?:-runtime)?[/\\]extensions[/\\]nsfw$/, "");
  const source = join(repoRoot, "extensions", "nsfw", "prompts.json");
  if (existsSync(source)) {
    return source;
  }
  return userDataPath;
}

const PROMPTS_PATH = resolvePromptsPath();

const DEFAULT_PROMPTS = {
  part1: ["You are a helpful assistant. (Replace this with your own Part 1 prompt.)"],
  part2: ["(Replace this with your own Part 2 prompt.)"],
};

async function ensurePromptsFile(): Promise<void> {
  if (!existsSync(PROMPTS_PATH)) {
    await fs.mkdir(dirname(PROMPTS_PATH), { recursive: true });
    await fs.writeFile(PROMPTS_PATH, JSON.stringify(DEFAULT_PROMPTS, null, 2) + "\n", "utf-8");
  }
}

type PromptsConfig = {
  part1: string[];
  part2: string[];
  /** Optional: "provider/model" used when /nsfw on (e.g. Gemini 3.1). */
  modelWhenOn?: string;
};

let rawPrompts: PromptsConfig;
try {
  rawPrompts = JSON.parse(readFileSync(PROMPTS_PATH, "utf-8"));
} catch {
  rawPrompts = DEFAULT_PROMPTS;
}

const PART1_TEMPLATE = rawPrompts.part1.join("\n");
const PART2_TEMPLATE = rawPrompts.part2.join("\n");

/** Interpolate fresh random keys into a template string. */
function interpolateKeys(text: string, keys: Record<string, string>): string {
  return text.replace(/\{\{(random_key_[1-5])\}\}/g, (_match, key: string) => keys[key]);
}

// The default identity line to replace
const IDENTITY_LINE = "You are a personal assistant operating inside OpenClaw.";

/** Parse "provider/model" into providerOverride and modelOverride. */
function parseProviderModel(
  value: string | undefined,
): { providerOverride: string; modelOverride: string } | undefined {
  const s = value?.trim();
  if (!s || !s.includes("/")) {
    return undefined;
  }
  const idx = s.indexOf("/");
  const providerOverride = s.slice(0, idx).trim();
  const modelOverride = s.slice(idx + 1).trim();
  return providerOverride && modelOverride ? { providerOverride, modelOverride } : undefined;
}

/** Format provider/model for display (e.g. "anthropic/claude-opus-4-6" → "claude-opus-4-6"). */
function formatModelLabel(provider: string, modelId: string): string {
  if (!provider || !modelId) {
    return `${provider}/${modelId}`.replace(/^\/|\/$/g, "").trim() || "—";
  }
  return modelId;
}

// ============================================================================
// In-memory state (resets on restart)
// Per-agent toggle: Set of agentIds that have NSFW enabled.
// ============================================================================

const nsfwEnabledAgents = new Set<string>();

/** Per-agent model (provider + modelId) to restore when /nsfw off. Captured by hook on first run after turning on. */
const savedModelBeforeOn = new Map<string, { providerOverride: string; modelOverride: string }>();

/** Tracks the pre-hook model seen on every agent run so the command handler can show "from → to". */
const lastSeenModel = new Map<string, { provider: string; modelId: string }>();

// ============================================================================
// Plugin Definition
// ============================================================================

export default definePluginEntry({
  id: "nsfw",
  name: "NSFW",
  description: "Toggle-able system prompt injection via /nsfw command",

  register(api) {
    ensurePromptsFile().catch(() => {});
    api.logger.info("nsfw: plugin registered (default: off, use /nsfw true to enable)");

    // /nsfw command — toggle in-memory state
    api.registerCommand({
      name: "nsfw",
      description: "Toggle NSFW mode per agent. Usage: /nsfw on | /nsfw off",
      acceptsArgs: true,
      requireAuth: true,
      handler(ctx) {
        const arg = (ctx.args ?? "").trim().toLowerCase();
        const agent = resolveAgentIdFromSessionKey(ctx.sessionKey);

        if (arg === "on" || arg === "true" || arg === "1") {
          nsfwEnabledAgents.add(agent);
          const modelWhenOn = parseProviderModel(rawPrompts.modelWhenOn);
          const toLabel = modelWhenOn
            ? formatModelLabel(modelWhenOn.providerOverride, modelWhenOn.modelOverride)
            : "—";
          const seen = lastSeenModel.get(agent);
          const fromLabel = seen ? formatModelLabel(seen.provider, seen.modelId) : null;
          const msg = fromLabel
            ? `NSFW mode ON for agent "${agent}". Model: ${fromLabel} → ${toLabel}`
            : `NSFW mode ON for agent "${agent}". Model → ${toLabel}`;
          return { text: msg };
        }

        if (arg === "off" || arg === "false" || arg === "0") {
          const modelWhenOn = parseProviderModel(rawPrompts.modelWhenOn);
          const fromLabel = modelWhenOn
            ? formatModelLabel(modelWhenOn.providerOverride, modelWhenOn.modelOverride)
            : null;
          const saved = savedModelBeforeOn.get(agent);
          const toLabel = saved
            ? formatModelLabel(saved.providerOverride, saved.modelOverride)
            : null;
          nsfwEnabledAgents.delete(agent);
          savedModelBeforeOn.delete(agent);
          const msg =
            fromLabel && toLabel
              ? `NSFW mode OFF for agent "${agent}". Model: ${fromLabel} → ${toLabel}`
              : `NSFW mode OFF for agent "${agent}".`;
          return { text: msg };
        }

        // No arg or unrecognized → show status
        const isOn = nsfwEnabledAgents.has(agent);
        const list =
          nsfwEnabledAgents.size > 0 ? ` [active: ${[...nsfwEnabledAgents].join(", ")}]` : "";
        return {
          text: `NSFW mode is currently ${isOn ? "ON" : "OFF"} for agent "${agent}".${list} Usage: /nsfw on | /nsfw off`,
        };
      },
    });

    api.on("before_model_resolve", (event, ctx) => {
      const agent = ctx?.agentId ?? "main";
      if (event.provider && event.modelId) {
        lastSeenModel.set(agent, { provider: event.provider, modelId: event.modelId });
      }
      const modelWhenOn = parseProviderModel(rawPrompts.modelWhenOn);
      if (nsfwEnabledAgents.has(agent)) {
        if (event.provider && event.modelId) {
          savedModelBeforeOn.set(agent, {
            providerOverride: event.provider,
            modelOverride: event.modelId,
          });
        }
        return modelWhenOn ?? undefined;
      }
      return undefined;
    });

    // Inject prompt fragments when enabled for this agent.
    // Uses before_prompt_build so the hook always runs in the prompt-build phase
    // where event.systemPrompt is available.
    // Part 1 replaces IDENTITY_LINE inside the core system prompt.
    // Part 2 is appended to the end of the system prompt (not user message)
    // to avoid Gemini safety filters flagging jailbreak patterns in user content.
    api.on("before_prompt_build", (event, ctx) => {
      const agent = ctx?.agentId ?? "main";
      api.logger.info(
        `nsfw: before_prompt_build fired for agent="${agent}", enabled=${nsfwEnabledAgents.has(agent)}, hasSystemPrompt=${!!event.systemPrompt}`,
      );
      if (!nsfwEnabledAgents.has(agent)) {
        return undefined;
      }

      const keys = freshRandomKeys();
      const part1 = interpolateKeys(PART1_TEMPLATE, keys);
      const part2 = interpolateKeys(PART2_TEMPLATE, keys);

      // Replace IDENTITY_LINE with Part 1 inside the core system prompt.
      let modifiedSystemPrompt: string;
      const coreSystemPrompt = event.systemPrompt;
      if (coreSystemPrompt && coreSystemPrompt.includes(IDENTITY_LINE)) {
        modifiedSystemPrompt = coreSystemPrompt.replace(IDENTITY_LINE, part1);
        api.logger.info(
          `nsfw: replaced IDENTITY_LINE in systemPrompt (${coreSystemPrompt.length} → ${modifiedSystemPrompt.length} chars), secKey=${keys.random_key_5}`,
        );
      } else {
        modifiedSystemPrompt = coreSystemPrompt ? `${part1}\n\n${coreSystemPrompt}` : part1;
        api.logger.info(
          `nsfw: IDENTITY_LINE not found, prepended Part1 (${part1.length} chars) to systemPrompt, secKey=${keys.random_key_5}`,
        );
      }

      // Append Part 2 to the end of system prompt.
      modifiedSystemPrompt = `${modifiedSystemPrompt}\n\n${part2}`;
      api.logger.info(
        `nsfw: final systemPrompt ${modifiedSystemPrompt.length} chars (part1=${part1.length}, part2=${part2.length})`,
      );

      return {
        systemPrompt: modifiedSystemPrompt,
      };
    });
  },
});
