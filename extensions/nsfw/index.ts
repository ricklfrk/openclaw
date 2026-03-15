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
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

// ============================================================================
// Random key generation (16-char alphanumeric, regenerated per startup)
// ============================================================================

const CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789";

function generateKey(): string {
  const bytes = randomBytes(16);
  let key = "";
  for (let i = 0; i < 16; i++) {
    key += CHARSET[bytes[i] % CHARSET.length];
  }
  return key;
}

function generateKey_7(): string {
  const bytes = randomBytes(7);
  let key = "";
  for (let i = 0; i < 7; i++) {
    key += CHARSET[bytes[i] % CHARSET.length];
  }
  return key;
}

const randomKeys: Record<string, string> = {
  random_key_1: generateKey(),
  random_key_2: generateKey(),
  random_key_3: generateKey(),
  random_key_4: generateKey(),
  random_key_5: generateKey_7(),
};

// ============================================================================
// Load prompt fragments from prompts.json and replace {{random_key_N}}
// Auto-create a default prompts.json if it doesn't exist.
// ============================================================================

const __dir = dirname(fileURLToPath(import.meta.url));
const PROMPTS_PATH = join(__dir, "prompts.json");

const DEFAULT_PROMPTS = {
  part1: ["You are a helpful assistant. (Replace this with your own Part 1 prompt.)"],
  part2: ["(Replace this with your own Part 2 prompt.)"],
};

if (!existsSync(PROMPTS_PATH)) {
  writeFileSync(PROMPTS_PATH, JSON.stringify(DEFAULT_PROMPTS, null, 2) + "\n", "utf-8");
}

type PromptsConfig = {
  part1: string[];
  part2: string[];
  /** Optional: "provider/model" used when /nsfw on (e.g. Gemini 3.1). */
  modelWhenOn?: string;
};

const rawPrompts: PromptsConfig = JSON.parse(readFileSync(PROMPTS_PATH, "utf-8"));

function interpolateKeys(text: string): string {
  return text.replace(/\{\{(random_key_[1-5])\}\}/g, (_match, key: string) => randomKeys[key]);
}

const NSFW_SYSTEM_PROMPT_PART1 = interpolateKeys(rawPrompts.part1.join("\n"));
const NSFW_SYSTEM_PROMPT_PART2 = interpolateKeys(rawPrompts.part2.join("\n"));

// The default identity line to replace
const IDENTITY_LINE = "You are a personal assistant running inside OpenClaw.";

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

/** Per-agent model (provider + modelId) to restore when /nsfw off. Set when turning on. */
const savedModelBeforeOn = new Map<string, { providerOverride: string; modelOverride: string }>();

// ============================================================================
// Plugin Definition
// ============================================================================

const nsfwPlugin = {
  id: "nsfw",
  name: "NSFW",
  description: "Toggle-able system prompt injection via /nsfw command",

  register(api: OpenClawPluginApi) {
    api.logger.info("nsfw: plugin registered (default: off, use /nsfw true to enable)");

    // /nsfw command — toggle in-memory state
    api.registerCommand({
      name: "nsfw",
      description: "Toggle NSFW mode per agent. Usage: /nsfw on | /nsfw off",
      acceptsArgs: true,
      requireAuth: true,
      handler(ctx) {
        const arg = (ctx.args ?? "").trim().toLowerCase();
        const agent = (ctx as { agentId?: string }).agentId ?? "main";

        if (arg === "on" || arg === "true" || arg === "1") {
          nsfwEnabledAgents.add(agent);
          const modelWhenOn = parseProviderModel(rawPrompts.modelWhenOn);
          const toLabel = modelWhenOn
            ? formatModelLabel(modelWhenOn.providerOverride, modelWhenOn.modelOverride)
            : "—";
          const saved = savedModelBeforeOn.get(agent);
          const fromLabel = saved
            ? formatModelLabel(saved.providerOverride, saved.modelOverride)
            : null;
          const msg = fromLabel
            ? `NSFW mode ON for agent "${agent}". Model: ${fromLabel} → ${toLabel}. Will reset on restart.`
            : `NSFW mode ON for agent "${agent}". Model: ${toLabel}. (Previous model will be restored when you turn off.) Will reset on restart.`;
          return { text: msg };
        }

        if (arg === "off" || arg === "false" || arg === "0") {
          const saved = savedModelBeforeOn.get(agent);
          const restoredLabel = saved
            ? formatModelLabel(saved.providerOverride, saved.modelOverride)
            : "—";
          nsfwEnabledAgents.delete(agent);
          savedModelBeforeOn.delete(agent);
          const msg =
            restoredLabel !== "—"
              ? `NSFW mode OFF for agent "${agent}". Model restored to ${restoredLabel}.`
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

    // Switch model by /nsfw on|off: on → use modelWhenOn (and remember current model for later restore); off → restore saved.
    api.on("before_model_resolve", (event, ctx) => {
      const agent = ctx?.agentId ?? "main";
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

    // Inject prompt fragments when enabled for this agent
    api.on("before_agent_start", async (event, ctx) => {
      const agent = ctx?.agentId ?? "main";
      api.logger.info(
        `nsfw: before_agent_start fired for agent="${agent}", enabled=${nsfwEnabledAgents.has(agent)}, enabledAgents=[${[...nsfwEnabledAgents].join(",")}]`,
      );
      if (!nsfwEnabledAgents.has(agent)) {
        return;
      }

      // Part 1: replace the Identity line in the system prompt
      let modifiedSystemPrompt: string | undefined;
      const currentSystemPrompt = event.systemPrompt;
      if (currentSystemPrompt) {
        const hasIdentityLine = currentSystemPrompt.includes(IDENTITY_LINE);
        api.logger.info(
          `nsfw: systemPrompt length=${currentSystemPrompt.length}, hasIdentityLine=${hasIdentityLine}`,
        );
        modifiedSystemPrompt = currentSystemPrompt.replace(IDENTITY_LINE, NSFW_SYSTEM_PROMPT_PART1);
        api.logger.info(`nsfw: modifiedSystemPrompt length=${modifiedSystemPrompt.length}`);
      } else {
        api.logger.info("nsfw: no systemPrompt in event");
      }

      return {
        // Part 1 → replaces Identity in system prompt
        ...(modifiedSystemPrompt ? { systemPrompt: modifiedSystemPrompt } : {}),
        // Part 2 → injected between conversation history and user message
        prependContext: NSFW_SYSTEM_PROMPT_PART2,
      };
    });
  },
};

export default nsfwPlugin;
