/**
 * OpenClaw Regex Replace Plugin
 *
 * Applies regex-based find-and-replace rules to:
 *   1. Outgoing messages (message_sending) — what the user sees on their channel.
 *   2. Session transcript writes (before_message_write) — so the LLM's future
 *      context also reflects the replacements (both user input and assistant replies).
 *
 * Rules are defined in regex.json next to this file.
 *
 * Each rule has:
 *   pattern     — regex pattern string (no surrounding slashes)
 *   flags       — regex flags, e.g. "gm", "gi" (default: "g")
 *   replacement — the replacement string (default: "")
 *
 * The plugin hot-reloads regex.json on every message so you can edit
 * rules without restarting the gateway.
 */

import { existsSync, readFileSync } from "node:fs";
import os from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const __dir = dirname(fileURLToPath(import.meta.url));

// Priority: user data dir (~/.openclaw/extensions/regex-replace/) → code dir → source fallback.
// User data dir survives rebuilds; code dir may be overwritten by rsync.
function resolveRulesPath(): string {
  const userDataPath = join(os.homedir(), ".openclaw", "extensions", "regex-replace", "regex.json");
  if (existsSync(userDataPath)) {
    return userDataPath;
  }
  const local = join(__dir, "regex.json");
  if (existsSync(local)) {
    return local;
  }
  const repoRoot = __dir.replace(/[/\\]dist(?:-runtime)?[/\\]extensions[/\\]regex-replace$/, "");
  const source = join(repoRoot, "extensions", "regex-replace", "regex.json");
  if (existsSync(source)) {
    return source;
  }
  return userDataPath;
}

const RULES_PATH = resolveRulesPath();

export type Rule = {
  pattern: string;
  flags?: string;
  replacement?: string;
};

type ContentBlock = {
  type: string;
  text?: string;
  arguments?: unknown;
  [key: string]: unknown;
};

/** Load rules from disk (hot-reload on each call). */
function loadRules(logger: { warn: (msg: string) => void }): Rule[] {
  try {
    const raw = readFileSync(RULES_PATH, "utf-8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      logger.warn("regex-replace: regex.json must be a JSON array; skipping");
      return [];
    }
    return parsed;
  } catch (err) {
    logger.warn(`regex-replace: failed to load regex.json — ${String(err)}`);
    return [];
  }
}

/** Apply all rules to the input string. */
export function applyRules(
  input: string,
  rules: Rule[],
  logger: { warn: (msg: string) => void },
): string {
  let result = input;
  for (const rule of rules) {
    if (!rule.pattern) {
      continue;
    }
    try {
      const re = new RegExp(rule.pattern, rule.flags ?? "g");
      result = result.replace(re, rule.replacement ?? "");
    } catch (err) {
      logger.warn(`regex-replace: invalid pattern "${rule.pattern}" — ${String(err)}`);
    }
  }
  return result;
}

/**
 * Walk the top-level string fields of a toolCall block's `arguments` object
 * and apply regex rules to each. Only the top level is considered: rules here
 * are intended for user-facing prose (e.g. `arguments.message` on the built-in
 * `message` tool, `arguments.text` on other text-bearing tools), not for deep
 * structural fields. Non-string fields are passed through unchanged.
 *
 * Returns the rewritten arguments object if any string field changed,
 * otherwise `null` so the caller can keep the original block reference.
 */
function applyRulesToToolCallArguments(
  args: Record<string, unknown>,
  rules: Rule[],
  logger: { warn: (msg: string) => void },
): Record<string, unknown> | null {
  let touched = false;
  const next: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (typeof value === "string") {
      const transformed = applyRules(value, rules, logger);
      if (transformed !== value) {
        next[key] = transformed;
        touched = true;
        continue;
      }
    }
    next[key] = value;
  }
  return touched ? next : null;
}

/** Apply regex rules to a message's content, returning a new message if changed. */
export function applyRulesToMessage(
  msg: Record<string, unknown>,
  rules: Rule[],
  logger: { warn: (msg: string) => void },
): Record<string, unknown> | null {
  const { content } = msg;
  if (typeof content === "string") {
    const transformed = applyRules(content, rules, logger);
    if (transformed === content) {
      return null;
    }
    return { ...msg, content: transformed };
  }
  if (!Array.isArray(content)) {
    return null;
  }

  let touched = false;
  const blocks: ContentBlock[] = [];
  for (const block of content as ContentBlock[]) {
    if (typeof block.text === "string") {
      const transformed = applyRules(block.text, rules, logger);
      if (transformed !== block.text) {
        blocks.push({ ...block, text: transformed });
        touched = true;
        continue;
      }
    }
    // Assistant tool-call blocks carry user-facing prose inside `arguments`
    // (e.g. `message`, `text`). Without walking this field, rules like
    // `<disclaimer>…</disclaimer>` would silently skip every assistant turn
    // that used the `message` tool and the substrings would persist in the
    // session JSONL, where the LLM replays them as in-context training.
    if (
      block.type === "toolCall" &&
      block.arguments &&
      typeof block.arguments === "object" &&
      !Array.isArray(block.arguments)
    ) {
      const replaced = applyRulesToToolCallArguments(
        block.arguments as Record<string, unknown>,
        rules,
        logger,
      );
      if (replaced) {
        blocks.push({ ...block, arguments: replaced });
        touched = true;
        continue;
      }
    }
    blocks.push(block);
  }
  if (!touched) {
    return null;
  }
  return { ...msg, content: blocks };
}

const CONTEXT_ROLES = new Set(["user", "assistant"]);

export default definePluginEntry({
  id: "regex-replace",
  name: "Regex Replace",
  description:
    "Apply regex find-and-replace rules to outgoing messages and session context (edit regex.json to configure)",

  register(api) {
    api.logger.info("regex-replace: plugin registered (rules in regex.json)");

    // Hook 1: transform content before delivery to the channel
    api.on("message_sending", async (event) => {
      if (!event.content) {
        return undefined;
      }

      const rules = loadRules(api.logger);
      if (rules.length === 0) {
        return undefined;
      }

      const transformed = applyRules(event.content, rules, api.logger);

      if (transformed === event.content) {
        return undefined;
      }
      api.logger.info(
        `regex-replace: [sending] replaced (${event.content.length} → ${transformed.length} chars)`,
      );
      return { content: transformed };
    });

    // Hook 2: transform content before it's written to session JSONL,
    // so the LLM sees replaced text in future context.
    api.on("before_message_write", (event) => {
      const msg = event.message as unknown as Record<string, unknown>;
      const role = msg.role as string | undefined;
      if (!role || !CONTEXT_ROLES.has(role)) {
        return undefined;
      }

      const rules = loadRules(api.logger);
      if (rules.length === 0) {
        return undefined;
      }

      const replaced = applyRulesToMessage(msg, rules, api.logger);
      if (!replaced) {
        return undefined;
      }

      api.logger.info(`regex-replace: [session] replaced content in ${role} message`);
      return { message: replaced as unknown as typeof event.message };
    });
  },
});
