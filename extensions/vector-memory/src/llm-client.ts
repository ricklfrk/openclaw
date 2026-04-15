/**
 * LLM Client for memory extraction and dedup decisions.
 * Simplified from memory-lancedb-pro: API-key mode only.
 */

import OpenAI from "openai";

export interface LlmClientConfig {
  apiKey?: string;
  model: string;
  baseURL?: string;
  timeoutMs?: number;
  log?: (msg: string) => void;
}

export interface LlmClient {
  completeJson<T>(prompt: string, label?: string): Promise<T | null>;
  getLastError(): string | null;
}

function extractJsonFromResponse(text: string): string | null {
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }

  const firstBrace = text.indexOf("{");
  if (firstBrace === -1) {
    return null;
  }

  let depth = 0;
  let lastBrace = -1;
  for (let i = firstBrace; i < text.length; i++) {
    if (text[i] === "{") {
      depth++;
    } else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        lastBrace = i;
        break;
      }
    }
  }

  if (lastBrace === -1) {
    return null;
  }
  return text.substring(firstBrace, lastBrace + 1);
}

function previewText(value: string, maxLen = 200): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLen) {
    return normalized;
  }
  return `${normalized.slice(0, maxLen - 3)}...`;
}

function nextNonWhitespaceChar(text: string, start: number): string | undefined {
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (!/\s/.test(ch)) {
      return ch;
    }
  }
  return undefined;
}

function repairCommonJson(text: string): string {
  let result = "";
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (escaped) {
      result += ch;
      escaped = false;
      continue;
    }

    if (inString) {
      if (ch === "\\") {
        result += ch;
        escaped = true;
        continue;
      }
      if (ch === '"') {
        const nextCh = nextNonWhitespaceChar(text, i + 1);
        if (
          nextCh === undefined ||
          nextCh === "," ||
          nextCh === "}" ||
          nextCh === "]" ||
          nextCh === ":"
        ) {
          result += ch;
          inString = false;
        } else {
          result += '\\"';
        }
        continue;
      }
      if (ch === "\n") {
        result += "\\n";
        continue;
      }
      if (ch === "\r") {
        result += "\\r";
        continue;
      }
      if (ch === "\t") {
        result += "\\t";
        continue;
      }
      result += ch;
      continue;
    }

    if (ch === '"') {
      result += ch;
      inString = true;
      continue;
    }

    if (ch === ",") {
      const nextCh = nextNonWhitespaceChar(text, i + 1);
      if (nextCh === "}" || nextCh === "]") {
        continue;
      }
    }

    result += ch;
  }

  return result;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

export function createLlmClient(config: LlmClientConfig): LlmClient {
  const log = config.log ?? (() => {});

  if (!config.apiKey) {
    throw new Error("vector-memory: LLM requires apiKey (llm.apiKey or embedding.apiKey)");
  }

  const resolvedKey = resolveEnvVars(config.apiKey);
  const client = new OpenAI({
    apiKey: resolvedKey,
    baseURL: config.baseURL,
    timeout: config.timeoutMs ?? 30000,
  });
  let lastError: string | null = null;

  return {
    async completeJson<T>(prompt: string, label = "generic"): Promise<T | null> {
      lastError = null;
      try {
        const response = await client.chat.completions.create({
          model: config.model,
          messages: [
            {
              role: "system",
              content:
                "You are a memory extraction assistant. Always respond with valid JSON only.",
            },
            { role: "user", content: prompt },
          ],
          temperature: 0.1,
        });

        const raw = response.choices?.[0]?.message?.content;
        if (!raw || typeof raw !== "string") {
          lastError = `vector-memory: llm [${label}] empty/non-string response from ${config.model}`;
          log(lastError);
          return null;
        }

        const jsonStr = extractJsonFromResponse(raw);
        if (!jsonStr) {
          lastError = `vector-memory: llm [${label}] no JSON found (chars=${raw.length}, preview=${JSON.stringify(previewText(raw))})`;
          log(lastError);
          return null;
        }

        try {
          return JSON.parse(jsonStr) as T;
        } catch (err) {
          const repaired = repairCommonJson(jsonStr);
          if (repaired !== jsonStr) {
            try {
              return JSON.parse(repaired) as T;
            } catch {
              // fall through
            }
          }
          lastError = `vector-memory: llm [${label}] JSON.parse failed: ${err instanceof Error ? err.message : String(err)}`;
          log(lastError);
          return null;
        }
      } catch (err) {
        lastError = `vector-memory: llm [${label}] request failed: ${err instanceof Error ? err.message : String(err)}`;
        log(lastError);
        return null;
      }
    },
    getLastError(): string | null {
      return lastError;
    },
  };
}
