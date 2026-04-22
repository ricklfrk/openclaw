import type { AssistantMessage, ImageContent, Usage, UserMessage } from "@mariozechner/pi-ai";
import { SessionManager } from "@mariozechner/pi-coding-agent";
import { prewarmSessionFile } from "../pi-embedded-runner/session-manager-cache.js";
import { repairSessionFileIfNeeded } from "../session-file-repair.js";
import { guardSessionManager } from "../session-tool-result-guard-wrapper.js";
import { cliBackendLog } from "./log.js";

/**
 * Appends user + assistant messages for a CLI-backend turn into the session
 * JSONL, mirroring what the embedded (`pi-agent-core`) loop writes on its own.
 *
 * The CLI backends (`claude-cli`, `google-gemini-cli`, ...) spawn external
 * binaries and bypass `pi-agent-core`, so without this seam the session file
 * never receives those turns. That breaks downstream consumers that rely on
 * the transcript (vector-memory autoCapture, compaction anchor, sleep-reset
 * idle tracking, freshness checks, etc.).
 *
 * We open the shared SessionManager, wrap it with `guardSessionManager` so
 * `before_message_write` plugin hooks fire (NSFW `<think>` stripper,
 * regex-replace cleanup, input-provenance stamping), and append the pair.
 * All failures are swallowed — persistence is best-effort and must never
 * fail a successful CLI run.
 */
export type PersistCliTurnParams = {
  sessionFile: string | undefined;
  agentId?: string;
  sessionKey?: string;
  prompt: string;
  images?: ImageContent[];
  assistantText: string | undefined;
  provider: string;
  model: string;
  api?: string;
  startedAt: number;
  finishedAt?: number;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
};

function buildUsage(params: PersistCliTurnParams["usage"]): Usage {
  const input = params?.input ?? 0;
  const output = params?.output ?? 0;
  const cacheRead = params?.cacheRead ?? 0;
  const cacheWrite = params?.cacheWrite ?? 0;
  const totalTokens = params?.total ?? input + output + cacheRead + cacheWrite;
  return {
    input,
    output,
    cacheRead,
    cacheWrite,
    totalTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

export async function persistCliTurn(params: PersistCliTurnParams): Promise<void> {
  const sessionFile = params.sessionFile?.trim();
  if (!sessionFile) {
    return;
  }
  const text = params.assistantText?.trim();
  if (!text) {
    // Nothing useful to persist — don't leave a dangling user message either,
    // since the embedded path only records the pair after a successful run.
    return;
  }
  try {
    await repairSessionFileIfNeeded({ sessionFile });
    await prewarmSessionFile(sessionFile);
    const sessionManager = guardSessionManager(SessionManager.open(sessionFile), {
      agentId: params.agentId,
      sessionKey: params.sessionKey,
    });

    const userContent: UserMessage["content"] =
      params.images && params.images.length > 0
        ? [{ type: "text", text: params.prompt }, ...params.images]
        : params.prompt;
    const userMessage: UserMessage = {
      role: "user",
      content: userContent,
      timestamp: params.startedAt,
    };
    sessionManager.appendMessage(userMessage);

    const assistantMessage: AssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text }],
      api: params.api ?? params.provider,
      provider: params.provider,
      model: params.model,
      stopReason: "stop",
      usage: buildUsage(params.usage),
      timestamp: params.finishedAt ?? Date.now(),
    };
    sessionManager.appendMessage(assistantMessage);
  } catch (err) {
    cliBackendLog.warn(
      `cli session persist failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}
