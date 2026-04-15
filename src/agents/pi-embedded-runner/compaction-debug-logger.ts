import {
  calculateContextTokens,
  getLatestCompactionEntry,
  shouldCompact,
  type SessionEntry,
} from "@mariozechner/pi-coding-agent";
import { createSubsystemLogger } from "../../logging/subsystem.js";

const log = createSubsystemLogger("compaction-debug");

type AssistantMessageLike = {
  timestamp?: number;
  stopReason?: string;
  usage?: Parameters<typeof calculateContextTokens>[0];
};

type SessionLike = {
  _checkCompaction?: (
    assistantMessage: AssistantMessageLike,
    skipAbortedCheck?: boolean,
  ) => Promise<void>;
  model?: { contextWindow?: number; id?: string; provider?: string } | null;
  settingsManager?: {
    getCompactionSettings: () => {
      enabled: boolean;
      reserveTokens: number;
      keepRecentTokens: number;
    };
  };
  sessionManager?: { getBranch: () => SessionEntry[] };
};

/**
 * Diagnostic wrapper around Pi's private `_checkCompaction`. When enabled it
 * logs why the check did (or did not) trigger `_runAutoCompaction`, covering
 * every early-return branch in the SDK (disabled, aborted, stale-vs-boundary,
 * below-threshold). Intended for manual debugging only.
 *
 * Gated on `OPENCLAW_DEBUG_COMPACTION=1` so it never runs in production by
 * default. Uses a per-instance method override (not prototype mutation) so the
 * original SDK class is untouched.
 *
 * A missing `_checkCompaction entered` log at `agent_end` means the SDK never
 * even called `_checkCompaction` — usually because `_lastAssistantMessage` was
 * undefined or cleared by retry logic. That absence is itself the signal.
 */
export function installCheckCompactionDebugLogger(params: {
  session: unknown;
  agentId?: string;
}): () => void {
  if (process.env.OPENCLAW_DEBUG_COMPACTION !== "1") {
    return () => {};
  }

  const session = params.session as SessionLike;
  const agentId = params.agentId ?? "unknown";

  const original = session._checkCompaction;
  if (typeof original !== "function") {
    log.warn(
      `installCheckCompactionDebugLogger: session._checkCompaction not found (agent=${agentId}); skipping`,
    );
    return () => {};
  }
  const boundOriginal = original.bind(session);

  const wrapped: SessionLike["_checkCompaction"] = async function (
    assistantMessage,
    skipAbortedCheck = true,
  ) {
    try {
      const settings = session.settingsManager?.getCompactionSettings();
      const contextWindow = session.model?.contextWindow ?? 0;
      const branch = session.sessionManager?.getBranch();
      const compactionEntry = branch ? getLatestCompactionEntry(branch) : null;
      const msgTs =
        typeof assistantMessage?.timestamp === "number" ? assistantMessage.timestamp : undefined;
      const boundaryTs = compactionEntry
        ? new Date(compactionEntry.timestamp).getTime()
        : undefined;
      const stopReason = assistantMessage?.stopReason;
      const assistantIsFromBeforeCompaction =
        compactionEntry !== null && msgTs !== undefined && msgTs <= (boundaryTs ?? 0);

      let contextTokens: number | undefined;
      if (stopReason !== "error" && assistantMessage?.usage) {
        contextTokens = calculateContextTokens(assistantMessage.usage);
      }
      const wouldCompact =
        contextTokens !== undefined && settings
          ? shouldCompact(contextTokens, contextWindow, settings)
          : undefined;

      log.info(
        `_checkCompaction entered agent=${agentId} ` +
          `enabled=${settings?.enabled} stopReason=${stopReason ?? "(none)"} ` +
          `contextWindow=${contextWindow} reserveTokens=${settings?.reserveTokens ?? "?"} ` +
          `contextTokens=${contextTokens ?? "n/a"} shouldCompact=${wouldCompact ?? "n/a"} ` +
          `msgTs=${msgTs ?? "?"} boundaryTs=${boundaryTs ?? "(none)"} ` +
          `isBeforeBoundary=${assistantIsFromBeforeCompaction} skipAbortedCheck=${skipAbortedCheck}`,
      );

      if (!settings?.enabled) {
        log.info(`_checkCompaction early-return: compaction disabled (agent=${agentId})`);
      } else if (skipAbortedCheck && stopReason === "aborted") {
        log.info(
          `_checkCompaction early-return: last assistant message was aborted (agent=${agentId})`,
        );
      } else if (assistantIsFromBeforeCompaction) {
        log.info(
          `_checkCompaction early-return: assistant message timestamp (${msgTs}) is before ` +
            `latest compaction boundary (${boundaryTs}); SDK skips to avoid re-triggering ` +
            `on stale usage (agent=${agentId})`,
        );
      } else if (wouldCompact === false) {
        log.info(
          `_checkCompaction no-op: contextTokens below threshold ` +
            `(${contextTokens}/${contextWindow - (settings?.reserveTokens ?? 0)}) agent=${agentId}`,
        );
      } else if (wouldCompact === true) {
        log.info(`_checkCompaction will invoke _runAutoCompaction(threshold) agent=${agentId}`);
      }
    } catch (err) {
      log.warn(
        `compaction-debug wrap error agent=${agentId}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    return boundOriginal(assistantMessage, skipAbortedCheck);
  };

  session._checkCompaction = wrapped;

  return () => {
    if (session._checkCompaction === wrapped) {
      session._checkCompaction = original;
    }
  };
}
