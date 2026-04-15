/**
 * Idle Reminder Plugin
 *
 * Sends a follow-up prompt when the agent goes idle after a conversation.
 * Uses the primary model with full session context.
 *
 * Hooks:
 *   agent_end    -> start/reset idle timer after non-heartbeat runs
 *   gateway_stop -> clean up all timers on shutdown
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import {
  sendSimulatedHeartbeat,
  hasNewNonHeartbeatContent,
  resolveTranscriptPath,
  resolveTranscriptSize,
} from "./src/delivery.js";
import * as state from "./src/state.js";

const DEFAULT_IDLE_TIMEOUT_MS = 3 * 60 * 1000;
const MAX_REMIND_COUNT = 1;

let registered = false;

export default definePluginEntry({
  id: "idle-reminder",
  name: "Idle Reminder",
  description: "Sends a follow-up prompt when the agent goes idle after a conversation",

  register(api) {
    const log = api.logger;
    const { runtime } = api;

    if (registered) {
      return;
    }
    registered = true;

    log.info("idle-reminder plugin registered");

    const storePath = runtime.agent.session.resolveStorePath(undefined, {});
    const agentIdentity = runtime.agent.resolveAgentIdentity(api.config, "main");

    const deliveryDeps = {
      loadConfig: runtime.config.loadConfig,
      loadSessionStore: runtime.agent.session.loadSessionStore as (
        path: string,
      ) => Record<string, Record<string, unknown>>,
      saveSessionStore: runtime.agent.session.saveSessionStore as (
        path: string,
        store: Record<string, Record<string, unknown>>,
      ) => Promise<void>,
      resolveSessionFilePath: runtime.agent.session.resolveSessionFilePath as (
        sessionId: string,
        entry: Record<string, unknown>,
      ) => string,
      resolveStorePath: runtime.agent.session.resolveStorePath as (
        base: string | undefined,
        opts: Record<string, unknown>,
      ) => string,
      agentName: agentIdentity?.name,
    };

    function handleTimeout(sessionKey: string): void {
      void checkAndMaybeRemind(sessionKey);
    }

    async function checkAndMaybeRemind(sessionKey: string): Promise<void> {
      const current = state.getState(sessionKey);
      if (!current) {
        log.info(`timer fired but state gone: session=${sessionKey}`);
        return;
      }

      log.info(
        `timer fired: session=${sessionKey} count=${current.count}/${MAX_REMIND_COUNT} msgs=${current.lastMessages.length}`,
      );

      const transcriptPath = resolveTranscriptPath(deliveryDeps, sessionKey, storePath);
      if (transcriptPath) {
        const { hasNew, currentSize } = hasNewNonHeartbeatContent(
          sessionKey,
          transcriptPath,
          current.transcriptSizeAtStart,
        );
        if (hasNew) {
          log.info(`new activity detected, resetting: ${sessionKey}`);
          state.resetCycle(sessionKey, handleTimeout, currentSize);
          return;
        }
      }

      if (current.count >= MAX_REMIND_COUNT) {
        log.info(`max reminders reached: ${sessionKey} count=${current.count}`);
        state.stop(sessionKey);
        return;
      }

      log.info(`session idle, sending simulated heartbeat: ${sessionKey} count=${current.count}`);

      let result: Awaited<ReturnType<typeof sendSimulatedHeartbeat>> = "skipped_empty";
      try {
        result = await sendSimulatedHeartbeat({
          sessionKey,
          storePath,
          lastMessages: current.lastMessages,
          deps: deliveryDeps,
          log,
        });
      } catch (err) {
        log.info(`simulated heartbeat failed: ${sessionKey} ${String(err)}`);
      }

      log.info(`delivery result: session=${sessionKey} result=${result}`);

      if (result === "delivered") {
        const maxReached = state.incrementCount(sessionKey, MAX_REMIND_COUNT);
        if (maxReached) {
          log.info(`max reached after delivery, stopping: session=${sessionKey}`);
          state.stop(sessionKey);
        } else {
          log.info(`rescheduling next cycle: session=${sessionKey}`);
          state.reschedule(sessionKey, handleTimeout);
        }
      } else {
        log.info(`agent confirmed idle (${result}), stopping: session=${sessionKey}`);
        state.stop(sessionKey);
      }
    }

    // --- Hooks ---

    // Stop the idle timer as soon as a new user-triggered agent run begins.
    // This prevents the timer from firing while the agent is actively processing.
    api.on("before_agent_start", (_event, ctx) => {
      if (ctx.trigger === "heartbeat") {
        return;
      }
      const sessionKey = ctx.sessionKey;
      if (!sessionKey) {
        return;
      }
      if (state.isActive(sessionKey)) {
        log.info(`stopping timer: agent run starting for session=${sessionKey}`);
        state.stop(sessionKey);
      }
    });

    api.on("agent_end", (_event, ctx) => {
      log.info(
        `agent_end hook fired: session=${ctx.sessionKey ?? "?"} trigger=${ctx.trigger ?? "?"} agent=${ctx.agentId ?? "?"}`,
      );

      if (ctx.trigger === "heartbeat") {
        log.info(`skipping heartbeat trigger: session=${ctx.sessionKey ?? "?"}`);
        return;
      }
      const sessionKey = ctx.sessionKey;
      if (!sessionKey) {
        log.info("skipping: no sessionKey in agent_end context");
        return;
      }

      // Extract last user + agent text from the LLM messages snapshot.
      const messages = (_event.messages ?? []) as Array<{
        role?: string;
        content?: string | Array<{ type?: string; text?: string }>;
      }>;
      let lastUserText: string | undefined;
      let lastReplyText: string | undefined;
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        const text = extractText(msg?.content);
        if (!text) {
          continue;
        }
        if (msg?.role === "assistant" && !lastReplyText) {
          lastReplyText = text;
        } else if (msg?.role === "user" && !lastUserText) {
          lastUserText = text;
        }
        if (lastUserText && lastReplyText) {
          break;
        }
      }

      const txSize = resolveTranscriptSize(deliveryDeps, sessionKey, storePath);
      const existing = state.isActive(sessionKey);
      state.startTimer({
        sessionKey,
        timeoutMs: DEFAULT_IDLE_TIMEOUT_MS,
        lastUserText,
        lastReplyText,
        transcriptSize: txSize,
        onTimeout: handleTimeout,
      });
      log.info(
        `timer ${existing ? "reset" : "started"}: session=${sessionKey} timeoutMs=${DEFAULT_IDLE_TIMEOUT_MS} txSize=${txSize} userText=${(lastUserText ?? "").slice(0, 60)} replyText=${(lastReplyText ?? "").slice(0, 60)}`,
      );
    });

    api.on("gateway_stop", () => {
      const count = state.activeCount();
      if (count > 0) {
        log.info(`gateway stopping, clearing all idle reminders: count=${count}`);
      }
      state.stopAll();
    });
  },
});

function extractText(
  content: string | Array<{ type?: string; text?: string }> | undefined,
): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    return content
      .filter((c) => c.type === "text")
      .map((c) => c.text ?? "")
      .join("")
      .trim();
  }
  return "";
}
