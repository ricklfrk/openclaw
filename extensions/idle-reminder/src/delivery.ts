import fs from "node:fs";
import type { OpenClawConfig } from "openclaw/plugin-sdk/core";
import {
  getReplyFromConfig,
  stripHeartbeatToken,
  HEARTBEAT_TOKEN,
  isSilentReplyText,
  deliverOutboundPayloads,
  buildOutboundSessionContext,
  resolveHeartbeatDeliveryTarget,
  resolveHeartbeatSenderContext,
} from "openclaw/plugin-sdk/reply-runtime";
import type { ReplyPayload } from "openclaw/plugin-sdk/reply-runtime";
import { buildIdleReminderPrompt } from "./prompt.js";
import type { StoredMessage } from "./state.js";

// Billing proxies may replace HEARTBEAT_OK with this token to avoid
// Anthropic's streaming classifier. Recognize both.
const PROXY_HEARTBEAT_TOKEN = "PULSE_ACK";

export type DeliveryResult = "delivered" | "skipped_empty" | "skipped_heartbeat_ok" | "no_target";

export type DeliveryLogger = {
  info: (message: string) => void;
  debug?: (message: string) => void;
  warn: (message: string) => void;
};

export type DeliveryDeps = {
  loadConfig: () => OpenClawConfig;
  loadSessionStore: (path: string) => Record<string, Record<string, unknown>>;
  saveSessionStore: (path: string, store: Record<string, Record<string, unknown>>) => Promise<void>;
  resolveSessionFilePath: (sessionId: string, entry: Record<string, unknown>) => string;
  resolveStorePath: (base: string | undefined, opts: Record<string, unknown>) => string;
  agentName?: string;
};

/**
 * Check whether the transcript grew with non-heartbeat content since `lastSize`.
 * Returns the new byte size regardless.
 */
export function hasNewNonHeartbeatContent(
  sessionId: string,
  transcriptPath: string,
  lastSize: number,
): { hasNew: boolean; currentSize: number } {
  try {
    const stat = fs.statSync(transcriptPath);
    const currentSize = stat.size;
    if (currentSize <= lastSize) {
      return { hasNew: false, currentSize };
    }

    const fd = fs.openSync(transcriptPath, "r");
    try {
      const buffer = Buffer.alloc(currentSize - lastSize);
      fs.readSync(fd, buffer, 0, buffer.length, lastSize);
      const newContent = buffer.toString("utf-8");
      const lines = newContent.split("\n").filter(Boolean);

      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          const content = parsed.content;
          let text = "";
          if (typeof content === "string") {
            text = content;
          } else if (Array.isArray(content)) {
            text = content
              .filter((c: Record<string, unknown>) => c.type === "text")
              .map((c: Record<string, unknown>) => c.text)
              .join("");
          }
          const trimmedText = text.trim();
          if (trimmedText === HEARTBEAT_TOKEN || trimmedText === PROXY_HEARTBEAT_TOKEN) {
            continue;
          }
          if (
            (text.includes(HEARTBEAT_TOKEN) && text.replace(HEARTBEAT_TOKEN, "").trim() === "") ||
            (text.includes(PROXY_HEARTBEAT_TOKEN) &&
              text.replace(PROXY_HEARTBEAT_TOKEN, "").trim() === "")
          ) {
            continue;
          }
          if (text.trim()) {
            return { hasNew: true, currentSize };
          }
        } catch {
          // Malformed JSONL line
        }
      }
      return { hasNew: false, currentSize };
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return { hasNew: false, currentSize: lastSize };
  }
}

export function resolveTranscriptSize(
  deps: Pick<DeliveryDeps, "loadSessionStore" | "resolveSessionFilePath">,
  sessionKey: string,
  storePath: string,
): number {
  try {
    const store = deps.loadSessionStore(storePath);
    const entry = store[sessionKey];
    if (!entry) {
      return 0;
    }
    const sessionId = entry.sessionId as string | undefined;
    if (!sessionId) {
      return 0;
    }
    const transcriptPath = deps.resolveSessionFilePath(sessionId, entry);
    return fs.statSync(transcriptPath).size;
  } catch {
    return 0;
  }
}

export function resolveTranscriptPath(
  deps: Pick<DeliveryDeps, "loadSessionStore" | "resolveSessionFilePath">,
  sessionKey: string,
  storePath: string,
): string | undefined {
  try {
    const store = deps.loadSessionStore(storePath);
    const entry = store[sessionKey];
    if (!entry) {
      return undefined;
    }
    const sessionId = entry.sessionId as string | undefined;
    if (!sessionId) {
      return undefined;
    }
    return deps.resolveSessionFilePath(sessionId, entry);
  } catch {
    return undefined;
  }
}

/** Send a simulated heartbeat and return the outcome. */
export async function sendSimulatedHeartbeat(params: {
  sessionKey: string;
  storePath: string;
  lastMessages: StoredMessage[];
  deps: DeliveryDeps;
  log: DeliveryLogger;
}): Promise<DeliveryResult> {
  const { sessionKey, storePath, lastMessages, deps, log } = params;
  const cfg = deps.loadConfig();
  const store = deps.loadSessionStore(storePath);
  const entry = store[sessionKey];
  if (!entry) {
    log.warn(`session entry not found: ${sessionKey}`);
    return "no_target";
  }

  // Resolve delivery target
  const heartbeatDelivery = resolveHeartbeatDeliveryTarget({ cfg, entry: entry as never });
  let deliveryChannel: string;
  let deliveryTo: string;
  let deliveryAccountId: string | undefined;

  if (heartbeatDelivery.channel !== "none" && heartbeatDelivery.to) {
    deliveryChannel = heartbeatDelivery.channel;
    deliveryTo = heartbeatDelivery.to;
    deliveryAccountId = heartbeatDelivery.accountId;
  } else if (entry.lastChannel && entry.lastTo) {
    deliveryChannel = entry.lastChannel as string;
    deliveryTo = entry.lastTo as string;
    deliveryAccountId = (entry.lastAccountId as string) ?? undefined;
  } else {
    log.info(`no delivery target: ${sessionKey}`);
    return "no_target";
  }

  const delivery = {
    ...heartbeatDelivery,
    channel: deliveryChannel,
    to: deliveryTo,
    accountId: deliveryAccountId,
  };

  const { sender } = resolveHeartbeatSenderContext({
    cfg,
    entry: entry as never,
    delivery: delivery as never,
  });

  const prompt = buildIdleReminderPrompt(lastMessages, deps.agentName);
  const ctx = {
    Body: prompt,
    From: sender,
    To: sender,
    OriginatingChannel: deliveryChannel,
    OriginatingTo: deliveryTo,
    AccountId: deliveryAccountId,
    Provider: "heartbeat",
    SessionKey: sessionKey,
  };

  // heartbeatUsePrimaryModel skips the heartbeat.model override and uses
  // the agent's primary model. The core fix in get-reply.ts ensures session
  // stored model overrides are also skipped.
  const replyResult = await getReplyFromConfig(
    ctx,
    { isHeartbeat: true, heartbeatUsePrimaryModel: true },
    cfg,
  );
  const replyPayload = pickLastPayload(replyResult);

  if (!replyPayload || isEmpty(replyPayload)) {
    log.debug?.(`empty reply, agent confirmed idle: ${sessionKey}`);
    return "skipped_empty";
  }

  // Check NO_REPLY, HEARTBEAT_OK, and PULSE_ACK (billing proxy variant).
  if (isSilentReplyText(replyPayload.text ?? "") && !hasMedia(replyPayload)) {
    log.debug?.(`NO_REPLY response, agent confirmed idle: ${sessionKey}`);
    return "skipped_heartbeat_ok";
  }

  const stripped = stripHeartbeatToken(replyPayload.text, { mode: "heartbeat" });
  if (stripped.shouldSkip && !hasMedia(replyPayload)) {
    log.debug?.(`HEARTBEAT_OK reply, agent confirmed idle: ${sessionKey}`);
    return "skipped_heartbeat_ok";
  }

  // Also strip PULSE_ACK (proxy-transformed HEARTBEAT_OK)
  let text = stripped.text || replyPayload.text || "";
  if (text.includes(PROXY_HEARTBEAT_TOKEN)) {
    text = text.replace(new RegExp(PROXY_HEARTBEAT_TOKEN, "g"), "").trim();
    if (!text && !hasMedia(replyPayload)) {
      log.debug?.(`PULSE_ACK-only reply, agent confirmed idle: ${sessionKey}`);
      return "skipped_heartbeat_ok";
    }
  }

  if (!text.trim() && !hasMedia(replyPayload)) {
    log.debug?.(`stripped reply empty, skipping: ${sessionKey}`);
    return "skipped_empty";
  }

  const payloads: ReplyPayload[] = [];
  if (hasMedia(replyPayload)) {
    payloads.push({
      text,
      mediaUrls: replyPayload.mediaUrls ?? (replyPayload.mediaUrl ? [replyPayload.mediaUrl] : []),
    });
  } else {
    payloads.push({ text });
  }

  await deliverOutboundPayloads({
    cfg,
    channel: deliveryChannel as never,
    to: deliveryTo,
    accountId: deliveryAccountId,
    payloads,
    session: buildOutboundSessionContext({ cfg, sessionKey }),
  });

  // Update lastHeartbeatSentAt
  const freshStore = deps.loadSessionStore(storePath);
  const current = freshStore[sessionKey];
  if (current) {
    freshStore[sessionKey] = { ...current, lastHeartbeatSentAt: Date.now() };
    await deps.saveSessionStore(storePath, freshStore);
  }

  log.info(`simulated heartbeat delivered: ${deliveryChannel} ${sessionKey} ${text.slice(0, 100)}`);

  return "delivered";
}

function pickLastPayload(
  result: ReplyPayload | ReplyPayload[] | undefined,
): ReplyPayload | undefined {
  if (!result) {
    return undefined;
  }
  if (!Array.isArray(result)) {
    return result;
  }
  for (let i = result.length - 1; i >= 0; i--) {
    const p = result[i];
    if (p && (p.text || p.mediaUrl || (p.mediaUrls && p.mediaUrls.length > 0))) {
      return p;
    }
  }
  return undefined;
}

function isEmpty(p: ReplyPayload): boolean {
  return !p.text && !p.mediaUrl && !(p.mediaUrls && p.mediaUrls.length > 0);
}

function hasMedia(p: ReplyPayload): boolean {
  return Boolean(p.mediaUrl || (p.mediaUrls && p.mediaUrls.length > 0));
}
