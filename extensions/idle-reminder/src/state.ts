export type StoredMessage = {
  role: "user" | "agent";
  text: string;
  timestamp: number;
};

export type IdleReminderState = {
  sessionKey: string;
  timer: NodeJS.Timeout | null;
  timeoutMs: number;
  count: number;
  lastMessages: StoredMessage[];
  /** Transcript byte size when the timer was (re)started, used to detect new activity. */
  transcriptSizeAtStart: number;
};

const MAX_STORED_MESSAGES = 6;

const activeReminders = new Map<string, IdleReminderState>();

export function getState(sessionKey: string): IdleReminderState | undefined {
  return activeReminders.get(sessionKey);
}

export function startTimer(params: {
  sessionKey: string;
  timeoutMs: number;
  lastUserText?: string;
  lastReplyText?: string;
  transcriptSize?: number;
  onTimeout: (sessionKey: string) => void;
}): void {
  const { sessionKey, timeoutMs, onTimeout } = params;

  const existing = activeReminders.get(sessionKey);
  if (existing?.timer) {
    clearTimeout(existing.timer);
  }

  const previousMessages = existing?.lastMessages ?? [];
  const now = Date.now();
  const newMessages: StoredMessage[] = [];
  const userText = (params.lastUserText ?? "").trim();
  if (userText) {
    newMessages.push({ role: "user", text: userText, timestamp: now });
  }
  const agentText = (params.lastReplyText ?? "").trim();
  if (agentText) {
    newMessages.push({ role: "agent", text: agentText, timestamp: now });
  }
  const updatedMessages =
    newMessages.length > 0
      ? [...previousMessages, ...newMessages].slice(-MAX_STORED_MESSAGES)
      : previousMessages;

  const state: IdleReminderState = {
    sessionKey,
    timer: null,
    timeoutMs,
    count: 0,
    lastMessages: updatedMessages,
    transcriptSizeAtStart: params.transcriptSize ?? 0,
  };

  state.timer = setTimeout(() => onTimeout(sessionKey), timeoutMs);
  state.timer.unref?.();

  activeReminders.set(sessionKey, state);
}

export function reschedule(sessionKey: string, onTimeout: (sessionKey: string) => void): void {
  const state = activeReminders.get(sessionKey);
  if (!state) {
    return;
  }
  if (state.timer) {
    clearTimeout(state.timer);
  }
  state.timer = setTimeout(() => onTimeout(sessionKey), state.timeoutMs);
  state.timer.unref?.();
}

/** Reset the count and reschedule (e.g. new activity detected). */
export function resetCycle(
  sessionKey: string,
  onTimeout: (sessionKey: string) => void,
  newTranscriptSize?: number,
): void {
  const state = activeReminders.get(sessionKey);
  if (!state) {
    return;
  }
  state.count = 0;
  if (newTranscriptSize !== undefined) {
    state.transcriptSizeAtStart = newTranscriptSize;
  }
  reschedule(sessionKey, onTimeout);
}

/** Increment count after a successful delivery. Returns true if max reached. */
export function incrementCount(sessionKey: string, maxCount: number): boolean {
  const state = activeReminders.get(sessionKey);
  if (!state) {
    return true;
  }
  state.count++;
  return state.count >= maxCount;
}

export function stop(sessionKey: string): void {
  const state = activeReminders.get(sessionKey);
  if (!state) {
    return;
  }
  if (state.timer) {
    clearTimeout(state.timer);
  }
  activeReminders.delete(sessionKey);
}

export function stopAll(): void {
  for (const [, state] of activeReminders) {
    if (state.timer) {
      clearTimeout(state.timer);
    }
  }
  activeReminders.clear();
}

export function isActive(sessionKey: string): boolean {
  return activeReminders.has(sessionKey);
}

export function activeCount(): number {
  return activeReminders.size;
}
