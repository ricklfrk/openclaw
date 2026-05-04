export const AUTO_RECALL_HARD_TIMEOUT_MS = 60_000;

export type AutoRecallDeadline = Readonly<{
  startedAtMs: number;
  timeoutMs: number;
  deadlineMs: number;
}>;

export function createAutoRecallDeadline(
  startedAtMs: number,
  timeoutMs = AUTO_RECALL_HARD_TIMEOUT_MS,
): AutoRecallDeadline {
  return {
    startedAtMs,
    timeoutMs,
    deadlineMs: startedAtMs + timeoutMs,
  };
}

export function remainingAutoRecallMs(
  deadline: AutoRecallDeadline,
  nowMs = performance.now(),
): number {
  return Math.max(1, Math.ceil(deadline.deadlineMs - nowMs));
}

export function timeoutWithinAutoRecallDeadline(
  requestedMs: number,
  deadline: AutoRecallDeadline,
  nowMs = performance.now(),
): number {
  return Math.max(1, Math.min(requestedMs, remainingAutoRecallMs(deadline, nowMs)));
}

export function assertWithinAutoRecallDeadline(
  deadline: AutoRecallDeadline,
  label = "vector-memory auto-recall",
  nowMs = performance.now(),
): void {
  if (nowMs >= deadline.deadlineMs) {
    throw new Error(`${label} timed out after ${deadline.timeoutMs}ms`);
  }
}
