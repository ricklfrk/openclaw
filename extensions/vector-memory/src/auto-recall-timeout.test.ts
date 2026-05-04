import { describe, expect, it } from "vitest";
import {
  assertWithinAutoRecallDeadline,
  AUTO_RECALL_HARD_TIMEOUT_MS,
  createAutoRecallDeadline,
  remainingAutoRecallMs,
  timeoutWithinAutoRecallDeadline,
} from "./auto-recall-timeout.js";

describe("auto-recall hard timeout", () => {
  it("defaults to a one minute deadline", () => {
    const deadline = createAutoRecallDeadline(100);

    expect(deadline.timeoutMs).toBe(60_000);
    expect(deadline.timeoutMs).toBe(AUTO_RECALL_HARD_TIMEOUT_MS);
    expect(deadline.deadlineMs).toBe(60_100);
  });

  it("caps child operation timeouts to the remaining deadline budget", () => {
    const deadline = createAutoRecallDeadline(1_000);

    expect(timeoutWithinAutoRecallDeadline(12_000, deadline, 10_000)).toBe(12_000);
    expect(timeoutWithinAutoRecallDeadline(12_000, deadline, 60_500)).toBe(500);
  });

  it("keeps a minimal positive timeout for Promise.race timers", () => {
    const deadline = createAutoRecallDeadline(1_000);

    expect(remainingAutoRecallMs(deadline, 61_001)).toBe(1);
    expect(timeoutWithinAutoRecallDeadline(12_000, deadline, 61_001)).toBe(1);
  });

  it("throws once the hard deadline is reached", () => {
    const deadline = createAutoRecallDeadline(1_000);

    expect(() => assertWithinAutoRecallDeadline(deadline, "recall", 60_999)).not.toThrow();
    expect(() => assertWithinAutoRecallDeadline(deadline, "recall", 61_000)).toThrow(
      "recall timed out after 60000ms",
    );
  });
});
