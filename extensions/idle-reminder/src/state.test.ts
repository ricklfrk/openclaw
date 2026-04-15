import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as state from "./state.js";

describe("idle-reminder state", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    state.stopAll();
    vi.useRealTimers();
  });

  it("starts a timer and calls onTimeout", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "test-session",
      timeoutMs: 5000,
      onTimeout,
    });

    expect(state.isActive("test-session")).toBe(true);
    expect(state.activeCount()).toBe(1);

    vi.advanceTimersByTime(5000);
    expect(onTimeout).toHaveBeenCalledWith("test-session");
  });

  it("resets timer on re-start", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "test-session",
      timeoutMs: 5000,
      onTimeout,
    });

    vi.advanceTimersByTime(3000);
    expect(onTimeout).not.toHaveBeenCalled();

    state.startTimer({
      sessionKey: "test-session",
      timeoutMs: 5000,
      lastUserText: "hello",
      onTimeout,
    });

    vi.advanceTimersByTime(3000);
    expect(onTimeout).not.toHaveBeenCalled();

    vi.advanceTimersByTime(2000);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it("stores messages in rolling buffer", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      lastUserText: "msg1",
      lastReplyText: "reply1",
      onTimeout,
    });

    const s = state.getState("s1");
    expect(s?.lastMessages).toHaveLength(2);
    expect(s?.lastMessages[0].role).toBe("user");
    expect(s?.lastMessages[1].role).toBe("agent");
  });

  it("incrementCount returns true when max reached", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      onTimeout,
    });

    expect(state.incrementCount("s1", 2)).toBe(false);
    expect(state.incrementCount("s1", 2)).toBe(true);
  });

  it("stop clears timer and removes state", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 5000,
      onTimeout,
    });

    state.stop("s1");
    expect(state.isActive("s1")).toBe(false);

    vi.advanceTimersByTime(5000);
    expect(onTimeout).not.toHaveBeenCalled();
  });

  it("stopAll clears all timers", () => {
    const onTimeout = vi.fn();
    state.startTimer({ sessionKey: "s1", timeoutMs: 1000, onTimeout });
    state.startTimer({ sessionKey: "s2", timeoutMs: 1000, onTimeout });

    expect(state.activeCount()).toBe(2);

    state.stopAll();
    expect(state.activeCount()).toBe(0);

    vi.advanceTimersByTime(1000);
    expect(onTimeout).not.toHaveBeenCalled();
  });

  it("resetCycle resets count and reschedules", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 5000,
      onTimeout,
    });

    state.incrementCount("s1", 10);
    expect(state.getState("s1")?.count).toBe(1);

    state.resetCycle("s1", onTimeout);
    expect(state.getState("s1")?.count).toBe(0);

    vi.advanceTimersByTime(5000);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it("stores transcriptSizeAtStart from params", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      transcriptSize: 42000,
      onTimeout,
    });

    expect(state.getState("s1")?.transcriptSizeAtStart).toBe(42000);
  });

  it("defaults transcriptSizeAtStart to 0 when not provided", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      onTimeout,
    });

    expect(state.getState("s1")?.transcriptSizeAtStart).toBe(0);
  });

  it("carries over messages from previous timer start", () => {
    const onTimeout = vi.fn();
    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      lastUserText: "first",
      onTimeout,
    });

    state.startTimer({
      sessionKey: "s1",
      timeoutMs: 1000,
      lastUserText: "second",
      lastReplyText: "reply",
      onTimeout,
    });

    const s = state.getState("s1");
    expect(s?.lastMessages).toHaveLength(3);
    expect(s?.lastMessages[0].text).toBe("first");
    expect(s?.lastMessages[1].text).toBe("second");
    expect(s?.lastMessages[2].text).toBe("reply");
  });
});
