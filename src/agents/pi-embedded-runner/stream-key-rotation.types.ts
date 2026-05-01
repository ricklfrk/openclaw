/** Shared mutable state between the streamFn wrapper and the run.ts while-loop. */
export type KeyRotationState = {
  /** Set to true when the wrapper has exhausted all candidate keys on a 429. */
  allKeysExhausted: boolean;
  /** The profile ID that was last used by the wrapper (for error attribution). */
  lastProfileId?: string;
  /** Current rotation index into profileCandidates. */
  rotationIndex: number;
  /**
   * Profiles to unconditionally skip for the remainder of this turn.
   * Populated by both the inner key-rotation loop (429/503) and the outer
   * run loop (empty response). Ensures no profile is retried within the
   * same turn even if its time-based cooldown expires mid-turn.
   */
  skipProfiles: Set<string>;
};

export function createKeyRotationState(): KeyRotationState {
  return { allKeysExhausted: false, rotationIndex: 0, skipProfiles: new Set() };
}
