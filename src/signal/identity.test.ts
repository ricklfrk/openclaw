import { describe, expect, it } from "vitest";
import {
  isSignalSenderAllowed,
  looksLikeUuid,
  resolveSignalPeerId,
  resolveSignalRecipient,
  resolveSignalSender,
} from "./identity.js";

describe("looksLikeUuid", () => {
  it("accepts hyphenated UUIDs", () => {
    expect(looksLikeUuid("123e4567-e89b-12d3-a456-426614174000")).toBe(true);
  });

  it("accepts compact UUIDs", () => {
    expect(looksLikeUuid("123e4567e89b12d3a456426614174000")).toBe(true);
  });

  it("accepts uuid-like hex values with letters", () => {
    expect(looksLikeUuid("abcd-1234")).toBe(true);
  });

  it("rejects numeric ids and phone-like values", () => {
    expect(looksLikeUuid("1234567890")).toBe(false);
    expect(looksLikeUuid("+15555551212")).toBe(false);
  });
});

describe("resolveSignalSender", () => {
  it("returns phone sender with uuid when both sourceNumber and sourceUuid are present", () => {
    const sender = resolveSignalSender({
      sourceNumber: "+15550001111",
      sourceUuid: "cb274c30-17ce-49ee-97c6-55dd9ce14595",
    });
    expect(sender).toEqual({
      kind: "phone",
      raw: "+15550001111",
      e164: "+15550001111",
      uuid: "cb274c30-17ce-49ee-97c6-55dd9ce14595",
    });
  });

  it("returns phone sender without uuid when sourceUuid is missing", () => {
    const sender = resolveSignalSender({ sourceNumber: "+15550001111" });
    expect(sender).toEqual({
      kind: "phone",
      raw: "+15550001111",
      e164: "+15550001111",
      uuid: undefined,
    });
  });

  it("falls back to uuid sender when sourceNumber is absent", () => {
    const sender = resolveSignalSender({
      sourceUuid: "cb274c30-17ce-49ee-97c6-55dd9ce14595",
    });
    expect(sender).toEqual({
      kind: "uuid",
      raw: "cb274c30-17ce-49ee-97c6-55dd9ce14595",
    });
  });

  it("returns null when neither identifier is present", () => {
    expect(resolveSignalSender({})).toBeNull();
  });

  it("maps uuid senders to recipient and peer ids", () => {
    const sender = { kind: "uuid", raw: "123e4567-e89b-12d3-a456-426614174000" } as const;
    expect(resolveSignalRecipient(sender)).toBe("123e4567-e89b-12d3-a456-426614174000");
    expect(resolveSignalPeerId(sender)).toBe("uuid:123e4567-e89b-12d3-a456-426614174000");
  });
});

describe("isSignalSenderAllowed", () => {
  it("matches phone allowlist entry against phone sender", () => {
    const sender = { kind: "phone" as const, raw: "+15550001111", e164: "+15550001111" };
    expect(isSignalSenderAllowed(sender, ["+15550001111"])).toBe(true);
    expect(isSignalSenderAllowed(sender, ["+15559999999"])).toBe(false);
  });

  it("matches uuid allowlist entry against uuid sender", () => {
    const sender = { kind: "uuid" as const, raw: "cb274c30-17ce-49ee-97c6-55dd9ce14595" };
    expect(isSignalSenderAllowed(sender, ["uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595"])).toBe(true);
    expect(isSignalSenderAllowed(sender, ["uuid:00000000-0000-0000-0000-000000000000"])).toBe(
      false,
    );
  });

  it("matches uuid allowlist entry against phone sender that carries a uuid", () => {
    const sender = {
      kind: "phone" as const,
      raw: "+15550001111",
      e164: "+15550001111",
      uuid: "cb274c30-17ce-49ee-97c6-55dd9ce14595",
    };
    expect(isSignalSenderAllowed(sender, ["uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595"])).toBe(true);
  });

  it("rejects uuid allowlist entry when phone sender has no uuid", () => {
    const sender = { kind: "phone" as const, raw: "+15550001111", e164: "+15550001111" };
    expect(isSignalSenderAllowed(sender, ["uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595"])).toBe(
      false,
    );
  });

  it("allows wildcard (*)", () => {
    const sender = { kind: "phone" as const, raw: "+15550001111", e164: "+15550001111" };
    expect(isSignalSenderAllowed(sender, ["*"])).toBe(true);
  });

  it("returns false for empty allowlist", () => {
    const sender = { kind: "phone" as const, raw: "+15550001111", e164: "+15550001111" };
    expect(isSignalSenderAllowed(sender, [])).toBe(false);
  });
});
