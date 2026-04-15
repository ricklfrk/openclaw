import fs from "node:fs/promises";
import { describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { resolveMainSessionKey } from "../config/sessions.js";
import { buildAgentPeerSessionKey } from "../routing/session-key.js";
import { runHeartbeatOnce } from "./heartbeat-runner.js";
import { installHeartbeatRunnerTestRuntime } from "./heartbeat-runner.test-harness.js";
import { withTempHeartbeatSandbox } from "./heartbeat-runner.test-utils.js";

vi.mock("jiti", () => ({ createJiti: () => () => ({}) }));

installHeartbeatRunnerTestRuntime();

describe("heartbeat per-peer session resolution", () => {
  it("loads per-peer session transcript when target=last and dmScope=per-peer", async () => {
    await withTempHeartbeatSandbox(
      async ({ tmpDir, storePath, replySpy }) => {
        const cfg: OpenClawConfig = {
          agents: {
            defaults: {
              workspace: tmpDir,
              heartbeat: { every: "5m", target: "last" },
            },
          },
          channels: { whatsapp: { allowFrom: ["*"] } },
          session: {
            store: storePath,
            dmScope: "per-peer",
            identityLinks: {
              testuser: ["whatsapp:+15551234567"],
            },
          },
        };

        const mainSessionKey = resolveMainSessionKey(cfg);
        const peerSessionKey = buildAgentPeerSessionKey({
          agentId: "main",
          channel: "whatsapp",
          peerKind: "direct",
          peerId: "+15551234567",
          dmScope: "per-peer",
          identityLinks: cfg.session?.identityLinks,
        });

        await fs.writeFile(
          storePath,
          JSON.stringify({
            [mainSessionKey]: {
              sessionId: "sid-main",
              updatedAt: Date.now(),
              lastChannel: "whatsapp",
              lastTo: "+15551234567",
            },
            [peerSessionKey]: {
              sessionId: "sid-peer",
              updatedAt: Date.now() + 1000,
            },
          }),
        );

        replySpy.mockResolvedValue([{ text: "Hello from heartbeat" }]);
        const sendWhatsApp = vi.fn().mockResolvedValue({ messageId: "m1", toJid: "jid" });

        await runHeartbeatOnce({
          cfg,
          deps: { whatsapp: sendWhatsApp as unknown, getQueueSize: () => 0, nowMs: () => 0 },
        });

        expect(replySpy).toHaveBeenCalledWith(
          expect.objectContaining({ SessionKey: peerSessionKey }),
          expect.any(Object),
          cfg,
        );
      },
      { prefix: "openclaw-hb-per-peer-" },
    );
  });

  it("falls back to main session when per-peer session does not exist", async () => {
    await withTempHeartbeatSandbox(
      async ({ tmpDir, storePath, replySpy }) => {
        const cfg: OpenClawConfig = {
          agents: {
            defaults: {
              workspace: tmpDir,
              heartbeat: { every: "5m", target: "last" },
            },
          },
          channels: { whatsapp: { allowFrom: ["*"] } },
          session: {
            store: storePath,
            dmScope: "per-peer",
            identityLinks: {
              testuser: ["whatsapp:+15551234567"],
            },
          },
        };

        const mainSessionKey = resolveMainSessionKey(cfg);

        await fs.writeFile(
          storePath,
          JSON.stringify({
            [mainSessionKey]: {
              sessionId: "sid-main",
              updatedAt: Date.now(),
              lastChannel: "whatsapp",
              lastTo: "+15551234567",
            },
          }),
        );

        replySpy.mockResolvedValue([{ text: "Hello from heartbeat" }]);
        const sendWhatsApp = vi.fn().mockResolvedValue({ messageId: "m1", toJid: "jid" });

        await runHeartbeatOnce({
          cfg,
          deps: { whatsapp: sendWhatsApp as unknown, getQueueSize: () => 0, nowMs: () => 0 },
        });

        expect(replySpy).toHaveBeenCalledWith(
          expect.objectContaining({ SessionKey: mainSessionKey }),
          expect.any(Object),
          cfg,
        );
      },
      { prefix: "openclaw-hb-peer-fallback-" },
    );
  });

  it("uses main session when dmScope is main (default)", async () => {
    await withTempHeartbeatSandbox(
      async ({ tmpDir, storePath, replySpy }) => {
        const cfg: OpenClawConfig = {
          agents: {
            defaults: {
              workspace: tmpDir,
              heartbeat: { every: "5m", target: "last" },
            },
          },
          channels: { whatsapp: { allowFrom: ["*"] } },
          session: { store: storePath },
        };

        const mainSessionKey = resolveMainSessionKey(cfg);

        await fs.writeFile(
          storePath,
          JSON.stringify({
            [mainSessionKey]: {
              sessionId: "sid-main",
              updatedAt: Date.now(),
              lastChannel: "whatsapp",
              lastTo: "+15551234567",
            },
          }),
        );

        replySpy.mockResolvedValue([{ text: "Hello from heartbeat" }]);
        const sendWhatsApp = vi.fn().mockResolvedValue({ messageId: "m1", toJid: "jid" });

        await runHeartbeatOnce({
          cfg,
          deps: { whatsapp: sendWhatsApp as unknown, getQueueSize: () => 0, nowMs: () => 0 },
        });

        expect(replySpy).toHaveBeenCalledWith(
          expect.objectContaining({ SessionKey: mainSessionKey }),
          expect.any(Object),
          cfg,
        );
      },
      { prefix: "openclaw-hb-dmscope-main-" },
    );
  });
});
