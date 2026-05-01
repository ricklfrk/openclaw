import { describe, expect, it } from "vitest";
import {
  stripIndexNoise,
  stripRecallBoilerplate,
  stripRecallNoiseForPrompt,
} from "./workspace-memory.js";

const IMAGE_RESEND_BOILERPLATE =
  "To send an image back, prefer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image. fer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image.jpg (spaces ok, quote if needed) or a safe relative path like MEDIA:./image.jpg. Avoid absolute paths (MEDIA:/...) and ~ paths — they are blocked for security. Keep caption in the text body.";
const HEARTBEAT_FOLLOW_UP_PROMPT =
  "Check if there's anything you should follow up on with the user. Say it, then DO it — words without action = nothing happened. If nothing to follow up, reply HEARTBEAT_OK.";

describe("stripRecallBoilerplate", () => {
  it("removes duplicated image resend instructions before rerank", () => {
    const input = `user: [historical-media-ref; path-preserved-for-resend: /tmp/old.jpg]
${IMAGE_RESEND_BOILERPLATE}
assistant: kept memory content`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("/tmp/old.jpg");
    expect(out).toContain("assistant: kept memory content");
    expect(out).not.toContain("To send an image back");
    expect(out).not.toContain("MEDIA:https://example.com/image.jpg");
  });

  it("handles the non-duplicated canonical wording", () => {
    const input =
      "prefix To send an image back, prefer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image.jpg (spaces ok, quote if needed) or a safe relative path like MEDIA:./image.jpg. Avoid absolute paths (MEDIA:/...) and ~ paths — they are blocked for security. Keep caption in the text body. suffix";

    expect(stripRecallBoilerplate(input)).toBe("prefix  suffix");
  });

  it("removes heartbeat follow-up boilerplate and no-op replies", () => {
    const input = `before
${HEARTBEAT_FOLLOW_UP_PROMPT}
Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK. When reading HEARTBEAT.md, use workspace file /Users/mea/.openclaw/workspace/HEARTBEAT.md (exact case). Do not read docs/heartbeat.md. Current time: Monday, March 9th, 2026 — 11:39 PM (Asia/Hong_Kong) / 2026-03-09 15:39 UTC
assistant: HEARTBEAT_OK
assistant: NO_REPLY
assistant: NO_REPLY</final>
<final> NO_REPLY </final>
after`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("before");
    expect(out).toContain("after");
    expect(out).not.toContain("Check if there's anything");
    expect(out).not.toContain("Read HEARTBEAT.md");
    expect(out).not.toContain("Current time:");
    expect(out).not.toContain("HEARTBEAT_OK");
    expect(out).not.toContain("NO_REPLY");
  });

  it("removes previous-message replay blocks", () => {
    const input = `before
user: Here are the previous last 2 messages:
"""
USER (16:14)
[media attached: /tmp/old.jpg]
To send an image back, prefer the message tool (media/path/filePath). If you must inline, use MEDIA:https://example.com/image.jpg (spaces ok, quote if needed) or a safe relative path like MEDIA:./image.jpg. Avoid absolute paths (MEDIA:/...) and ~ paths — they are blocked for security. Keep caption in the text body.
"""
after`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("before");
    expect(out).toContain("after");
    expect(out).not.toContain("previous last");
    expect(out).not.toContain("media attached");
  });

  it("removes queued-message wrappers and system status lines", () => {
    const input = `user: [Queued messages while agent was busy]
---
Queued #1
System: [2026-04-11 14:39:14 GMT+8] Node: MEA的Mac mini (192.168.1.168) · app 2026.4.10 (2026041090) · mode local
user: System: [2026-04-11 14:39:14 GMT+8] Node: MEA的Mac mini (192.168.1.168) · app 2026.4.10 (2026041090) · mode local
System (untrusted): [2026-04-11 14:22:34 GMT+8] Model switched to opus (anthropic/claude-opus-4-6).
真實訊息`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("真實訊息");
    expect(out).not.toContain("Queued messages while agent was busy");
    expect(out).not.toContain("Queued #1");
    expect(out).not.toContain("Model switched");
    expect(out).not.toContain("Node: MEA");
  });

  it("removes pre-compaction memory flush boilerplate", () => {
    const input = `before
user: Pre-compaction memory flush. Store durable memories now (use memory/2026-03-06.md; create memory/ if needed). IMPORTANT: If the file already exists, APPEND new content only — do not overwrite existing entries. Do NOT create timestamped variant files (e.g., 2026-03-06-HHMM.md); always use the canonical 2026-03-06.md filename. If nothing to store, reply with NO_REPLY.
Current time: Friday, March 6th, 2026 — 7:51 PM (Asia/Hong_Kong) / 2026-03-06 11:51 UTC
after`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("before");
    expect(out).toContain("after");
    expect(out).not.toContain("Pre-compaction");
    expect(out).not.toContain("Current time:");
  });

  it("unwraps assistant final tags without removing useful content", () => {
    const input = `assistant: <final>
有用的回覆
</final>
assistant: <final>NO_REPLY</final>NO_REPLY
assistant: <think>
hidden reasoning HEARTBEAT_OK
</think>
<final>[[reply_to_current]] 可見回覆</final>
normal mention: <final> should stay`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("assistant:");
    expect(out).toContain("有用的回覆");
    expect(out).toContain("[[reply_to_current]] 可見回覆");
    expect(out).toContain("normal mention: <final> should stay");
    expect(out).not.toContain("hidden reasoning");
    expect(out).not.toContain("NO_REPLY");
    expect(out).not.toContain("</final>");
    expect(out).not.toContain("assistant: <final>");
  });

  it("removes dream section headings and legacy message id lines", () => {
    const input = `## Light Sleep
## REM Sleep
user: 今天天氣如何?
[message_id: 1770066346671]
assistant: 香港現在是 17°C`;

    const out = stripRecallBoilerplate(input);

    expect(out).toContain("user: 今天天氣如何?");
    expect(out).toContain("assistant: 香港現在是 17°C");
    expect(out).not.toContain("Light Sleep");
    expect(out).not.toContain("REM Sleep");
    expect(out).not.toContain("message_id");
  });
});

describe("stripIndexNoise", () => {
  it("compacts conversation and sender metadata into an attribution prefix", () => {
    const input = `Conversation info (untrusted metadata):
\`\`\`json
{
  "message_id": "1773994543025",
  "sender_id": "uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595",
  "sender": "Ricklf",
  "timestamp": "Fri 2026-03-20 16:15 GMT+8"
}
\`\`\`

Sender (untrusted metadata):
\`\`\`json
{
  "label": "Ricklf (+85264483210)",
  "id": "uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595",
  "name": "Ricklf",
  "e164": "+85264483210"
}
\`\`\`

成功了！！`;

    const out = stripIndexNoise(input);

    expect(out).toContain("Ricklf(Fri 2026-03-20 16:15 GMT+8): 成功了！！");
    expect(out).not.toContain("message_id");
    expect(out).not.toContain("sender_id");
    expect(out).not.toContain("+85264483210");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
  });

  it("compacts bare message metadata into an attribution prefix", () => {
    const input = `\`\`\`json
{
  "message_id": "1773994547712",
  "sender_id": "uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595",
  "sender": "Ricklf",
  "timestamp": "Fri 2026-03-20 16:15 GMT+8"
}
\`\`\`

成功了！！`;

    const out = stripIndexNoise(input);

    expect(out).toContain("Ricklf(Fri 2026-03-20 16:15 GMT+8): 成功了！！");
    expect(out).not.toContain("message_id");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
  });

  it("removes nested relevant-memories blocks from indexed daily journals", () => {
    const input = `before
<relevant-memories>
  <from-daily-memory>
    stale recalled memory
  </from-daily-memory>
</relevant-memories>
after`;

    const out = stripIndexNoise(input);

    expect(out).not.toContain("<relevant-memories>");
    expect(out).not.toContain("stale recalled memory");
    expect(out).toContain("before");
    expect(out).toContain("after");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
  });

  it("removes image resend boilerplate while preserving line count", () => {
    const input = `line 1
${IMAGE_RESEND_BOILERPLATE}
line 3`;

    const out = stripIndexNoise(input);

    expect(out).not.toContain("To send an image back");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
    expect(out).toContain("line 1");
    expect(out).toContain("line 3");
  });

  it("removes heartbeat boilerplate while preserving line count", () => {
    const input = `line 1
${HEARTBEAT_FOLLOW_UP_PROMPT}
assistant: HEARTBEAT_OK
assistant: NO_REPLY
line 4`;

    const out = stripIndexNoise(input);

    expect(out).not.toContain("Check if there's anything");
    expect(out).not.toContain("HEARTBEAT_OK");
    expect(out).not.toContain("NO_REPLY");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
    expect(out).toContain("line 1");
    expect(out).toContain("line 4");
  });

  it("removes replay and channel wrapper noise while preserving line count", () => {
    const input = `line 1
## Deep Sleep
user: Here are the previous last 1 messages:
"""
USER (14:22)
duplicated old context
"""
user: [Queued messages while agent was busy]
---
Queued #1
System: [2026-04-11 14:39:14 GMT+8] Node: MEA的Mac mini · mode local
[message_id: 1770066346671]
line 11`;

    const out = stripIndexNoise(input);

    expect(out).toContain("line 1");
    expect(out).toContain("line 11");
    expect(out).not.toContain("duplicated old context");
    expect(out).not.toContain("Queued #1");
    expect(out).not.toContain("Node: MEA");
    expect(out).not.toContain("Deep Sleep");
    expect(out).not.toContain("message_id");
    expect(out.split("\n")).toHaveLength(input.split("\n").length);
  });

  it("rewrites raw historical media markers while preserving paths", () => {
    const input =
      "user: [media attached: /Users/mea/.openclaw/media/inbound/old.jpg (image/jpeg) | /Users/mea/.openclaw/media/inbound/old.jpg]\nwhat is this?";

    const out = stripIndexNoise(input);

    expect(out).toContain("[historical-media-ref;");
    expect(out).toContain("not-current-attachment");
    expect(out).toContain("/Users/mea/.openclaw/media/inbound/old.jpg");
    expect(out).not.toContain("[media attached:");
  });
});

describe("stripRecallNoiseForPrompt", () => {
  it("removes nested relevant memories and resend boilerplate before rerank", () => {
    const input = `intro
<relevant-memories>
  <from-vector-memory>
    stale recalled memory
  </from-vector-memory>
</relevant-memories>
path: /tmp/old.jpg
[Image: source: /tmp/screenshot.png]
user: Here are the previous last 1 messages:
"""
duplicated old context
"""
${IMAGE_RESEND_BOILERPLATE}
${HEARTBEAT_FOLLOW_UP_PROMPT}
assistant: HEARTBEAT_OK
assistant: NO_REPLY
real memory`;

    const out = stripRecallNoiseForPrompt(input);

    expect(out).toContain("intro");
    expect(out).toContain("path: /tmp/old.jpg");
    expect(out).toContain("[historical-image-ref;");
    expect(out).toContain("/tmp/screenshot.png");
    expect(out).toContain("real memory");
    expect(out).not.toContain("<relevant-memories>");
    expect(out).not.toContain("stale recalled memory");
    expect(out).not.toContain("To send an image back");
    expect(out).not.toContain("duplicated old context");
    expect(out).not.toContain("[Image: source:");
    expect(out).not.toContain("HEARTBEAT_OK");
    expect(out).not.toContain("NO_REPLY");
  });

  it("compacts blank lines after load-time line-preserving cleanup", () => {
    const input = `assistant: NO_REPLY
Conversation info (untrusted metadata):
\`\`\`json
{
  "message_id": "1773994547712",
  "sender_id": "uuid:cb274c30-17ce-49ee-97c6-55dd9ce14595",
  "sender": "Ricklf",
  "timestamp": "Fri 2026-03-20 16:15 GMT+8"
}
\`\`\`

成功了！！
assistant: NO_REPLY`;

    const out = stripRecallNoiseForPrompt(stripIndexNoise(input));

    expect(out).toBe("Ricklf(Fri 2026-03-20 16:15 GMT+8): 成功了！！");
  });
});
