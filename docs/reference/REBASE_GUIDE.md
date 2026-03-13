# Custom Branch Rebase Guide

Reference for rebasing the `custom` branch onto upstream `origin/main`.
Last rebase: **2026-03-06** (upstream at ~324 commits ahead).

## Custom Commits (newest first)

| Commit      | Summary                           | Key files                                                                                                                |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ddf111b75` | fix: post-rebase cleanup          | `run.ts`, `attempt.ts`, `qmd-manager.ts`, `signal-enhancements.ts`                                                       |
| `15a540b31` | feat: signal delivery retry       | `signal-enhancements.ts`, `event-handler.ts`, `send.ts`                                                                  |
| `97ea9f7d9` | feat: gcli non-streaming          | `gcli-nonstream.ts`, `google-nonstream.ts`, `buffered-stream.ts`                                                         |
| `5827f63d3` | fix: skipProfiles for 429/503     | `stream-key-rotation.ts`, `run.ts`                                                                                       |
| `d72b71327` | fix: unify key rotation loop      | `stream-key-rotation.ts`, `run.ts`                                                                                       |
| `8b8564b6e` | fix: rebase cleanup               | test mocks, dead code                                                                                                    |
| `f474a4ea7` | chore: docs, scripts, deps        | `rebuild-mac.sh`, `restart-mac.sh`, docs                                                                                 |
| `496e948e4` | feat: config/gateway/cron/QMD     | `zod-schema.*.ts`, `connect-policy.ts`, `cron/ops.ts`, `qmd-manager.ts`                                                  |
| `c36f5b134` | feat: plugins & hooks             | `extensions/nsfw/`, `extensions/regex-replace/`, `idle-reminder.ts`                                                      |
| `dcaac68a1` | feat: Signal enhancements         | `signal-enhancements.ts`, `event-handler.ts`, `identity.ts`                                                              |
| `a8956359f` | feat: agent runner & key rotation | `run.ts`, `attempt.ts`, `stream-key-rotation.ts`, `non-conforming-retry.ts`, `compact.ts`, `thinking.ts`, `subscribe.ts` |

## High-Conflict Files

These files touch both upstream and custom logic heavily. Expect merge conflicts on every rebase.

### 1. `src/agents/pi-embedded-runner/run.ts`

**Custom additions:**

- `keyRotationState` + `createKeyRotationState()` — shared mutable state between streamFn wrapper and while-loop
- `nonConformingRetryCount` — retry counter for `<final>` tag enforcement
- `allKeysExhausted` error path with `buildErrorAgentMeta()`
- `skipProfiles` integration — empty response adds profile to skip set → triggers inner rotation exhaustion
- `FailoverError` throw when `fallbackConfigured && allKeysExhausted`

**Conflict pattern:** Upstream adds new params to `runEmbeddedAttempt()` call (line ~809-876). Custom adds `keyRotationState`, `keyRotationCandidates`, `keyRotationAuthStore`, `keyRotationResolveApiKey`. Resolution: keep ALL params from both sides.

**Critical ordering dependency (documented in code):**

```
allKeysExhausted check → shouldRotate check → checkNonConformingOutput
```

`allKeysExhausted` MUST run before `checkNonConformingOutput`. Empty responses from exhausted keys would otherwise trigger non-conforming "fail" → continue → infinite loop.

### 2. `src/agents/pi-embedded-runner/run/attempt.ts`

**Custom additions:**

- `wrapStreamFnWithKeyRotation` import and wrapper application (must be outermost after buffer)
- `legacyBeforeAgentStartResult: undefined` — intentional, forces `before_agent_start` hook to re-run inside `resolvePromptBuildHookResult` so plugins receive `systemPrompt`
- `wrapGcliNonStreaming` and `wrapGoogleNonStreaming` wrappers
- `wrapStreamFnWithBuffer` wrapper

**StreamFn wrapper ordering (inner → outer):**

1. `dropThinkingBlocks` (if needed)
2. `sanitizeToolCallIds` (if needed)
3. `downgradeOpenAIFunctionCallReasoningPairs` (if openai-responses)
4. `wrapStreamFnTrimToolCallNames`
5. `wrapStreamFnDecodeXaiToolCallArguments` (if xAI)
6. `anthropicPayloadLogger` (if enabled)
7. `wrapGoogleNonStreaming`
8. `wrapGcliNonStreaming`
9. `wrapStreamFnWithBuffer`
10. **`wrapStreamFnWithKeyRotation`** (must be outermost)

**Conflict pattern:** Upstream adds new imports and new wrapper steps. Resolution: keep all imports from both sides; maintain wrapper ordering.

### 3. `src/memory/qmd-manager.ts`

**Custom additions:**

- `McporterToolName`, `MCPORTER_TOOL_ALIASES`, `mcporterToolForSearchMode` — tool name resolution for mcporter MCP
- `isMcporterToolNotFoundError`, `parseMcporterJson` — error/parse helpers
- `runQmdSearchViaMcporter` — search via mcporter daemon (accepts docid-less results)
- `runMcporterAcrossCollections` — multi-collection dedup using `buildQmdResultKey`

**Known pitfalls:**

- `QMD_CONFIG_DIR` env var: upstream uses `this.xdgConfigHome` (workaround for QMD bug). Do NOT add `path.join(this.xdgConfigHome, "qmd")` — it was a merge artifact that shadowed the correct value.
- Docid-less results: `runQmdSearchViaMcporter` must allow `docid: undefined`; dedup uses `buildQmdResultKey` which falls back to `collection+file` when docid is missing.

### 4. `src/signal/monitor/event-handler.ts`

**Custom additions:**

- `signalDeliveryWithRetry` wrapping all send calls
- `buildEnhancedMessage` replacing inline message parsing
- `checkRequireMention` — custom mention gating (replaces upstream's `resolveChannelGroupRequireMention` when `enhancementDeps` present)
- Multi-attachment support (`mediaPaths[]` / `mediaTypes[]`)
- `preCacheGroupMedia` — persistent media index
- `hasNativeSignalMention` — Signal native @mention detection
- `stripMentionPlaceholders` — U+FFFC stripping for command detection
- `resolveAgentRoute` moved earlier (before mention check)

**Conflict pattern:** Upstream refactors imports (e.g. Slack account inspection renames). Resolution: keep all imports from both sides.

### 5. `src/config/zod-schema.agent-runtime.ts`

**Custom additions:**

- `AgentModelSchema` — inline `z.union([z.string(), z.object({ primary, compact, fallbacks })])` replacing upstream's plain string schema
- Must coexist with upstream's `AgentRuntimeAcpSchema` and `AgentRuntimeSchema`

### 6. `src/channels/dock.ts`

**Custom additions:**

- `looksLikeUuid` import from `signal/identity.js` for UUID allowlist matching

**Conflict pattern:** Upstream renames Slack imports. Resolution: keep both import sets.

## Low-Conflict Custom Files (rarely touch upstream)

These files are mostly new or isolated — unlikely to conflict:

| File                                   | Purpose                                     |
| -------------------------------------- | ------------------------------------------- |
| `stream-key-rotation.ts` + `.test.ts`  | Per-turn API key rotation wrapper           |
| `non-conforming-retry.ts` + `.test.ts` | `<final>` tag enforcement retry logic       |
| `buffered-stream.ts`                   | Google model stream buffering               |
| `gcli-nonstream.ts`                    | Cloud Code Assist non-streaming             |
| `google-nonstream.ts`                  | Google Generative AI non-streaming          |
| `signal-enhancements.ts`               | All Signal custom features                  |
| `custom-context-to-blocks.ts`          | `<call>` tag / historical context promotion |
| `idle-reminder.ts`                     | Idle session follow-up                      |
| `extensions/nsfw/`                     | NSFW toggle plugin                          |
| `extensions/regex-replace/`            | Regex replace plugin                        |
| `rebuild-mac.sh`                       | Dev rebuild script                          |

## Rebase Procedure

```bash
# 1. Fetch upstream
git fetch origin

# 2. Update fork/main to match origin/main
git push fork origin/main:refs/heads/main

# 3. Rebase custom onto updated origin/main
git rebase origin/main

# 4. For each conflict:
#    - Read BOTH sides carefully
#    - Keep ALL additive changes from both upstream and custom
#    - Check the "High-Conflict Files" section above for specific patterns
#    - Resolve, then: git add <file> && GIT_EDITOR=true git rebase --continue

# 5. For pnpm-lock.yaml conflicts: accept upstream version, then run pnpm install

# 6. Post-rebase verification:
#    - grep for duplicate definitions (e.g. QMD_CONFIG_DIR)
#    - grep for console.log (should be log.debug)
#    - check that buildErrorAgentMeta is used in ALL error return paths
#    - verify streamFn wrapper ordering in attempt.ts
#    - run: pnpm check && pnpm test
```

## Past Bugs Found During Rebase

### 2026-03-06 Rebase (7 bugs fixed in `ddf111b75`)

| #   | File                     | Bug                                                                                                      | Fix                                                       |
| --- | ------------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1   | `qmd-manager.ts`         | `QMD_CONFIG_DIR` defined twice (auto-merge artifact)                                                     | Removed custom line, kept upstream `this.xdgConfigHome`   |
| 2   | `qmd-manager.ts`         | mcporter path drops docid-less results                                                                   | Accept undefined docid; use `buildQmdResultKey` for dedup |
| 3   | `signal-enhancements.ts` | `isTransientDeliveryError` had redundant `if` block                                                      | Removed dead code                                         |
| 4   | `run.ts`                 | `allKeysExhausted` path used inline `agentMeta` instead of `buildErrorAgentMeta()`                       | Replaced with `buildErrorAgentMeta()` call                |
| 5   | `run.ts`                 | Two `console.log` debug traces                                                                           | Changed to `log.debug`                                    |
| 6   | `attempt.ts`             | Hook `systemPrompt` set twice (custom `setSystemPrompt` + upstream `applySystemPromptOverrideToSession`) | Removed custom `setSystemPrompt` block                    |
| 7   | `run.ts`                 | Implicit ordering between `allKeysExhausted` and `checkNonConformingOutput`                              | Added ordering comment                                    |

## Design Notes

### `legacyBeforeAgentStartResult: undefined` (intentional)

`run.ts:842` passes `undefined` instead of the cached hook result. This forces `before_agent_start` to re-run inside `resolvePromptBuildHookResult` with `systemPrompt` available. The NSFW plugin depends on this to replace the identity line in the system prompt.

Trade-off: the hook fires **twice per turn** (once for model override at top level, once for prompt context inside attempt). First invocation is a no-op for existing plugins. Third-party plugins with side effects would fire twice.

### `sanitizeSessionHistory` in compact.ts (intentional omission)

`compact.ts` omits `modelApi`/`modelId`/`provider`/`allowedToolNames` from the `sanitizeSessionHistory` call. This prevents the compact model from writing a model snapshot that would cause a spurious "model changed" event on the next real run. The `policy` param is passed directly (already resolved), so sanitization behavior is unaffected.

### Diagnostic `log.warn` in subscribe handlers

`pi-embedded-subscribe.handlers.messages.ts` has two `[msg-end-diag]` traces at `log.warn` level, gated behind `enforceFinalTag`. These fire on every Google/Gemini/Ollama/Minimax message. Consider downgrading to `log.debug` if log noise becomes an issue.

### `sanitizeForPlainText` vs `message_sending` hook ordering

`src/infra/outbound/deliver.ts` — `deliverOutboundPayloads` has two-phase normalization:

1. **Pre-hook**: `normalizePayloadsForChannelDelivery(payloads, channel, { skipPlainTextSanitize: true })` — media/reply parsing only, no HTML tag stripping.
2. **Post-hook**: `sanitizeForPlainText()` runs after the `message_sending` plugin hook.

**Why:** `sanitizeForPlainText` strips all HTML-like tags via `/<\/?[a-z][a-z0-9]*\b[^>]*>/gi`. This matched `<disclaimer>` (lowercase, no special chars) but NOT `<Content_Target>` (underscore breaks `[a-z0-9]*`). When sanitization ran before the hook, the `regex-replace` plugin saw text without `<disclaimer>` tags and reported `no match` — the disclaimer content was delivered unredacted to Signal and other plain-text surfaces.

**Edge cases preserved:**

- Telegram `sendPayload` with `channelData` uses `textMode: "html"` — sanitization is skipped for this path.
- HTML-only payloads (e.g. `<br><br>`) that become empty after sanitization are dropped.

**Note:** Signal monitor's `deliverReplies` (in `src/signal/monitor.ts`) does NOT call `sanitizeForPlainText` — it runs the hook directly on the raw text, then passes to `sendMessageSignal` which calls `markdownToSignalText`. This path was already correct; only the `deliverOutboundPayloads` path (used by followup-runner and heartbeat-runner) was affected.

### Google tool schema sanitization scope

`src/agents/pi-embedded-runner/google.ts` — `sanitizeToolsForGoogle` applies `cleanToolSchemaForGemini` when any of:

- `isGoogleModelApi(modelApi)` (google-generative-ai, google-gemini-cli)
- `provider === "google-gemini-cli"`
- `modelId` starts with `"gemini"` (covers OpenAI-compatible proxies forwarding to Gemini)

**Why the modelId check:** After rebase, upstream gained `Type.Union` schemas in tool definitions. These produce `anyOf`/`oneOf` in JSON Schema, which Gemini rejects with `400 Invalid JSON payload`. The `modelApi` check alone missed Gemini models accessed via non-Google providers (e.g. `openai-chat` proxy → gemini-2.5-flash).

## Delivery Path Reference

Two distinct paths deliver messages to Signal (and other channels). Understanding which path applies is critical for debugging hook issues.

### Path A: Signal monitor `deliverReplies` (direct inbound reply)

```
Signal monitor receives message
  → getReplyFromConfig → agent run → payloads
  → deliverReplies (src/signal/monitor.ts:295)
    → message_sending hook (raw text)
    → sendMessageSignal → markdownToSignalText
    → [signal] delivered reply (logged)
```

Used for: direct replies to incoming Signal messages when the dispatcher is wired.

### Path B: `deliverOutboundPayloads` (followup/heartbeat/route-reply)

```
followup-runner / heartbeat-runner
  → routeReply → deliverOutboundPayloads (src/infra/outbound/deliver.ts)
    → normalizePayloadsForChannelDelivery (skip plain-text sanitize)
    → message_sending hook (raw text with tags)
    → sanitizeForPlainText (strip HTML tags)
    → sendSignalTextChunks → sendMessageSignal
    → (no "[signal] delivered reply" log from this path)
```

Used for: heartbeat replies, followup queue, cross-channel routing.

**Key difference:** Path B has no `[signal] delivered reply` log entry. If you see `[followup-trace] sending N payloads` without a corresponding `[signal] delivered reply`, the message was sent via Path B. Look for `regex-replace: replaced/no match` immediately after the followup-trace to confirm hook execution.

### Path C: Discord `deliverDiscordReply` (direct Discord reply)

```
Discord monitor receives message
  → getReplyFromConfig → agent run → payloads
  → deliverDiscordReply (src/discord/monitor/reply-delivery.ts)
    → message_sending hook (raw text)
    → convertMarkdownTables → chunkDiscordTextWithMode
    → sendMessageDiscord / sendWebhookMessageDiscord
```

Note: The `message_sending` hook was missing from Discord's delivery path until the 2026-03-06 rebase fix. If disclaimer content leaks on Discord, verify this hook is present.
