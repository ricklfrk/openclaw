# Non-conforming output 處理說明（enforceFinalTag 模式）

## 策略

所有非合規輸出都會從 session 回滾（`branchWithSummary`），model 看不到自己的 wrong output。

### 格式提示（Format Hint）

首次非合規失敗後，在 system prompt 注入一次性格式提示（`NON_CONFORMING_FORMAT_HINT`），告訴 model：

- Tool call 的正確格式（native functionCall，不包 `<think>`）
- Reply 的正確格式（`<think>` + `<final>`）
- 禁止項（`[Historical tool call: ...]`、thinking-only、無 `<final>` 的回覆）

此提示只注入一次，之後所有 retry（同 profile 或換 profile）都帶著這個提示。

### Key / Profile 升級

| 階段                | 條件                  | 動作                                                       |
| ------------------- | --------------------- | ---------------------------------------------------------- |
| 首次失敗            | `!formatHintInjected` | `fail` skipProfile=false → 注入 hint → **同 profile 再試** |
| 同 profile 再失敗   | `formatHintInjected`  | `fail` skipProfile=true → **換 profile**                   |
| 換 profile 後再失敗 | `formatHintInjected`  | `fail` skipProfile=true → **繼續換**                       |
| 所有 profile 耗盡   | —                     | FailoverError → model-level fallback                       |

Non-conforming text 額外：

- 每次失敗都會保留一份乾淨的 stripped raw text 作為 deferred fallback 候選
- 但**不會**提早送出
- 只有在這一輪所有 profile 都跑完、再也沒有可試 key/profile 時，才會把最後一份 deferred fallback text 送給用戶

### Session 回滾

`attempt.ts` 在 return 前偵測非合規輸出 → `branchWithSummary()` 回滾 bad assistant + user message。
`branchWithSummary()` 持久化到 session 檔案，下次 attempt 從乾淨 save point 開始。

## 程式位置

- 判定 + 格式提示：`src/agents/pi-embedded-runner/run/non-conforming-retry.ts`
- Session 回滾：`src/agents/pi-embedded-runner/run/attempt.ts`
- 主迴圈 + hint 注入：`src/agents/pi-embedded-runner/run.ts`
