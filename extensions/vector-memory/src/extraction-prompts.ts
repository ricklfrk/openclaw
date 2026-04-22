/**
 * Prompt templates for intelligent memory extraction.
 * From memory-lancedb-pro with minor simplification.
 */

export interface BuildExtractionPromptOptions {
  /**
   * When true, `user:` messages may come from multiple different humans
   * (group chat). The prompt instructs the LLM to identify each actual
   * speaker from conversation content rather than collapsing them into
   * one name.
   */
  multiUser?: boolean;
}

export function buildExtractionPrompt(
  conversationText: string,
  user: string,
  assistantName?: string,
  opts?: BuildExtractionPromptOptions,
): string {
  const currentDate = new Date().toISOString().split("T")[0];
  const aiName = assistantName?.trim() || "Assistant";
  const multiUser = opts?.multiUser === true;

  const rolesSection = multiUser
    ? `## Conversation Roles (Group Chat)

This is a **group chat**. \`user:\` messages may come from **multiple different humans** — do NOT collapse them into one person.

- \`user:\` messages → identify the actual speaker from conversation content (names, nicknames, @mentions, self-references, or metadata like "From: X"). If you cannot confidently attribute a line to a specific person, SKIP it rather than guessing.
- \`assistant:\` messages → all from **${aiName}** (the AI).

Attribution rules:
- Use each speaker's actual name (e.g. "Alice 提出 X", "Bob 反對 Y", "${aiName} 回覆 Z"). Never use generic pronouns like "用戶", "助手", "the user", "someone".
- Each memory must be attributable to a specific named person. Group-wide statements without a clear speaker → SKIP.
- Put the speaker's name in \`entity_tags\` so the memory is retrievable later.
- Never write third-person narration ("the group discussed ...", "用戶們 talked about ...").`
    : `## Conversation Roles

- \`user:\` messages are from **${user}**.
- \`assistant:\` messages are from **${aiName}**.

Both speakers may say things worth remembering. The only hard rule is **correct attribution**:
- What ${user} said/did/prefers → write "${user} ..." (use their name).
- What ${aiName} said/promised/decided → write "${aiName} ..." (use the name).
- Never mix them up. Never describe one speaker's words as the other's.
- Never use generic pronouns like "用戶", "助手", "the user", "the assistant" — always use the actual names above.
- Never write third-person narration ("the user interacts with ...", "用戶與 X 保持高頻互動"). Use direct attribution ("${user} 偏好 X", "${aiName} 答應了 Y").`;

  return `Analyze the following session context and extract memories worth long-term preservation.

${multiUser ? `Mode: Group Chat (multiple human speakers)` : `User: ${user}`}
Assistant: ${aiName}
Current Date: ${currentDate}

${rolesSection}

## Language Rule

Memories MUST be written in the same language the conversation actually uses.
If the conversation is in Chinese, write the memory in Chinese. If English, write in English.
Do NOT translate or switch languages — preserve the original wording and nuance.

## Recent Conversation
${conversationText}

# Memory Extraction Criteria

## What is worth remembering?
- Personalized information: Information specific to a named speaker${multiUser ? "" : ` (${user} or ${aiName})`}, not general domain knowledge
- Long-term validity: Information that will still be useful in future sessions
- Specific and clear: Has concrete details, not vague generalizations

## What is NOT worth remembering?
- General knowledge that anyone would know
- System/platform metadata: message IDs, sender IDs, timestamps, channel info, JSON envelopes — these are infrastructure noise
- Temporary information: One-time questions or conversations
- Vague information: "${multiUser ? "Someone" : user} has questions about a feature" (no specific details)
- Tool output, error logs, or boilerplate
- Recall queries / meta-questions: "Do you remember X?", "你還記得X嗎?" — retrieval requests, NOT new info
- Degraded or incomplete references
- Role-confused extractions: if you cannot tell which speaker said it, skip rather than guess
- Third-person narration about either speaker — rewrite to direct attribution or skip${multiUser ? "\n- Unattributed group-wide statements with no clearly identifiable speaker" : ""}

# Memory Classification

| Question | Answer | Category |
|----------|--------|----------|
| Who is this person? | Identity, attributes | profile |
| What do they prefer? | Preferences, habits | preferences |
| What is this thing? | Person, project, organization | entities |
| What happened? | Decision, milestone | events |
| How was it solved? | Problem + solution | cases |
| What is the process? | Reusable steps | patterns |

## Precise Definition
**profile** - A named person's identity (static attributes). Test: "<Name> is..."
**preferences** - A named person's preferences (tendencies). Test: "<Name> prefers/likes..."
**entities** - Continuously existing nouns. Test: "XXX's state is..."
**events** - Things that happened. Test: "<Name> did/completed..."
**cases** - Problem + solution pairs. Test: Contains "problem -> solution"
**patterns** - Reusable processes. Test: Can be used in "similar situations"

# Three-Level Structure

**abstract (L0)**: One-liner index (your own summary). For events, ALWAYS prepend the date and the speaker's actual name (e.g. "2026-04-08: ${multiUser ? "Alice" : user} went to..."). Never use role labels.
**overview (L1)**: Structured Markdown summary (your own summary). Always refer to speakers by their actual names.
**content (L2)**: **Verbatim quotes** from the conversation that support this memory. Copy the relevant original text exactly — do NOT paraphrase or rewrite. Keep each line's original speaker label. Trim only irrelevant filler; preserve the original wording, code snippets, names, and numbers.

# Entity Tags

For each memory, extract a flat list of **entity tags** — proper nouns and key concepts that identify the *who*, *what*, and *where* of this memory. These tags enable precise retrieval later.

Tag rules:
- People: real names only (e.g. "John", "哥哥"), not pronouns
- Projects/products: exact name (e.g. "NeonLedger", "Alpha")
- Technologies: specific tools/languages (e.g. "Rust", "Docker", "LanceDB")
- Organizations: company/team names
- Locations: only if relevant (e.g. "屯門", "AWS us-east-1")
- Keep tags short (1-3 words each), lowercase-normalized
- Maximum 5 tags per memory
- Omit generic terms ("code", "project", "app") — tags must be *specific*

# Output Format

Return JSON:
{
  "memories": [
    {
      "category": "profile|preferences|entities|events|cases|patterns",
      "abstract": "One-line index",
      "overview": "Structured Markdown summary",
      "content": "Verbatim conversation excerpt(s)",
      "entity_tags": ["tag1", "tag2"]
    }
  ]
}

Notes:
- Language: follow the Language Rule above — write each memory in the same language the relevant conversation used. Do not translate.
- Only extract truly valuable personalized information
- If nothing worth recording, return {"memories": []}
- Maximum 5 memories per extraction
- Preferences should be aggregated by topic
- L2 content MUST be original text from the conversation, not your rewrite. A separate distillation process will compress it later.`;
}

export function buildDedupPrompt(
  candidateAbstract: string,
  candidateOverview: string,
  candidateContent: string,
  existingMemories: string,
): string {
  return `Determine how to handle this candidate memory.

**Candidate Memory**:
Abstract: ${candidateAbstract}
Overview: ${candidateOverview}
Content: ${candidateContent}

**Existing Similar Memories**:
${existingMemories}

Please decide:
- SKIP: Candidate duplicates existing memories, or contains LESS information than existing
- CREATE: Completely new information not covered by any existing memory
- MERGE: Candidate adds genuinely NEW details to an existing memory
- SUPERSEDE: Same mutable fact has changed over time (only for preferences/entities)

IMPORTANT:
- "events" and "cases" are independent records — only SKIP or CREATE.
- A candidate with less info than existing should always SKIP.
- For SUPERSEDE: existing "Preferred editor: VS Code", candidate "Preferred editor: Zed".

Return JSON:
{
  "decision": "skip|create|merge|supersede",
  "match_index": 1,
  "reason": "Decision reason"
}

- If decision is "merge"/"supersede", set "match_index" to the number of the existing memory (1-based).`;
}

export function buildMergePrompt(
  existingAbstract: string,
  existingOverview: string,
  existingContent: string,
  newAbstract: string,
  newOverview: string,
  newContent: string,
  category: string,
): string {
  return `Merge the following memory into a single coherent record with all three levels.

**Category**: ${category}

**Existing Memory:**
Abstract: ${existingAbstract}
Overview:
${existingOverview}
Content:
${existingContent}

**New Information:**
Abstract: ${newAbstract}
Overview:
${newOverview}
Content:
${newContent}

Requirements:
- Remove duplicate information
- Keep the most up-to-date details
- Maintain a coherent narrative
- Keep code identifiers / URIs / model names unchanged when they are proper nouns

Return JSON:
{
  "abstract": "Merged one-line abstract",
  "overview": "Merged structured Markdown overview",
  "content": "Merged full content"
}`;
}
