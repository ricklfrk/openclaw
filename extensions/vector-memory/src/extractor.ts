/**
 * Smart Memory Extractor — simplified extraction pipeline.
 * Pipeline: conversation → LLM extract → dedup → persist.
 * No admission control, no workspace boundary, no noise prototypes.
 */

import {
  type CandidateMemory,
  type ExtractionStats,
  ALWAYS_MERGE_CATEGORIES,
  MERGE_SUPPORTED_CATEGORIES,
  TEMPORAL_VERSIONED_CATEGORIES,
  normalizeCategory,
  mapToStoreCategory,
  getDefaultImportance,
} from "./categories.js";
import type { Embedder } from "./embedder.js";
import { buildExtractionPrompt, buildDedupPrompt, buildMergePrompt } from "./extraction-prompts.js";
import type { LlmClient } from "./llm-client.js";
import { isNoise } from "./noise-filter.js";
import type { MemoryStore, MemorySearchResult } from "./store.js";

// ============================================================================
// Envelope Metadata Stripping
// ============================================================================

export function stripEnvelopeMetadata(text: string): string {
  let cleaned = text.replace(/^System:\s*\[[\d\-: +GMT]+\]\s+\S+\[.*?\].*$/gm, "");
  cleaned = cleaned.replace(
    /(?:Conversation info|Sender|Replied message)\s*\(untrusted[^)]*\):\s*```json\s*\{[\s\S]*?\}\s*```/g,
    "",
  );
  cleaned = cleaned.replace(
    /```json\s*\{[^}]*"message_id"\s*:[^}]*"sender_id"\s*:[^}]*\}\s*```/g,
    "",
  );
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n");
  return cleaned.trim();
}

// ============================================================================
// Constants
// ============================================================================

const SIMILARITY_THRESHOLD = 0.7;
const MAX_SIMILAR_FOR_PROMPT = 3;
const MAX_MEMORIES_PER_EXTRACTION = 8;
const VALID_DECISIONS = new Set(["create", "merge", "skip", "supersede"]);

// ============================================================================
// Smart Extractor
// ============================================================================

export interface SmartExtractorConfig {
  user?: string;
  extractMinMessages?: number;
  extractMaxChars?: number;
  log?: (msg: string) => void;
  admissionController?: import("./admission-control.js").AdmissionController;
  conversationText?: string;
}

export class SmartExtractor {
  private log: (msg: string) => void;

  constructor(
    private store: MemoryStore,
    private embedder: Embedder,
    private llm: LlmClient,
    private config: SmartExtractorConfig = {},
  ) {
    this.log = config.log ?? ((msg: string) => console.log(msg));
  }

  async extractAndPersist(
    conversationText: string,
    sessionKey = "unknown",
  ): Promise<ExtractionStats> {
    const stats: ExtractionStats = { created: 0, merged: 0, skipped: 0 };

    const candidates = await this.extractCandidates(conversationText);
    if (candidates.length === 0) {
      this.log("vector-memory: extractor: no memories extracted");
      return stats;
    }

    this.log(`vector-memory: extractor: extracted ${candidates.length} candidate(s)`);

    for (const candidate of candidates.slice(0, MAX_MEMORIES_PER_EXTRACTION)) {
      try {
        await this.processCandidate(candidate, sessionKey, stats);
      } catch (err) {
        this.log(`vector-memory: extractor: failed [${candidate.category}]: ${String(err)}`);
      }
    }

    return stats;
  }

  // --------------------------------------------------------------------------
  // Step 1: LLM Extraction
  // --------------------------------------------------------------------------

  private async extractCandidates(conversationText: string): Promise<CandidateMemory[]> {
    const maxChars = this.config.extractMaxChars ?? 12000;
    const truncated =
      conversationText.length > maxChars ? conversationText.slice(-maxChars) : conversationText;
    const cleaned = stripEnvelopeMetadata(truncated);
    const user = this.config.user ?? "User";
    const prompt = buildExtractionPrompt(cleaned, user);

    const result = await this.llm.completeJson<{
      memories: Array<{
        category: string;
        abstract: string;
        overview: string;
        content: string;
        entity_tags?: string[];
      }>;
    }>(prompt, "extract-candidates");

    if (!result?.memories || !Array.isArray(result.memories)) {
      return [];
    }

    const candidates: CandidateMemory[] = [];
    for (const raw of result.memories) {
      const category = normalizeCategory(raw.category ?? "");
      if (!category) {
        continue;
      }
      const abstract = (raw.abstract ?? "").trim();
      const overview = (raw.overview ?? "").trim();
      const content = (raw.content ?? "").trim();
      if (!abstract || abstract.length < 5) {
        continue;
      }
      if (isNoise(abstract)) {
        continue;
      }
      const entityTags = Array.isArray(raw.entity_tags)
        ? raw.entity_tags
            .filter((t): t is string => typeof t === "string" && t.trim().length > 0)
            .map((t) => t.trim().toLowerCase())
            .slice(0, 5)
        : undefined;
      candidates.push({ category, abstract, overview, content, entityTags });
    }

    return candidates;
  }

  // --------------------------------------------------------------------------
  // Step 2: Dedup + Persist
  // --------------------------------------------------------------------------

  private async processCandidate(
    candidate: CandidateMemory,
    sessionKey: string,
    stats: ExtractionStats,
  ): Promise<void> {
    // Profile always merges
    if (ALWAYS_MERGE_CATEGORIES.has(candidate.category)) {
      const result = await this.handleProfileMerge(candidate, sessionKey);
      if (result === "created") {
        stats.created++;
      } else {
        stats.merged++;
      }
      return;
    }

    // Embed candidate
    const embeddingText = `${candidate.abstract} ${candidate.content}`;
    const vector = await this.embedder.embed(embeddingText);

    // Admission control gate (if enabled)
    if (this.config.admissionController && this.config.conversationText) {
      try {
        const admission = await this.config.admissionController.evaluate({
          candidateCategory: candidate.category,
          candidateText: `${candidate.abstract} ${candidate.content}`,
          candidateVector: vector,
          conversationText: this.config.conversationText,
        });
        if (admission.decision === "reject") {
          this.log(
            `vector-memory: extractor: admission rejected [${candidate.category}] ${candidate.abstract.slice(0, 60)} — ${admission.reason}`,
          );
          stats.skipped++;
          return;
        }
      } catch (err) {
        this.log(`vector-memory: extractor: admission check failed, continuing: ${String(err)}`);
      }
    }

    // Dedup
    const dedup = await this.deduplicate(candidate, vector);

    switch (dedup.decision) {
      case "create":
        await this.storeCandidate(candidate, vector, sessionKey);
        stats.created++;
        break;

      case "merge":
        if (dedup.matchId && MERGE_SUPPORTED_CATEGORIES.has(candidate.category)) {
          await this.handleMerge(candidate, dedup.matchId);
          stats.merged++;
        } else {
          await this.storeCandidate(candidate, vector, sessionKey);
          stats.created++;
        }
        break;

      case "supersede":
        if (dedup.matchId && TEMPORAL_VERSIONED_CATEGORIES.has(candidate.category)) {
          await this.handleSupersede(candidate, vector, dedup.matchId, sessionKey);
          stats.created++;
          stats.superseded = (stats.superseded ?? 0) + 1;
        } else {
          await this.storeCandidate(candidate, vector, sessionKey);
          stats.created++;
        }
        break;

      case "skip":
        this.log(
          `vector-memory: extractor: skipped [${candidate.category}] ${candidate.abstract.slice(0, 60)}`,
        );
        stats.skipped++;
        break;
    }
  }

  // --------------------------------------------------------------------------
  // Dedup Pipeline
  // --------------------------------------------------------------------------

  private async deduplicate(
    candidate: CandidateMemory,
    candidateVector: number[],
  ): Promise<{ decision: string; reason: string; matchId?: string }> {
    const similar = await this.store.vectorSearch(candidateVector, 5, SIMILARITY_THRESHOLD);
    if (similar.length === 0) {
      return { decision: "create", reason: "No similar memories found" };
    }

    return this.llmDedupDecision(candidate, similar);
  }

  private async llmDedupDecision(
    candidate: CandidateMemory,
    similar: MemorySearchResult[],
  ): Promise<{ decision: string; reason: string; matchId?: string }> {
    const topSimilar = similar.slice(0, MAX_SIMILAR_FOR_PROMPT);
    const existingFormatted = topSimilar
      .map((r, i) => {
        let metaObj: Record<string, unknown> = {};
        try {
          metaObj = JSON.parse(r.entry.metadata || "{}");
        } catch {}
        const abstract = (metaObj.l0_abstract as string) || r.entry.text;
        const overview = (metaObj.l1_overview as string) || "";
        return `${i + 1}. [${(metaObj.memory_category as string) || r.entry.category}] ${abstract}\n   Overview: ${overview}\n   Score: ${r.score.toFixed(3)}`;
      })
      .join("\n");

    const prompt = buildDedupPrompt(
      candidate.abstract,
      candidate.overview,
      candidate.content,
      existingFormatted,
    );

    try {
      const data = await this.llm.completeJson<{
        decision: string;
        reason: string;
        match_index?: number;
      }>(prompt, "dedup");
      if (!data) {
        return { decision: "create", reason: "LLM response unparseable" };
      }

      const decision = data.decision?.toLowerCase() ?? "create";
      if (!VALID_DECISIONS.has(decision)) {
        return { decision: "create", reason: `Unknown: ${data.decision}` };
      }

      const idx = data.match_index;
      const hasValidIndex = typeof idx === "number" && idx >= 1 && idx <= topSimilar.length;
      const matchEntry = hasValidIndex ? topSimilar[idx - 1] : topSimilar[0];

      // For destructive decisions, require valid match_index
      if (decision === "supersede" && !hasValidIndex) {
        return { decision: "create", reason: `${decision} degraded: missing match_index` };
      }

      return {
        decision,
        reason: data.reason ?? "",
        matchId: ["merge", "supersede"].includes(decision) ? matchEntry?.entry.id : undefined,
      };
    } catch (err) {
      return { decision: "create", reason: `LLM failed: ${String(err)}` };
    }
  }

  // --------------------------------------------------------------------------
  // Merge & Supersede
  // --------------------------------------------------------------------------

  private async handleProfileMerge(
    candidate: CandidateMemory,
    sessionKey: string,
  ): Promise<"merged" | "created"> {
    const vector = await this.embedder.embed(`${candidate.abstract} ${candidate.content}`);
    const existing = await this.store.vectorSearch(vector, 1, 0.3);
    const profileMatch = existing.find((r) => {
      try {
        return JSON.parse(r.entry.metadata || "{}").memory_category === "profile";
      } catch {
        return false;
      }
    });

    if (profileMatch) {
      await this.handleMerge(candidate, profileMatch.entry.id);
      return "merged";
    }
    await this.storeCandidate(candidate, vector, sessionKey);
    return "created";
  }

  private async handleMerge(candidate: CandidateMemory, matchId: string): Promise<void> {
    const existing = await this.store.getById(matchId);
    if (!existing) {
      return;
    }

    let existingAbstract = existing.text;
    let existingOverview = "";
    let existingContent = existing.text;
    try {
      const meta = JSON.parse(existing.metadata || "{}");
      existingAbstract = meta.l0_abstract || existing.text;
      existingOverview = meta.l1_overview || "";
      existingContent = meta.l2_content || existing.text;
    } catch {}

    const prompt = buildMergePrompt(
      existingAbstract,
      existingOverview,
      existingContent,
      candidate.abstract,
      candidate.overview,
      candidate.content,
      candidate.category,
    );
    const merged = await this.llm.completeJson<{
      abstract: string;
      overview: string;
      content: string;
    }>(prompt, "merge");
    if (!merged) {
      return;
    }

    const newVector = await this.embedder.embed(`${merged.abstract} ${merged.content}`);
    const metadata = JSON.stringify({
      l0_abstract: merged.abstract,
      l1_overview: merged.overview,
      l2_content: merged.content,
      memory_category: candidate.category,
    });

    await this.store.update(matchId, { text: merged.abstract, vector: newVector, metadata });
    this.log(
      `vector-memory: extractor: merged [${candidate.category}] into ${matchId.slice(0, 8)}`,
    );
  }

  private async handleSupersede(
    candidate: CandidateMemory,
    vector: number[],
    matchId: string,
    sessionKey: string,
  ): Promise<void> {
    const existing = await this.store.getById(matchId);
    if (!existing) {
      await this.storeCandidate(candidate, vector, sessionKey);
      return;
    }

    // Create new entry
    const storeCategory = mapToStoreCategory(candidate.category);
    const supersedeMetaObj: Record<string, unknown> = {
      l0_abstract: candidate.abstract,
      l1_overview: candidate.overview,
      l2_content: candidate.content,
      memory_category: candidate.category,
      source_session: sessionKey,
      supersedes: matchId,
    };
    if (candidate.entityTags && candidate.entityTags.length > 0) {
      supersedeMetaObj.entity_tags = candidate.entityTags;
    }
    const newEntry = await this.store.store({
      text: candidate.abstract,
      vector,
      category: storeCategory,
      importance: getDefaultImportance(candidate.category),
      metadata: JSON.stringify(supersedeMetaObj),
    });

    // Mark old entry as invalidated
    let existingMeta: Record<string, unknown> = {};
    try {
      existingMeta = JSON.parse(existing.metadata || "{}");
    } catch {}
    existingMeta.invalidated_at = Date.now();
    existingMeta.superseded_by = newEntry.id;
    await this.store.update(matchId, { metadata: JSON.stringify(existingMeta) });

    this.log(
      `vector-memory: extractor: superseded [${candidate.category}] ${matchId.slice(0, 8)} -> ${newEntry.id.slice(0, 8)}`,
    );
  }

  // --------------------------------------------------------------------------
  // Store Helper
  // --------------------------------------------------------------------------

  private async storeCandidate(
    candidate: CandidateMemory,
    vector: number[],
    sessionKey: string,
  ): Promise<void> {
    const storeCategory = mapToStoreCategory(candidate.category);
    const metaObj: Record<string, unknown> = {
      l0_abstract: candidate.abstract,
      l1_overview: candidate.overview,
      l2_content: candidate.content,
      memory_category: candidate.category,
      source_session: sessionKey,
    };
    if (candidate.entityTags && candidate.entityTags.length > 0) {
      metaObj.entity_tags = candidate.entityTags;
    }
    const metadata = JSON.stringify(metaObj);

    await this.store.store({
      text: candidate.abstract,
      vector,
      category: storeCategory,
      importance: getDefaultImportance(candidate.category),
      metadata,
    });

    this.log(
      `vector-memory: extractor: created [${candidate.category}] ${candidate.abstract.slice(0, 60)}`,
    );
  }
}
