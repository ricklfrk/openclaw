/**
 * Query preprocessing for autoRecall.
 *
 * Real user messages reaching `before_prompt_build` are noisy:
 *   - YouTube/web URLs that dominate the embedding signal
 *   - Envelope metadata blocks (`Conversation info` / `Sender` JSON fences)
 *   - Tool result fragments leaking through
 *   - Mixed alpha+digit tokens like `POCKET4` that don't lexically match
 *     stored variants like `Pocket 4`
 *
 * Helpers here:
 *   - `cleanRecallQuery`         strip URLs / code fences / envelope JSON,
 *                                NFKC normalize, collapse whitespace.
 *                                Used as the embedding-side query.
 *   - `expandKeywordVariants`    extract identifier-shaped tokens and emit
 *                                alpha/digit-split variants (POCKET4 →
 *                                ["pocket4", "pocket 4", "pocket", "4"]).
 *   - `buildAugmentedBm25Query`  prepend variants to the cleaned query so
 *                                LanceDB FTS / BM25 search sees both forms.
 */

const URL_RE = /\bhttps?:\/\/\S+/gi;
const FENCED_CODE_RE = /```[\s\S]*?```/g;

// Envelope blocks like:
//   Conversation info (untrusted metadata):
//   ```json
//   { ... }
//   ```
// Already covered by FENCED_CODE_RE; the label line itself is also stripped
// to avoid leaving "Conversation info (untrusted metadata):" as bare noise.
const ENVELOPE_LABEL_RE = /^[ \t]*[A-Za-z][A-Za-z _-]*\(untrusted metadata\):[ \t]*$/gm;

// `<relevant-memories>...</relevant-memories>` and similar plugin-injected
// blocks. Callers usually pre-strip via `stripPluginInjections`, but be
// defensive in case the query came from a raw `event.prompt`.
const RELEVANT_MEMORIES_RE = /<relevant-memories>[\s\S]*?<\/relevant-memories>/g;

// System markers like `System: __openclaw_memory_core_short_term_promotion_dream__`
// — pure breadcrumbs, no semantic value.
const SYSTEM_MARKER_RE = /^System:\s*__[a-z0-9_]+__\s*$/gim;

/**
 * Clean a raw user query before sending to the embedder.
 *
 * The goal is to keep the user's actual intent and drop the boilerplate
 * that drowns short queries (e.g. "幫我看... 我查一下POCKET4哪裡有賣" wrapped
 * in a YouTube URL + 200 chars of envelope JSON).
 *
 * NOT lowercased — most multilingual embedders are case-aware and casing
 * carries weak but real signal. NFKC normalization is applied so half-width
 * variants and compatibility forms collapse.
 */
export function cleanRecallQuery(raw: string): string {
  if (!raw) {
    return "";
  }
  let q = raw;
  q = q.replace(RELEVANT_MEMORIES_RE, " ");
  q = q.replace(FENCED_CODE_RE, " ");
  q = q.replace(ENVELOPE_LABEL_RE, " ");
  q = q.replace(SYSTEM_MARKER_RE, " ");
  q = q.replace(URL_RE, " ");
  q = q.normalize("NFKC");
  q = q.replace(/\s+/g, " ").trim();
  return q;
}

/**
 * Extract identifier-shaped Latin tokens from `text` and emit lexical
 * variants useful for BM25 keyword matching.
 *
 * Examples:
 *   "POCKET4 哪裡有賣"  → ["pocket4", "pocket 4", "pocket", "4"]
 *   "S24Ultra 螢幕"     → ["s24ultra", "s 24 ultra", "s", "24", "ultra"]
 *   "iPhone15Pro"       → ["iphone15pro", "i phone 15 pro", "phone", "15", "pro"]
 *   "DJI Pocket 4"      → ["dji", "pocket", "4"]   (already space-split, no extra variants)
 *
 * Variants are deduplicated and lowercased. Returns at most `maxVariants`
 * (default 24) entries to avoid blowing up the BM25 query.
 *
 * Pure CJK runs are left to the existing CJK-bigram tokenizer in
 * retriever.ts — we only operate on Latin alphanumerics here because that's
 * where the camel/Pascal/all-caps + digit ambiguity lives.
 */
export function expandKeywordVariants(text: string, maxVariants = 24): string[] {
  if (!text) {
    return [];
  }
  const out = new Set<string>();
  // Latin alphanumeric runs (also picks up "POCKET4", "S24Ultra", "v1.2.3" parts).
  const runs = text.match(/[A-Za-z][A-Za-z0-9]+/g) ?? [];
  for (const raw of runs) {
    if (raw.length < 2) {
      continue;
    }
    const lower = raw.toLowerCase();
    out.add(lower);
    // Split at:
    //   - letter ↔ digit boundary  (POCKET|4 / S|24|Ultra / iphone|15|pro)
    //   - lower ↔ upper boundary   (Pascal/Camel: iPhone → i|Phone)
    //   - upper-run ↔ Title-case   (XMLHttpRequest → XML|HttpRequest, then Http|Request)
    const split = raw
      .replace(/([A-Za-z])([0-9])/g, "$1 $2")
      .replace(/([0-9])([A-Za-z])/g, "$1 $2")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
      .toLowerCase();
    if (split !== lower) {
      out.add(split);
      for (const piece of split.split(/\s+/)) {
        // Keep multi-char pieces unconditionally; for single-char pieces
        // accept digits only (so "4" from "POCKET4" is kept as a useful
        // BM25 keyword, but lone letters like "i" / "s" — high-frequency
        // English words — are dropped).
        if (piece.length >= 2 || /^\d$/.test(piece)) {
          out.add(piece);
        }
      }
    }
  }
  return [...out].slice(0, maxVariants);
}

/**
 * Build a BM25-friendly query string by prepending variant keywords to
 * the cleaned base query.
 *
 * Variants come first so LanceDB's FTS analyzer treats them as high-frequency
 * matches against memories that store the canonical form (e.g. memory has
 * "Pocket 4" / "DJI Osmo Pocket 4", user wrote "POCKET4" — adding "pocket 4"
 * to the BM25 query lets FTS find the row).
 */
export function buildAugmentedBm25Query(cleanedQuery: string, variants: string[]): string {
  if (variants.length === 0) {
    return cleanedQuery;
  }
  const v = variants.join(" ");
  return cleanedQuery ? `${v} ${cleanedQuery}` : v;
}
