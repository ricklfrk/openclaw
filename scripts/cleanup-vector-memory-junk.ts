/**
 * One-off maintenance script: delete low-signal "image-placeholder" entities
 * that the vector-memory autoCapture accidentally stored when a user message
 * contained only a media breadcrumb plus a very short caption.
 *
 * Motivating example (2026-04-26 incident):
 *   text = "[image] 你選3間 [image data removed]"
 *   text = "[image] 你選3間"
 * These carry no useful semantic content but take up real recall slots, and
 * (before the media-ref neutralization fix) they contributed to the
 * "recalled media path re-attached as live multimodal input" leak.
 *
 * Usage (dry-run, prints candidates but keeps them):
 *   pnpm tsx scripts/cleanup-vector-memory-junk.ts --agent main
 *
 * Apply:
 *   pnpm tsx scripts/cleanup-vector-memory-junk.ts --agent main --apply
 *
 * Options:
 *   --agent <id>     Agent to clean (default: main)
 *   --pattern <str>  Additional substring to match in text (case-insensitive).
 *                    Can be repeated. Defaults to the known junk pattern.
 *   --max-chars <n>  Only delete entries whose l0/text is <= this many chars
 *                    (belt-and-suspenders against wiping legitimate long
 *                    memories that happen to contain the pattern). Default 80.
 *   --apply          Actually delete; without this flag nothing is changed.
 */

import os from "node:os";
import path from "node:path";
import { connect } from "@lancedb/lancedb";
import { MemoryStore } from "../extensions/vector-memory/src/store.ts";

type Candidate = {
  id: string;
  text: string;
  l0: string;
  chars: number;
  timestamp: number;
  reason: string;
};

function parseArg(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  if (idx < 0) {
    return undefined;
  }
  return process.argv[idx + 1];
}

function parseAllArgs(flag: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === flag && process.argv[i + 1]) {
      const value = process.argv[i + 1];
      if (value !== undefined) {
        out.push(value);
      }
    }
  }
  return out;
}

const JUNK_TEXT_PATTERNS: RegExp[] = [
  /^\[image\]\s*[^A-Za-z0-9\u4e00-\u9fff]*$/,
  /^\[image\]\s+.{0,40}\[image data removed\]\s*$/i,
];

function looksLikeJunk(text: string, l0: string, maxChars: number): string | null {
  const body = (l0 || text || "").trim();
  if (!body) {
    return null;
  }
  if (body.length > maxChars) {
    return null;
  }

  for (const re of JUNK_TEXT_PATTERNS) {
    if (re.test(body)) {
      return `matched junk pattern ${re}`;
    }
  }

  if (body.startsWith("[image]") && body.length <= maxChars) {
    const payload = body.replace(/^\[image\]\s*/, "").replace(/\[image data removed\]\s*$/i, "");
    if (payload.trim().length <= 10) {
      return `short [image] entity with ≤10 char payload ("${payload.trim()}")`;
    }
  }
  return null;
}

async function main() {
  const agent = parseArg("--agent") ?? "main";
  const apply = process.argv.includes("--apply");
  const maxChars = Number(parseArg("--max-chars") ?? "80");
  const extraPatterns = parseAllArgs("--pattern").map(
    (p) => new RegExp(p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"),
  );

  const base = path.join(os.homedir(), ".openclaw", "memory", "vector-memory");
  const storePath = path.join(base, agent);
  console.log(`[cleanup] agent=${agent} store=${storePath} apply=${apply} maxChars=${maxChars}`);

  // Auto-detect vector dimension from the existing table rather than hard-coding,
  // since embedding config varies per deployment (gemini-embedding-2-preview can
  // be 768 / 1536 / 3072 depending on config.dimensions).
  const db = await connect(storePath);
  const tbl = await db.openTable("memories");
  const sample = await tbl.query().limit(1).toArray();
  const vectorDim = sample.length > 0 ? (sample[0].vector as number[]).length : 768;
  console.log(`[cleanup] detected vectorDim=${vectorDim}`);

  const store = new MemoryStore({ dbPath: storePath, vectorDim });

  const candidates: Candidate[] = [];
  let scanned = 0;
  for await (const batch of store.iterateAll({ batchSize: 500, includeInvalidated: false })) {
    for (const entry of batch) {
      scanned++;
      let l0 = "";
      try {
        const meta = JSON.parse(entry.metadata);
        l0 = (meta.l0_abstract as string) || "";
      } catch {
        /* fall through */
      }
      const body = (l0 || entry.text || "").trim();

      let reason = looksLikeJunk(entry.text, l0, maxChars);
      if (!reason) {
        for (const re of extraPatterns) {
          if (re.test(body)) {
            reason = `matched user pattern ${re}`;
            break;
          }
        }
      }
      if (reason) {
        candidates.push({
          id: entry.id,
          text: entry.text.slice(0, 120),
          l0: l0.slice(0, 120),
          chars: body.length,
          timestamp: entry.timestamp,
          reason,
        });
      }
    }
  }

  console.log(`[cleanup] scanned=${scanned} candidates=${candidates.length}`);
  for (const c of candidates) {
    const date = new Date(c.timestamp).toISOString();
    console.log(
      `  - id=${c.id} chars=${c.chars} ts=${date}\n    reason: ${c.reason}\n    l0: ${JSON.stringify(c.l0)}\n    text: ${JSON.stringify(c.text)}`,
    );
  }

  if (!apply) {
    console.log("[cleanup] dry-run (no --apply flag). Rerun with --apply to delete.");
    return;
  }

  let deleted = 0;
  for (const c of candidates) {
    const ok = await store.deleteById(c.id);
    if (ok) {
      deleted++;
    } else {
      console.warn(`[cleanup] delete failed for ${c.id}`);
    }
  }
  console.log(`[cleanup] deleted=${deleted}/${candidates.length}`);
}

main().catch((err) => {
  console.error("[cleanup] fatal:", err);
  process.exit(1);
});
