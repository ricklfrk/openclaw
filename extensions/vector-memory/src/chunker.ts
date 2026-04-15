/**
 * Long Context Chunking — splits documents exceeding embedding context limits.
 * From memory-lancedb-pro.
 */

export interface ChunkResult {
  chunks: string[];
  totalOriginalLength: number;
  chunkCount: number;
}

export interface ChunkerConfig {
  maxChunkSize: number;
  overlapSize: number;
  minChunkSize: number;
  semanticSplit: boolean;
  maxLinesPerChunk: number;
}

export const EMBEDDING_CONTEXT_LIMITS: Record<string, number> = {
  "jina-embeddings-v5-text-small": 8192,
  "jina-embeddings-v5-text-nano": 8192,
  "text-embedding-3-small": 8192,
  "text-embedding-3-large": 8192,
  "text-embedding-004": 8192,
  "gemini-embedding-001": 2048,
  "gemini-embedding-2-preview": 8192,
  "nomic-embed-text": 8192,
  "all-MiniLM-L6-v2": 512,
  "all-mpnet-base-v2": 512,
};

const DEFAULT_CHUNKER_CONFIG: ChunkerConfig = {
  maxChunkSize: 4000,
  overlapSize: 200,
  minChunkSize: 200,
  semanticSplit: true,
  maxLinesPerChunk: 50,
};

const SENTENCE_ENDING = /[.!?。！？]/;
const CJK_RE = /[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]/;

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function countLines(s: string): number {
  return s.split(/\r\n|\n|\r/).length;
}

function findSplitEnd(
  text: string,
  start: number,
  maxEnd: number,
  minEnd: number,
  config: ChunkerConfig,
): number {
  const safeMinEnd = clamp(minEnd, start + 1, maxEnd);
  const safeMaxEnd = clamp(maxEnd, safeMinEnd, text.length);

  if (config.maxLinesPerChunk > 0) {
    const candidate = text.slice(start, safeMaxEnd);
    if (countLines(candidate) > config.maxLinesPerChunk) {
      let breaks = 0;
      for (let i = start; i < safeMaxEnd; i++) {
        if (text[i] === "\n") {
          breaks++;
          if (breaks >= config.maxLinesPerChunk) {
            return Math.max(i + 1, safeMinEnd);
          }
        }
      }
    }
  }

  if (config.semanticSplit) {
    for (let i = safeMaxEnd - 1; i >= safeMinEnd; i--) {
      if (SENTENCE_ENDING.test(text[i])) {
        let j = i + 1;
        while (j < safeMaxEnd && /\s/.test(text[j])) {
          j++;
        }
        return j;
      }
    }
    for (let i = safeMaxEnd - 1; i >= safeMinEnd; i--) {
      if (text[i] === "\n") {
        return i + 1;
      }
    }
  }

  for (let i = safeMaxEnd - 1; i >= safeMinEnd; i--) {
    if (/\s/.test(text[i])) {
      return i;
    }
  }

  return safeMaxEnd;
}

function chunkDocument(text: string, config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG): ChunkResult {
  if (!text || text.trim().length === 0) {
    return { chunks: [], totalOriginalLength: 0, chunkCount: 0 };
  }

  const totalOriginalLength = text.length;
  const chunks: string[] = [];
  let pos = 0;
  const maxGuard = Math.max(
    4,
    Math.ceil(text.length / Math.max(1, config.maxChunkSize - config.overlapSize)) + 5,
  );
  let guard = 0;

  while (pos < text.length && guard < maxGuard) {
    guard++;
    const remaining = text.length - pos;
    if (remaining <= config.maxChunkSize) {
      const chunk = text.slice(pos).trim();
      if (chunk.length > 0) {
        chunks.push(chunk);
      }
      break;
    }

    const maxEnd = Math.min(pos + config.maxChunkSize, text.length);
    const minEnd = Math.min(pos + config.minChunkSize, maxEnd);
    const end = findSplitEnd(text, pos, maxEnd, minEnd, config);
    const chunk = text.slice(pos, end).trim();

    if (chunk.length < config.minChunkSize) {
      const hardEnd = Math.min(pos + config.maxChunkSize, text.length);
      const hard = text.slice(pos, hardEnd).trim();
      if (hard.length > 0) {
        chunks.push(hard);
      }
      if (hardEnd >= text.length) {
        break;
      }
      pos = Math.max(hardEnd - config.overlapSize, pos + 1);
      continue;
    }

    chunks.push(chunk);
    if (end >= text.length) {
      break;
    }
    pos = Math.max(end - config.overlapSize, pos + 1);
  }

  return { chunks, totalOriginalLength, chunkCount: chunks.length };
}

function getCjkRatio(text: string): number {
  let cjk = 0;
  let total = 0;
  for (const ch of text) {
    if (/\s/.test(ch)) {
      continue;
    }
    total++;
    if (CJK_RE.test(ch)) {
      cjk++;
    }
  }
  return total === 0 ? 0 : cjk / total;
}

const CJK_CHAR_TOKEN_DIVISOR = 2.5;
const CJK_RATIO_THRESHOLD = 0.3;

export function smartChunk(text: string, embedderModel?: string): ChunkResult {
  const limit = embedderModel ? EMBEDDING_CONTEXT_LIMITS[embedderModel] : undefined;
  const base = limit ?? 8192;
  const cjkHeavy = getCjkRatio(text) > CJK_RATIO_THRESHOLD;
  const divisor = cjkHeavy ? CJK_CHAR_TOKEN_DIVISOR : 1;

  const config: ChunkerConfig = {
    maxChunkSize: Math.max(200, Math.floor((base * 0.7) / divisor)),
    overlapSize: Math.max(0, Math.floor((base * 0.05) / divisor)),
    minChunkSize: Math.max(100, Math.floor((base * 0.1) / divisor)),
    semanticSplit: true,
    maxLinesPerChunk: 50,
  };

  return chunkDocument(text, config);
}
