/**
 * Embedding-based Noise Prototype Bank
 *
 * Language-agnostic noise detection via cosine similarity against prototype
 * embeddings (recall queries, agent denials, greetings).
 *
 * Grows automatically when LLM extraction returns zero memories (feedback loop).
 * Adapted from memory-lancedb-pro.
 */

import type { Embedder } from "./embedder.js";

const BUILTIN_NOISE_TEXTS: readonly string[] = [
  "Do you remember what I told you?",
  "Can you recall my preferences?",
  "What did I say about that?",
  "你还记得我喜欢什么吗",
  "你知道我之前说过什么吗",
  "記得我上次提到的嗎",
  "我之前跟你说过吗",
  "I don't have any information about that",
  "I don't recall any previous conversation",
  "我没有相关的记忆",
  "Hello, how are you doing today?",
  "Hi there, what's up",
  "新的一天开始了",
];

const DEFAULT_THRESHOLD = 0.82;
const MAX_LEARNED_PROTOTYPES = 200;
const DEDUP_THRESHOLD = 0.95;

function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    return 0;
  }
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export class NoisePrototypeBank {
  private vectors: number[][] = [];
  private builtinCount = 0;
  private _initialized = false;
  private log: (msg: string) => void;

  constructor(log?: (msg: string) => void) {
    this.log = log ?? (() => {});
  }

  get initialized(): boolean {
    return this._initialized;
  }
  get size(): number {
    return this.vectors.length;
  }

  async init(embedder: Embedder): Promise<void> {
    if (this._initialized) {
      return;
    }

    // Batch embedding: single API call instead of 13 individual requests
    if (typeof embedder.embedBatch === "function") {
      try {
        const batchResults = await embedder.embedBatch([...BUILTIN_NOISE_TEXTS]);
        for (const v of batchResults) {
          if (v?.length) {
            this.vectors.push(v);
          }
        }
      } catch {
        // Batch failed, fall back to sequential
        for (const text of BUILTIN_NOISE_TEXTS) {
          try {
            const v = await embedder.embed(text);
            if (v?.length) {
              this.vectors.push(v);
            }
          } catch {}
        }
      }
    } else {
      for (const text of BUILTIN_NOISE_TEXTS) {
        try {
          const v = await embedder.embed(text);
          if (v?.length) {
            this.vectors.push(v);
          }
        } catch {}
      }
    }

    this.builtinCount = this.vectors.length;
    this._initialized = true;

    // Degeneracy check: if first two prototypes are near-identical, the embedding
    // model isn't discriminative enough — disable to avoid false positives.
    if (this.vectors.length >= 2 && cosine(this.vectors[0], this.vectors[1]) > 0.98) {
      this.log("vector-memory: noise-bank: degenerate embeddings, disabling");
      this._initialized = false;
      this.vectors = [];
      return;
    }

    this.log(`vector-memory: noise-bank: initialized with ${this.builtinCount} prototypes`);
  }

  isNoise(textVector: number[], threshold = DEFAULT_THRESHOLD): boolean {
    if (!this._initialized || this.vectors.length === 0) {
      return false;
    }
    for (const proto of this.vectors) {
      if (cosine(proto, textVector) >= threshold) {
        return true;
      }
    }
    return false;
  }

  /**
   * Feedback loop: add a vector when LLM extraction yields zero memories.
   * Deduplicates against existing prototypes (>= 0.95 cosine = skip).
   * Evicts oldest learned prototype when bank exceeds limit.
   */
  learn(textVector: number[]): void {
    if (!this._initialized) {
      return;
    }
    for (const proto of this.vectors) {
      if (cosine(proto, textVector) >= DEDUP_THRESHOLD) {
        return;
      }
    }
    this.vectors.push(textVector);
    if (this.vectors.length > this.builtinCount + MAX_LEARNED_PROTOTYPES) {
      this.vectors.splice(this.builtinCount, 1);
    }
    this.log(`vector-memory: noise-bank: learned (total: ${this.vectors.length})`);
  }
}
