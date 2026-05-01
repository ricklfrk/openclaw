/**
 * Embedding Abstraction Layer — multi-provider, auto-chunking, key rotation.
 * Full version from memory-lancedb-pro with branding changes.
 */

import { createHash } from "node:crypto";
import OpenAI from "openai";
import { smartChunk } from "./chunker.js";

// ============================================================================
// Embedding Cache (LRU with TTL)
// ============================================================================

interface CacheEntry {
  vector: number[];
  createdAt: number;
}

class EmbeddingCache {
  private cache = new Map<string, CacheEntry>();
  private readonly maxSize: number;
  private readonly ttlMs: number;
  public hits = 0;
  public misses = 0;

  constructor(maxSize = 256, ttlMinutes = 30) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMinutes * 60_000;
  }

  private key(text: string, task?: string): string {
    return createHash("sha256")
      .update(`${task || ""}:${text}`)
      .digest("hex")
      .slice(0, 24);
  }

  get(text: string, task?: string): number[] | undefined {
    const k = this.key(text, task);
    const entry = this.cache.get(k);
    if (!entry) {
      this.misses++;
      return undefined;
    }
    if (Date.now() - entry.createdAt > this.ttlMs) {
      this.cache.delete(k);
      this.misses++;
      return undefined;
    }
    this.cache.delete(k);
    this.cache.set(k, entry);
    this.hits++;
    return entry.vector;
  }

  set(text: string, task: string | undefined, vector: number[]): void {
    const k = this.key(text, task);
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(k, { vector, createdAt: Date.now() });
  }
}

// ============================================================================
// Types & Configuration
// ============================================================================

export interface EmbeddingConfig {
  provider: "openai-compatible" | "azure-openai";
  apiVersion?: string;
  apiKey: string | string[];
  model: string;
  baseURL?: string;
  dimensions?: number;
  taskQuery?: string;
  taskPassage?: string;
  normalized?: boolean;
  chunking?: boolean;
}

type EmbeddingProviderProfile =
  | "openai"
  | "azure-openai"
  | "jina"
  | "voyage-compatible"
  | "generic-openai-compatible";

interface EmbeddingCapabilities {
  encoding_format: boolean;
  normalized: boolean;
  taskField: string | null;
  taskValueMap?: Record<string, string>;
  dimensionsField: string | null;
}

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-004": 768,
  "gemini-embedding-001": 3072,
  "gemini-embedding-2-preview": 3072,
  "nomic-embed-text": 768,
  "mxbai-embed-large": 1024,
  "BAAI/bge-m3": 1024,
  "all-MiniLM-L6-v2": 384,
  "all-mpnet-base-v2": 512,
  "jina-embeddings-v5-text-small": 1024,
  "jina-embeddings-v5-text-nano": 768,
  "voyage-4": 1024,
  "voyage-4-lite": 1024,
  "voyage-4-large": 1024,
  "voyage-3": 1024,
  "voyage-3-lite": 512,
  "voyage-3-large": 1024,
};

// ============================================================================
// Utility Functions
// ============================================================================

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function getErrorStatus(error: unknown): number | undefined {
  if (!error || typeof error !== "object") {
    return undefined;
  }
  const err = error as Record<string, unknown>;
  if (typeof err.status === "number") {
    return err.status;
  }
  return undefined;
}

function detectEmbeddingProviderProfile(
  baseURL: string | undefined,
  model: string,
): EmbeddingProviderProfile {
  const base = baseURL || "";
  if (/api\.openai\.com/i.test(base)) {
    return "openai";
  }
  if (/\.openai\.azure\.com/i.test(base)) {
    return "azure-openai";
  }
  if (/api\.jina\.ai/i.test(base) || /^jina-/i.test(model)) {
    return "jina";
  }
  if (/api\.voyageai\.com/i.test(base) || /^voyage\b/i.test(model)) {
    return "voyage-compatible";
  }
  return "generic-openai-compatible";
}

function getEmbeddingCapabilities(profile: EmbeddingProviderProfile): EmbeddingCapabilities {
  switch (profile) {
    case "openai":
      return {
        encoding_format: true,
        normalized: false,
        taskField: null,
        dimensionsField: "dimensions",
      };
    case "jina":
      return {
        encoding_format: true,
        normalized: true,
        taskField: "task",
        dimensionsField: "dimensions",
      };
    case "voyage-compatible":
      return {
        encoding_format: false,
        normalized: false,
        taskField: "input_type",
        taskValueMap: {
          "retrieval.query": "query",
          "retrieval.passage": "document",
          query: "query",
          document: "document",
        },
        dimensionsField: "output_dimension",
      };
    default:
      return {
        encoding_format: true,
        normalized: false,
        taskField: null,
        dimensionsField: "dimensions",
      };
  }
}

function isGeminiEmbedding2(model: string): boolean {
  return /^gemini-embedding-2/i.test(model);
}

/**
 * For gemini-embedding-2-preview, task type is specified via text prefix
 * rather than the taskType API parameter. See:
 * https://ai.google.dev/gemini-api/docs/embeddings
 */
function applyGeminiEmbedding2TaskPrefix(text: string, task: string | undefined): string {
  if (!task) {
    return text;
  }
  switch (task) {
    case "retrieval.query":
    case "query":
      return `task: search result | query: ${text}`;
    case "retrieval.passage":
    case "passage":
    case "document":
      return `title: none | text: ${text}`;
    case "question_answering":
      return `task: question answering | query: ${text}`;
    case "fact_checking":
      return `task: fact checking | query: ${text}`;
    case "code_retrieval":
      return `task: code retrieval | query: ${text}`;
    case "classification":
      return `task: classification | query: ${text}`;
    case "clustering":
      return `task: clustering | query: ${text}`;
    case "similarity":
      return `task: sentence similarity | query: ${text}`;
    default:
      return `task: search result | query: ${text}`;
  }
}

function l2Normalize(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) {
    norm += v * v;
  }
  norm = Math.sqrt(norm);
  if (norm === 0) {
    return vec;
  }
  return vec.map((v) => v / norm);
}

function isAuthError(error: unknown): boolean {
  const status = getErrorStatus(error);
  if (status === 401 || status === 403) {
    return true;
  }
  const msg = getErrorMessage(error);
  return /\b401\b|\b403\b|invalid api key|unauthorized|forbidden/i.test(msg);
}

function getProviderLabel(baseURL: string | undefined, model: string): string {
  const profile = detectEmbeddingProviderProfile(baseURL, model);
  if (profile === "jina") {
    return "Jina";
  }
  if (profile === "voyage-compatible") {
    return "Voyage";
  }
  if (profile === "openai") {
    return "OpenAI";
  }
  if (profile === "azure-openai") {
    return "Azure OpenAI";
  }
  if (baseURL) {
    try {
      return new URL(baseURL).host;
    } catch {
      return baseURL;
    }
  }
  return "embedding provider";
}

export function formatEmbeddingProviderError(
  error: unknown,
  opts: { baseURL?: string; model: string; mode?: "single" | "batch" },
): string {
  const raw = getErrorMessage(error).trim();
  const provider = getProviderLabel(opts.baseURL, opts.model);
  if (isAuthError(error)) {
    return `Embedding provider authentication failed (${raw}). Check embedding.apiKey for ${provider}.`;
  }
  const prefix =
    opts.mode === "batch"
      ? `Failed to generate batch embeddings from ${provider}: `
      : `Failed to generate embedding from ${provider}: `;
  return `${prefix}${raw}`;
}

// ============================================================================
// Safety Constants
// ============================================================================

const MAX_EMBED_DEPTH = 3;
const EMBED_TIMEOUT_MS = 10_000;
const STRICT_REDUCTION_FACTOR = 0.5;

export function getVectorDimensions(model: string, overrideDims?: number): number {
  if (overrideDims && overrideDims > 0) {
    return overrideDims;
  }
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported embedding model: ${model}. Set embedding.dimensions in config.`);
  }
  return dims;
}

// ============================================================================
// Embedder Class
// ============================================================================

export class Embedder {
  private clients: OpenAI[];
  private _clientIndex = 0;
  public readonly dimensions: number;
  private readonly _cache: EmbeddingCache;
  private readonly _model: string;
  private readonly _baseURL?: string;
  private readonly _taskQuery?: string;
  private readonly _taskPassage?: string;
  private readonly _normalized?: boolean;
  private readonly _capabilities: EmbeddingCapabilities;
  private readonly _requestDimensions?: number;
  private readonly _autoChunk: boolean;
  private readonly _isGeminiEmb2: boolean;
  private readonly _needsL2Norm: boolean;

  constructor(config: EmbeddingConfig) {
    const apiKeys = Array.isArray(config.apiKey) ? config.apiKey : [config.apiKey];
    const resolvedKeys = apiKeys.map((k) => resolveEnvVars(k));

    this._model = config.model;
    this._baseURL = config.baseURL;
    this._taskQuery = config.taskQuery;
    this._taskPassage = config.taskPassage;
    this._normalized = config.normalized;
    this._requestDimensions = config.dimensions;
    this._autoChunk = config.chunking !== false;
    const profile = detectEmbeddingProviderProfile(this._baseURL, this._model);
    this._capabilities = getEmbeddingCapabilities(profile);

    this.clients = resolvedKeys.map((key) => {
      let defaultHeaders: Record<string, string> = {};
      let baseURL = config.baseURL;
      if (config.provider === "azure-openai" || profile === "azure-openai") {
        defaultHeaders["api-key"] = key;
        if (baseURL && config.apiVersion) {
          const url = new URL(baseURL);
          url.searchParams.set("api-version", config.apiVersion);
          baseURL = url.toString();
        }
      }
      return new OpenAI({
        apiKey: key,
        ...(baseURL ? { baseURL } : {}),
        defaultHeaders: Object.keys(defaultHeaders).length > 0 ? defaultHeaders : undefined,
      });
    });

    this.dimensions = getVectorDimensions(config.model, config.dimensions);
    this._cache = new EmbeddingCache(256, 30);
    this._isGeminiEmb2 = isGeminiEmbedding2(config.model);
    // MRL truncated dimensions (not native 3072) need L2 normalization
    const nativeDim = EMBEDDING_DIMENSIONS[config.model];
    this._needsL2Norm = !!config.dimensions && !!nativeDim && config.dimensions < nativeDim;
  }

  private nextClient(): OpenAI {
    const client = this.clients[this._clientIndex % this.clients.length];
    this._clientIndex = (this._clientIndex + 1) % this.clients.length;
    return client;
  }

  private isRateLimitError(error: unknown): boolean {
    if (!error || typeof error !== "object") {
      return false;
    }
    const err = error as Record<string, unknown>;
    if (err.status === 429 || err.status === 503) {
      return true;
    }
    const msg =
      error instanceof Error ? error.message : typeof error === "string" ? error : "unknown";
    return /rate.limit|quota|too many requests|429|503.*overload/i.test(msg);
  }

  private async embedWithRetry(payload: unknown, signal?: AbortSignal): Promise<unknown> {
    const maxAttempts = this.clients.length;
    let lastError: Error | undefined;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const client = this.nextClient();
      try {
        return await client.embeddings.create(
          payload as Parameters<typeof client.embeddings.create>[0],
          signal ? { signal } : undefined,
        );
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          throw error;
        }
        lastError = error instanceof Error ? error : new Error(String(error));
        if (this.isRateLimitError(error) && attempt < maxAttempts - 1) {
          continue;
        }
        if (!this.isRateLimitError(error)) {
          throw error;
        }
      }
    }
    throw new Error(
      `All ${maxAttempts} API keys exhausted (rate limited). Last: ${lastError?.message ?? "unknown"}`,
      { cause: lastError },
    );
  }

  private withTimeout<T>(fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);
    return fn(controller.signal).finally(() => clearTimeout(timeoutId));
  }

  private buildPayload(input: string | string[], task?: string): Record<string, unknown> {
    // gemini-embedding-2: task goes in text prefix, not API parameter
    let effectiveInput = input;
    if (this._isGeminiEmb2 && task) {
      effectiveInput =
        typeof input === "string"
          ? applyGeminiEmbedding2TaskPrefix(input, task)
          : input.map((t) => applyGeminiEmbedding2TaskPrefix(t, task));
    }
    const payload: Record<string, unknown> = { model: this._model, input: effectiveInput };
    if (this._capabilities.encoding_format) {
      payload.encoding_format = "float";
    }
    if (this._capabilities.normalized && this._normalized !== undefined) {
      payload.normalized = this._normalized;
    }
    // For gemini-embedding-2, skip task parameter (already in text prefix)
    if (!this._isGeminiEmb2 && this._capabilities.taskField && task) {
      const value = this._capabilities.taskValueMap?.[task] ?? task;
      payload[this._capabilities.taskField] = value;
    }
    if (
      this._capabilities.dimensionsField &&
      this._requestDimensions &&
      this._requestDimensions > 0
    ) {
      payload[this._capabilities.dimensionsField] = this._requestDimensions;
    }
    return payload;
  }

  private validateEmbedding(embedding: number[]): void {
    if (!Array.isArray(embedding) || embedding.length !== this.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimensions}, got ${Array.isArray(embedding) ? embedding.length : "non-array"}`,
      );
    }
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.withTimeout((signal) => this.embedSingle(text, this._taskQuery, 0, signal));
  }

  async embedPassage(text: string): Promise<number[]> {
    return this.withTimeout((signal) => this.embedSingle(text, this._taskPassage, 0, signal));
  }

  async embed(text: string): Promise<number[]> {
    return this.embedPassage(text);
  }

  private async embedSingle(
    text: string,
    task?: string,
    depth = 0,
    signal?: AbortSignal,
  ): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot embed empty text");
    }

    if (depth >= MAX_EMBED_DEPTH) {
      const safeLimit = Math.floor(text.length * STRICT_REDUCTION_FACTOR);
      if (safeLimit < 100) {
        throw new Error("Failed to embed: input too large after retries");
      }
      text = text.slice(0, safeLimit);
    }

    const cached = this._cache.get(text, task);
    if (cached) {
      return cached;
    }

    try {
      const response = (await this.embedWithRetry(this.buildPayload(text, task), signal)) as {
        data: Array<{ embedding: number[] }>;
      };
      let embedding = response.data[0]?.embedding;
      if (!embedding) {
        throw new Error("No embedding returned from provider");
      }
      this.validateEmbedding(embedding);
      if (this._needsL2Norm) {
        embedding = l2Normalize(embedding);
      }
      this._cache.set(text, task, embedding);
      return embedding;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      const isContextError = /context|too long|exceed|length/i.test(errorMsg);

      if (isContextError && this._autoChunk) {
        const chunkResult = smartChunk(text, this._model);
        if (chunkResult.chunks.length === 0) {
          throw new Error(`Failed to chunk: ${errorMsg}`, { cause: error });
        }

        if (chunkResult.chunks.length === 1 && chunkResult.chunks[0].length > text.length * 0.9) {
          const safeLimit = Math.floor(text.length * STRICT_REDUCTION_FACTOR);
          if (safeLimit < 100) {
            throw new Error("Chunking couldn't reduce input enough", { cause: error });
          }
          return this.embedSingle(text.slice(0, safeLimit), task, depth + 1, signal);
        }

        const chunkEmbeddings = await Promise.all(
          chunkResult.chunks.map((chunk) => this.embedSingle(chunk, task, depth + 1, signal)),
        );

        const avgEmbedding = chunkEmbeddings.reduce(
          (sum, emb) => {
            for (let i = 0; i < emb.length; i++) {
              sum[i] += emb[i];
            }
            return sum;
          },
          Array.from({ length: this.dimensions }, () => 0),
        );
        const finalEmbedding = avgEmbedding.map((v: number) => v / chunkEmbeddings.length);
        this._cache.set(text, task, finalEmbedding);
        return finalEmbedding;
      }

      throw new Error(
        formatEmbeddingProviderError(error, {
          baseURL: this._baseURL,
          model: this._model,
          mode: "single",
        }),
        { cause: error },
      );
    }
  }

  /**
   * Batch embed multiple texts in a single API call.
   * Falls back to sequential embedding if batch fails.
   */
  async embedBatch(texts: string[], task?: string): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }
    if (texts.length === 1) {
      return [await this.embedSingle(texts[0], task)];
    }

    try {
      const response = (await this.withTimeout((signal) =>
        this.embedWithRetry(this.buildPayload(texts, task), signal),
      )) as { data: Array<{ embedding: number[] }> };

      const results: number[][] = [];
      for (const item of response.data) {
        let emb = item?.embedding;
        if (!emb || emb.length !== this.dimensions) {
          continue;
        }
        if (this._needsL2Norm) {
          emb = l2Normalize(emb);
        }
        results.push(emb);
      }
      if (results.length > 0) {
        return results;
      }
    } catch {
      // Fall through to sequential
    }

    const results: number[][] = [];
    for (const text of texts) {
      try {
        results.push(await this.embedSingle(text, task));
      } catch {}
    }
    return results;
  }

  get model(): string {
    return this._model;
  }
  get keyCount(): number {
    return this.clients.length;
  }
  get isMultimodal(): boolean {
    return this._isGeminiEmb2;
  }

  /**
   * Embed a media item (image, audio, video, PDF) via the native Gemini
   * embedContent API. Only supported for gemini-embedding-2-preview.
   *
   * Supported modalities (per Gemini docs):
   *   Image: PNG, JPEG (max 6 per request)
   *   Audio: MP3, WAV (max 80s)
   *   Video: MP4, MOV / H264, H265, AV1, VP9 (max 120s)
   *   PDF:   max 6 pages
   *
   * When caption is provided, text + media are sent as parts of a single
   * content item, producing one aggregated embedding for both modalities.
   */
  async embedMedia(
    data: Buffer | Uint8Array,
    mimeType: string,
    caption?: string,
  ): Promise<number[]> {
    if (!this._isGeminiEmb2) {
      throw new Error("Multimodal embedding is only supported with gemini-embedding-2-preview");
    }

    return this.withTimeout(async (signal) => {
      const base64Data =
        data instanceof Buffer ? data.toString("base64") : Buffer.from(data).toString("base64");

      // Build parts: optional caption text + media inline_data
      // Sent as a single content → produces one aggregated embedding
      const parts: Array<Record<string, unknown>> = [];
      if (caption) {
        parts.push({ text: caption });
      }
      parts.push({ inline_data: { mime_type: mimeType, data: base64Data } });

      const client = this.nextClient();
      const apiKey = client.apiKey;

      // Always use the native Gemini API endpoint for multimodal embedding.
      // The OpenAI-compatible baseURL (e.g. .../v1beta/openai) does NOT
      // support multimodal embedContent, so we derive the native path.
      let nativeBase: string;
      if (this._baseURL) {
        // Strip /openai or /openai/ suffix if present
        nativeBase = this._baseURL.replace(/\/openai\/?$/i, "").replace(/\/+$/, "");
      } else {
        nativeBase = "https://generativelanguage.googleapis.com/v1beta";
      }
      const url = `${nativeBase}/models/${this._model}:embedContent?key=${apiKey}`;

      const body: Record<string, unknown> = {
        content: { parts },
      };
      if (this._requestDimensions && this._requestDimensions > 0) {
        body.output_dimensionality = this._requestDimensions;
      }

      const controller = new AbortController();
      if (signal) {
        signal.addEventListener("abort", () => controller.abort());
      }

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Gemini embedContent API error ${response.status}: ${errText}`);
      }

      // Single content item → response.embedding.values (not embeddings[])
      const result = (await response.json()) as {
        embedding?: { values?: number[] };
      };
      let embedding = result.embedding?.values;
      if (!embedding || !Array.isArray(embedding)) {
        throw new Error("No embedding returned from Gemini multimodal API");
      }
      if (this._needsL2Norm) {
        embedding = l2Normalize(embedding);
      }
      return embedding;
    });
  }

  async test(): Promise<{ success: boolean; error?: string; dimensions?: number }> {
    try {
      const emb = await this.embedPassage("test");
      return { success: true, dimensions: emb.length };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

export function createEmbedder(config: EmbeddingConfig): Embedder {
  return new Embedder(config);
}
