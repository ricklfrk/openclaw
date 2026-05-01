import { beforeEach, describe, expect, it, vi } from "vitest";

// --- Mock @huggingface/transformers ----------------------------------------
//
// We never load the real ~580MB XLM-RoBERTa model in tests. Instead we fake
// the two `from_pretrained` factories + the model's call signature. Tests
// verify the singleton/lazy-load semantics and error-handling contracts
// that `retriever.ts` depends on.

const {
  mockFromPretrainedTokenizer,
  mockFromPretrainedModel,
  mockTokenizerCall,
  mockModelCall,
  mockEnv,
} = vi.hoisted(() => ({
  mockFromPretrainedTokenizer: vi.fn(),
  mockFromPretrainedModel: vi.fn(),
  mockTokenizerCall: vi.fn(),
  mockModelCall: vi.fn(),
  // Mutable shared mock for transformers.env — the reranker writes
  // cacheDir here; tests verify post-conditions via this object.
  mockEnv: { cacheDir: null as string | null },
}));

vi.mock("@huggingface/transformers", () => ({
  AutoTokenizer: { from_pretrained: mockFromPretrainedTokenizer },
  AutoModelForSequenceClassification: { from_pretrained: mockFromPretrainedModel },
  env: mockEnv,
}));

async function loadModule() {
  // Dynamic import so we pick up a fresh module cache after resetModules().
  return await import("./local-onnx-rerank.js");
}

beforeEach(async () => {
  vi.resetModules();
  mockFromPretrainedTokenizer.mockReset();
  mockFromPretrainedModel.mockReset();
  mockTokenizerCall.mockReset();
  mockModelCall.mockReset();
  mockEnv.cacheDir = null;

  // Default happy-path: tokenizer() returns an opaque "inputs" object, and
  // model(inputs) returns logits in the same order as the passages fed.
  mockTokenizerCall.mockImplementation(() => ({ __inputs: true }));
  mockFromPretrainedTokenizer.mockResolvedValue(mockTokenizerCall);
  mockFromPretrainedModel.mockImplementation(async () => mockModelCall);
});

describe("scoreQueryPassagePairs", () => {
  it("returns logits in the same order as passages", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.1, 0.9, -0.3] } });
    const mod = await loadModule();

    const scores = await mod.scoreQueryPassagePairs("query", ["p1", "p2", "p3"]);

    expect(scores).toEqual([0.1, 0.9, -0.3]);
    expect(mockFromPretrainedTokenizer).toHaveBeenCalledTimes(1);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);
    expect(mockFromPretrainedTokenizer).toHaveBeenCalledWith(mod.DEFAULT_LOCAL_ONNX_RERANK_MODEL);
  });

  it("short-circuits on empty passages without loading the model", async () => {
    const mod = await loadModule();

    const scores = await mod.scoreQueryPassagePairs("query", []);

    expect(scores).toEqual([]);
    expect(mockFromPretrainedTokenizer).not.toHaveBeenCalled();
    expect(mockFromPretrainedModel).not.toHaveBeenCalled();
  });

  it("reuses the loaded model across calls with the same modelId (singleton)", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.5] } });
    const mod = await loadModule();

    await mod.scoreQueryPassagePairs("q", ["a"]);
    await mod.scoreQueryPassagePairs("q", ["b"]);
    await mod.scoreQueryPassagePairs("q", ["c"]);

    expect(mockFromPretrainedTokenizer).toHaveBeenCalledTimes(1);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);
  });

  it("reloads when the modelId changes", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.5] } });
    const mod = await loadModule();

    await mod.scoreQueryPassagePairs("q", ["a"], { modelId: "model-a" });
    await mod.scoreQueryPassagePairs("q", ["b"], { modelId: "model-b" });

    expect(mockFromPretrainedTokenizer).toHaveBeenCalledTimes(2);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(2);
    expect(mockFromPretrainedTokenizer).toHaveBeenNthCalledWith(1, "model-a");
    expect(mockFromPretrainedTokenizer).toHaveBeenNthCalledWith(2, "model-b");
  });

  it("concurrent first calls share one loader promise", async () => {
    // tsgo's flow analysis can't see cross-callback assignments, so hold
    // the resolver through a boxed ref to keep the call-site type-clean.
    const ref: { resolve: ((v: unknown) => void) | null } = { resolve: null };
    mockFromPretrainedModel.mockImplementation(
      () =>
        new Promise((resolve) => {
          ref.resolve = () => resolve(mockModelCall);
        }),
    );
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();

    const p1 = mod.scoreQueryPassagePairs("q", ["a"]);
    const p2 = mod.scoreQueryPassagePairs("q", ["b"]);
    const p3 = mod.scoreQueryPassagePairs("q", ["c"]);

    // Flush enough microtasks for Promise.all(tokenizer, model) to kick off.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);

    ref.resolve?.(undefined);
    await Promise.all([p1, p2, p3]);

    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);
  });

  it("resets loader cache on load failure so a subsequent call retries", async () => {
    mockFromPretrainedModel.mockRejectedValueOnce(new Error("fake load error"));
    const mod = await loadModule();

    await expect(mod.scoreQueryPassagePairs("q", ["a"])).rejects.toThrow("fake load error");

    mockFromPretrainedModel.mockResolvedValueOnce(mockModelCall);
    mockModelCall.mockResolvedValue({ logits: { data: [0.7] } });

    const scores = await mod.scoreQueryPassagePairs("q", ["a"]);
    expect(scores).toEqual([0.7]);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(2);
  });

  it("throws when the model returns a score count mismatch", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();

    await expect(mod.scoreQueryPassagePairs("q", ["a", "b", "c"])).rejects.toThrow(
      /expected 3 scores/,
    );
  });

  it("throws a descriptive error when the model returns no logits", async () => {
    mockModelCall.mockResolvedValue({});
    const mod = await loadModule();

    await expect(mod.scoreQueryPassagePairs("q", ["a"])).rejects.toThrow(/no logits/);
  });

  it("passes tokenizer arguments compatible with pair-mode cross-encoders", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();

    await mod.scoreQueryPassagePairs("my query", ["passage one"]);

    expect(mockTokenizerCall).toHaveBeenCalledTimes(1);
    const [queries, opts] = mockTokenizerCall.mock.calls[0];
    expect(queries).toEqual(["my query"]);
    expect(opts).toMatchObject({
      text_pair: ["passage one"],
      padding: true,
      truncation: true,
    });
    expect(typeof opts.max_length).toBe("number");
    expect(opts.max_length).toBeGreaterThan(0);
  });

  it("emits info logs around first model load", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();

    const logger = vi.fn<(level: "info" | "error", message: string) => void>();
    await mod.scoreQueryPassagePairs("q", ["a"], { logger });

    // Two info lines: "loading..." and "ready in <n>ms".
    const calls = logger.mock.calls as Array<[string, string]>;
    const infoMessages = calls.filter(([level]) => level === "info").map(([, msg]) => msg);
    expect(infoMessages.some((m) => m.includes("loading model"))).toBe(true);
    expect(infoMessages.some((m) => m.includes("ready in"))).toBe(true);
  });

  it("sets env.cacheDir to ~/.cache/huggingface/transformers when unset", async () => {
    const origHome = process.env.HOME;
    const origHfHome = process.env.HF_HOME;
    process.env.HOME = "/tmp/fake-home";
    delete process.env.HF_HOME;
    try {
      mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
      const mod = await loadModule();
      await mod.scoreQueryPassagePairs("q", ["a"]);
      expect(mockEnv.cacheDir).toBe("/tmp/fake-home/.cache/huggingface/transformers");
    } finally {
      if (origHome === undefined) {
        delete process.env.HOME;
      } else {
        process.env.HOME = origHome;
      }
      if (origHfHome === undefined) {
        delete process.env.HF_HOME;
      } else {
        process.env.HF_HOME = origHfHome;
      }
    }
  });

  it("honors $HF_HOME when set, appending /transformers", async () => {
    const origHfHome = process.env.HF_HOME;
    process.env.HF_HOME = "/custom/hf-home";
    try {
      mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
      const mod = await loadModule();
      await mod.scoreQueryPassagePairs("q", ["a"]);
      expect(mockEnv.cacheDir).toBe("/custom/hf-home/transformers");
    } finally {
      if (origHfHome === undefined) {
        delete process.env.HF_HOME;
      } else {
        process.env.HF_HOME = origHfHome;
      }
    }
  });

  it("does not overwrite a user-chosen cache dir outside node_modules", async () => {
    mockEnv.cacheDir = "/user/explicit/choice";
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();
    await mod.scoreQueryPassagePairs("q", ["a"]);
    expect(mockEnv.cacheDir).toBe("/user/explicit/choice");
  });

  it("replaces the default node_modules cache dir", async () => {
    mockEnv.cacheDir = "/repo/node_modules/@huggingface/transformers/.cache/";
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();
    await mod.scoreQueryPassagePairs("q", ["a"]);
    expect(mockEnv.cacheDir).not.toContain("node_modules");
  });

  it("__resetLocalOnnxRerankerForTesting forces the next call to reload", async () => {
    mockModelCall.mockResolvedValue({ logits: { data: [0.1] } });
    const mod = await loadModule();

    await mod.scoreQueryPassagePairs("q", ["a"]);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);

    mod.__resetLocalOnnxRerankerForTesting();
    await mod.scoreQueryPassagePairs("q", ["a"]);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(2);
  });
});
