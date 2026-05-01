import { loadBundledPluginTestApiSync } from "../../src/test-utils/bundled-plugin-public-surface.js";

type NsfwTestApiSurface = {
  NSFW_IDENTITY_LINES: readonly string[];
};

let cache: NsfwTestApiSurface | undefined;

/**
 * Load the identity lines the NSFW plugin recognizes when inline-replacing
 * the core system-prompt identity line with Part 1. Core invariant tests use
 * this to detect drift between NSFW's matcher and the core system prompt
 * without reaching into plugin internals.
 */
export function loadNsfwIdentityLines(): readonly string[] {
  if (!cache) {
    cache = loadBundledPluginTestApiSync<NsfwTestApiSurface>("nsfw");
  }
  return cache.NSFW_IDENTITY_LINES;
}
