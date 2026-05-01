/**
 * Public test surface for the NSFW plugin.
 *
 * Re-exports the identity lines the plugin looks for when inline-replacing the
 * core system-prompt identity line with Part 1. Core invariant tests consume
 * this via the bundled-plugin public-surface loader so they catch drift
 * between the NSFW plugin and `src/agents/system-prompt.ts` without
 * reaching into plugin internals.
 */
export { NSFW_IDENTITY_LINES } from "./index.js";
