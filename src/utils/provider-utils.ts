/**
 * Utility functions for provider-specific logic and capabilities.
 */

/**
 * Returns true if the provider requires reasoning to be wrapped in tags
 * (e.g. <think> and <final>) in the text stream, rather than using native
 * API fields for reasoning/thinking.
 *
 * Also checks the model name so Gemini/Minimax models proxied through
 * custom providers (e.g. myapi/gemini-3-flash) are still detected.
 */
export function isReasoningTagProvider(
  provider: string | undefined | null,
  model?: string | null,
): boolean {
  if (provider) {
    const normalized = provider.trim().toLowerCase();

    if (
      normalized === "google" ||
      normalized === "google-gemini-cli" ||
      normalized === "google-generative-ai"
    ) {
      return true;
    }

    if (normalized.includes("minimax")) {
      return true;
    }
  }

  // Check model name for known reasoning-tag models served through custom providers
  if (model) {
    const normalizedModel = model.trim().toLowerCase();
    if (normalizedModel.includes("gemini") || normalizedModel.includes("minimax")) {
      return true;
    }
  }

  return false;
}
