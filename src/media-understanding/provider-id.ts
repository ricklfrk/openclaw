import { normalizeProviderId } from "../agents/provider-id.js";

const GOOGLE_MEDIA_PROVIDER_ALIASES = new Set(["gemini", "lab", "google-gemini-cli"]);

export function normalizeMediaProviderId(id: string): string {
  const normalized = normalizeProviderId(id);
  if (GOOGLE_MEDIA_PROVIDER_ALIASES.has(normalized)) {
    return "google";
  }
  return normalized;
}
