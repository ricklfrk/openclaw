/**
 * Defence-in-depth rewriter for recalled memory text.
 *
 * Recalled memory chunks (from daily/workspace journals or entity summaries)
 * frequently contain verbatim markers like:
 *   `[media attached: /path/to/old.jpg (image/jpeg)]`
 *   `[media attached: media://inbound/<id>]`
 *   `[Image: source: /path/to/screenshot.png]`
 *
 * These markers were legitimate breadcrumbs for the ORIGINAL turn, but once
 * they are re-injected as recalled context the downstream multimodal runner
 * cannot safely distinguish a current-turn attachment from a stale, recalled
 * one. Without neutralization, a regex scanner upstream may follow the path,
 * load the file from disk, and re-attach it to the live multimodal request —
 * effectively resurrecting months-old images into the present turn and
 * confusing the model about temporal context.
 *
 * This helper rewrites the markers into inert forms (`[archived-media-ref …]`
 * / `[archived-image-ref …]`) so the original textual information is still
 * useful to the LLM as *description*, but cannot be re-interpreted as a live
 * attachment by any downstream scanner.
 *
 * This is defence-in-depth: the pi-embedded-runner image scanner *also*
 * strips `<relevant-memories>`/`<from-*-memory>` blocks before scanning, so
 * either layer alone prevents the leak. Keeping both layers means a future
 * prompt-assembly bug that removes the XML wrapper still cannot cause a
 * filesystem re-read from recalled memory content.
 */

const MEDIA_ATTACHED_PATTERN = /\[media attached(\s+\d+\/\d+)?:\s*([^\]]+)\]/gi;
const MESSAGE_IMAGE_PATTERN = /\[Image:\s*source:\s*([^\]]+)\]/gi;

/**
 * Rewrite live-attachment markers inside recalled memory text to inert forms.
 * The payload (path / mime / URL) is preserved verbatim inside the inert tag
 * so the LLM can still reason about *what* the historical attachment was —
 * the change is purely syntactic (different outer token) so downstream
 * regex scanners no longer match.
 */
export function neutralizeRecalledMediaRefs(text: string): string {
  if (!text) {
    return text;
  }
  return text
    .replace(MEDIA_ATTACHED_PATTERN, (_m, idx: string | undefined, body: string) => {
      const suffix = idx ? idx : "";
      return `[archived-media-ref${suffix}: ${body.trim()}]`;
    })
    .replace(MESSAGE_IMAGE_PATTERN, (_m, body: string) => {
      return `[archived-image-ref: ${body.trim()}]`;
    });
}
