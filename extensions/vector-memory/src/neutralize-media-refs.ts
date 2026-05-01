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
 * This helper rewrites the markers into explicit historical-reference forms
 * (`[historical-media-ref …]` / `[historical-image-ref …]`) so the path is
 * still available if the assistant is later asked to send the old file back,
 * but the reference is clearly not a current-turn attachment.
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
 * Rewrite live-attachment markers inside recalled memory text to explicit
 * historical forms. The payload (path / mime / URL) is preserved verbatim so
 * the LLM can still use the path for an intentional resend via a messaging
 * tool, while the outer marker makes clear that this is not a live attachment.
 */
export function neutralizeRecalledMediaRefs(text: string): string {
  if (!text) {
    return text;
  }
  return text
    .replace(MEDIA_ATTACHED_PATTERN, (_m, idx: string | undefined, body: string) => {
      const suffix = idx ? idx : "";
      return `[historical-media-ref${suffix}; not-current-attachment; path-preserved-for-resend: ${body.trim()}]`;
    })
    .replace(MESSAGE_IMAGE_PATTERN, (_m, body: string) => {
      return `[historical-image-ref; not-current-attachment; path-preserved-for-resend: ${body.trim()}]`;
    });
}
