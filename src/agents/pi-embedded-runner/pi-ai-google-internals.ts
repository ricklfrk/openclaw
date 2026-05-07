// Re-export pi-ai internal Google provider utilities.
//
// pi-ai 0.57.1 added an exports map that excludes dist/providers/*.
// These functions are needed by google-nonstream.ts and gcli-nonstream.ts.
// We locate the pi-ai package root via import.meta.resolve (which respects
// the exports map for the "." entry), then use createRequire with the
// absolute file path to load the internal provider modules — CJS require()
// with absolute paths bypasses exports map enforcement.

import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const piAiIndex = fileURLToPath(import.meta.resolve("@mariozechner/pi-ai"));
const piAiDist = dirname(piAiIndex);

const cjsRequire = createRequire(import.meta.url);

/* eslint-disable @typescript-eslint/no-explicit-any */
const googleShared: Record<string, any> = cjsRequire(
  resolve(piAiDist, "providers/google-shared.js"),
);

export const convertMessages: (...args: any[]) => any = googleShared.convertMessages;
export const convertTools: (...args: any[]) => any = googleShared.convertTools;
export const isThinkingPart: (part: any) => boolean = googleShared.isThinkingPart;
export const retainThoughtSignature: (existing: any, incoming: any) => any =
  googleShared.retainThoughtSignature;
export const mapStopReason: (reason: any) => any = googleShared.mapStopReason;
export const mapStopReasonString: (reason: string) => any = googleShared.mapStopReasonString;
export const mapToolChoice: (choice: any) => any = googleShared.mapToolChoice;
/* eslint-enable @typescript-eslint/no-explicit-any */

function parseRetryDelaySeconds(value: string): number | undefined {
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1000;
  }
  return undefined;
}

function parseRetryAfterHeader(value: string): number | undefined {
  const secondsDelay = parseRetryDelaySeconds(value.trim());
  if (secondsDelay !== undefined) {
    return secondsDelay;
  }
  const retryAtMs = Date.parse(value);
  if (Number.isFinite(retryAtMs)) {
    return Math.max(0, retryAtMs - Date.now());
  }
  return undefined;
}

function getResponseHeaders(response?: Response | Headers): Headers | undefined {
  if (!response) {
    return undefined;
  }
  return response instanceof Headers ? response : response.headers;
}

export function extractRetryDelay(
  errorText: string,
  response?: Response | Headers,
): number | undefined {
  const retryAfter = getResponseHeaders(response)?.get("retry-after");
  if (retryAfter) {
    const parsedHeader = parseRetryAfterHeader(retryAfter);
    if (parsedHeader !== undefined) {
      return parsedHeader;
    }
  }

  const retryDelay = errorText.match(/"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"/);
  if (retryDelay?.[1]) {
    return parseRetryDelaySeconds(retryDelay[1]);
  }
  return undefined;
}
