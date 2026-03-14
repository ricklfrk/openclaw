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

const googleShared = cjsRequire(resolve(piAiDist, "providers/google-shared.js")) as Record<
  string,
  (...a: unknown[]) => unknown
>;
const googleGeminiCli = cjsRequire(resolve(piAiDist, "providers/google-gemini-cli.js")) as {
  extractRetryDelay: (errorText: string, response?: Response | Headers) => number | undefined;
};

export const convertMessages = googleShared.convertMessages as (...args: unknown[]) => unknown;
export const convertTools = googleShared.convertTools as (...args: unknown[]) => unknown;
export const isThinkingPart = googleShared.isThinkingPart as (part: unknown) => boolean;
export const retainThoughtSignature = googleShared.retainThoughtSignature as (
  existing: unknown,
  incoming: unknown,
) => unknown;
export const mapStopReason = googleShared.mapStopReason as (reason: unknown) => unknown;
export const mapStopReasonString = googleShared.mapStopReasonString as (reason: string) => unknown;
export const mapToolChoice = googleShared.mapToolChoice as (choice: unknown) => unknown;
export const extractRetryDelay: (
  errorText: string,
  response?: Response | Headers,
) => number | undefined = googleGeminiCli.extractRetryDelay;
