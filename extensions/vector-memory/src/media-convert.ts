/**
 * Media format conversion for Gemini Embedding 2 compatibility.
 *
 * Gemini Embedding 2 only accepts:
 *   Image: PNG, JPEG
 *   Audio: MP3, WAV
 *   Video: MP4, MOV
 *   PDF
 *
 * This module converts unsupported formats on-the-fly:
 *   WebP/GIF  → JPEG  (via sharp, available in root node_modules)
 *   WebM      → MP4   (via ffmpeg, if available)
 */

import { execFile } from "node:child_process";
import { writeFile, readFile, unlink, mkdtemp, rmdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export interface ConvertedMedia {
  data: Buffer;
  mimeType: string;
  converted: boolean;
}

type SharpInstance = {
  jpeg: (opts?: { quality?: number; mozjpeg?: boolean }) => SharpInstance;
  toBuffer: () => Promise<Buffer>;
};
type SharpConstructor = (input: Buffer) => SharpInstance;

let _sharp: SharpConstructor | null | false = null;

async function loadSharp(): Promise<SharpConstructor | null> {
  if (_sharp === false) {
    return null;
  }
  if (_sharp) {
    return _sharp;
  }
  try {
    const mod = (await import("sharp")) as unknown as {
      default?: SharpConstructor;
    };
    _sharp = mod.default ?? (mod as unknown as SharpConstructor);
    return _sharp;
  } catch {
    _sharp = false;
    return null;
  }
}

let _ffmpegAvailable: boolean | null = null;

async function hasFfmpeg(): Promise<boolean> {
  if (_ffmpegAvailable !== null) {
    return _ffmpegAvailable;
  }
  try {
    await execFileAsync("ffmpeg", ["-version"], { timeout: 5000 });
    _ffmpegAvailable = true;
  } catch {
    _ffmpegAvailable = false;
  }
  return _ffmpegAvailable;
}

// Image: WebP/GIF → JPEG
async function convertImageToJpeg(buf: Buffer): Promise<Buffer> {
  const sharp = await loadSharp();
  if (!sharp) {
    throw new Error("sharp not available for image conversion");
  }
  return sharp(buf).jpeg({ quality: 85, mozjpeg: true }).toBuffer();
}

// Video: WebM → MP4 (via temp files + ffmpeg)
async function convertWebmToMp4(buf: Buffer): Promise<Buffer> {
  if (!(await hasFfmpeg())) {
    throw new Error("ffmpeg not available for video conversion");
  }

  const dir = await mkdtemp(join(tmpdir(), "vm-convert-"));
  const inputPath = join(dir, "input.webm");
  const outputPath = join(dir, "output.mp4");

  try {
    await writeFile(inputPath, buf);
    await execFileAsync(
      "ffmpeg",
      [
        "-i",
        inputPath,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-an", // strip audio (Gemini ignores video audio tracks)
        "-t",
        "120", // cap at 120s
        "-y",
        outputPath,
      ],
      { timeout: 60_000 },
    );
    return await readFile(outputPath);
  } finally {
    await unlink(inputPath).catch(() => {});
    await unlink(outputPath).catch(() => {});
    await rmdir(dir).catch(() => {});
  }
}

const NEEDS_CONVERSION: Record<string, string> = {
  "image/webp": "image/jpeg",
  "image/gif": "image/jpeg",
  "video/webm": "video/mp4",
};

/**
 * Convert media to a Gemini Embedding 2 compatible format if needed.
 * Returns the original data unchanged if already compatible.
 */
export async function convertForEmbedding(data: Buffer, mimeType: string): Promise<ConvertedMedia> {
  const targetMime = NEEDS_CONVERSION[mimeType];
  if (!targetMime) {
    return { data, mimeType, converted: false };
  }

  if (targetMime === "image/jpeg") {
    const converted = await convertImageToJpeg(data);
    return { data: converted, mimeType: "image/jpeg", converted: true };
  }

  if (targetMime === "video/mp4") {
    const converted = await convertWebmToMp4(data);
    return { data: converted, mimeType: "video/mp4", converted: true };
  }

  return { data, mimeType, converted: false };
}

/**
 * Convert a file on disk for embedding. Used by the store_media tool.
 * Returns the converted buffer and MIME type.
 */
export async function convertFileForEmbedding(
  filePath: string,
  mimeType: string,
): Promise<ConvertedMedia> {
  const targetMime = NEEDS_CONVERSION[mimeType];
  if (!targetMime) {
    const buf = await readFile(filePath);
    return { data: buf, mimeType, converted: false };
  }

  const buf = await readFile(filePath);
  return convertForEmbedding(buf, mimeType);
}
