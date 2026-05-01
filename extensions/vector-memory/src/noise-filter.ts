/**
 * Noise Filter — filters out low-quality memories.
 * From memory-lancedb-pro.
 */

const DENIAL_PATTERNS = [
  /i don'?t have (any )?(information|data|memory|record)/i,
  /i'?m not sure about/i,
  /i don'?t recall/i,
  /i don'?t remember/i,
  /it looks like i don'?t/i,
  /i wasn'?t able to find/i,
  /no (relevant )?memories found/i,
  /i don'?t have access to/i,
];

const META_QUESTION_PATTERNS = [
  /\bdo you (remember|recall|know about)\b/i,
  /\bcan you (remember|recall)\b/i,
  /\bdid i (tell|mention|say|share)\b/i,
  /\bhave i (told|mentioned|said)\b/i,
  /\bwhat did i (tell|say|mention)\b/i,
  /如果你知道.+只回复/i,
  /如果不知道.+只回复\s*none/i,
  /只回复精确代号/i,
  /只回复\s*none/i,
  /你还?记得/,
  /记不记得/,
  /还记得.*吗/,
  /你[知晓]道.+吗/,
  /我(?:之前|上次|以前)(?:说|提|讲).*(?:吗|呢|？|\?)/,
];

const BOILERPLATE_PATTERNS = [
  /^(hi|hello|hey|good morning|good evening|greetings)/i,
  /^fresh session/i,
  /^new session/i,
  /^HEARTBEAT/i,
];

const DIAGNOSTIC_ARTIFACT_PATTERNS = [
  /\bquery\s*->\s*(none|no explicit solution|unknown|not found)\b/i,
  /\buser asked for\b.*\b(none|no explicit solution|unknown|not found)\b/i,
  /\bno explicit solution\b/i,
];

export function isNoise(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length < 5) {
    return true;
  }
  if (DENIAL_PATTERNS.some((p) => p.test(trimmed))) {
    return true;
  }
  if (META_QUESTION_PATTERNS.some((p) => p.test(trimmed))) {
    return true;
  }
  if (BOILERPLATE_PATTERNS.some((p) => p.test(trimmed))) {
    return true;
  }
  if (DIAGNOSTIC_ARTIFACT_PATTERNS.some((p) => p.test(trimmed))) {
    return true;
  }
  return false;
}

export function filterNoise<T>(items: T[], getText: (item: T) => string): T[] {
  return items.filter((item) => !isNoise(getText(item)));
}
