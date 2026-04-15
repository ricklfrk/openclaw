import { normalizeOptionalLowercaseString } from "../shared/string-coerce.js";

export type ChatType = "direct" | "group" | "channel" | "webhook";

export function normalizeChatType(raw?: string): ChatType | undefined {
  const value = normalizeOptionalLowercaseString(raw);
  if (!value) {
    return undefined;
  }
  if (value === "direct" || value === "dm") {
    return "direct";
  }
  if (value === "group") {
    return "group";
  }
  if (value === "channel") {
    return "channel";
  }
  if (value === "webhook") {
    return "webhook";
  }
  return undefined;
}
