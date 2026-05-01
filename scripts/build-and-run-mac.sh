#!/usr/bin/env bash
# Fast dev restart for the OpenClaw macOS app (Swift-side only).
#
# What this does:
#   1. Incremental `swift build` (debug) for the OpenClaw product.
#   2. Swap the freshly-built binary into the existing
#      `/Applications/OpenClaw.app/Contents/MacOS/OpenClaw` bundle.
#   3. Ad-hoc re-codesign (required after the binary hash changes — macOS
#      refuses to launch an .app whose sealed binary hash no longer matches).
#   4. Launch /Applications/OpenClaw.app.
#
# Why not launch `.build-local/debug/OpenClaw` directly:
#   NSBundle.mainBundle resolves to the parent directory, which is NOT a
#   proper .app bundle. Any framework that requires a bundle proxy
#   (UNUserNotificationCenter, SMAppService, SwiftUI scene lifecycle, ...)
#   will throw `NSInternalInconsistencyException: bundleProxyForCurrentProcess
#   is nil` immediately on startup. Swapping into the installed .app keeps
#   the Info.plist / frameworks / codesign seal intact.
#
# Prerequisite:
#   /Applications/OpenClaw.app must already exist. If it does not, run
#   `scripts/restart-mac.sh` once to build and install the full bundle.
#   After that, this script handles fast Swift-only iterations.
#
# Intentionally NOT rebuilt here (use restart-mac.sh for these):
#   - `pnpm build` (dist/)           — Node/TS runtime
#   - scripts/package-mac-app.sh     — Frameworks + Info.plist repackaging
#   - Gateway/node daemon restart    — LaunchAgent bootout/bootstrap
#
# Usage:
#   scripts/build-and-run-mac.sh
#   scripts/build-and-run-mac.sh --release    # build with -c release instead

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACOS_DIR="$ROOT_DIR/apps/macos"
BUILD_PATH=".build-local"
PRODUCT="OpenClaw"
BUILD_CONFIG="debug"
APP_PATH="/Applications/OpenClaw.app"

for arg in "$@"; do
  case "$arg" in
    --release) BUILD_CONFIG="release" ;;
    --debug)   BUILD_CONFIG="debug" ;;
    --help|-h)
      sed -n '2,32p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) printf "Unknown arg: %s\n" "$arg" >&2; exit 2 ;;
  esac
done

BIN_REL="$BUILD_PATH/$BUILD_CONFIG/$PRODUCT"

log()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

if [[ ! -d "$APP_PATH" ]]; then
  fail "$APP_PATH not found — run scripts/restart-mac.sh once first to install the full bundle."
fi

APP_BIN="$APP_PATH/Contents/MacOS/$PRODUCT"
if [[ ! -f "$APP_BIN" ]]; then
  fail "$APP_BIN missing inside the installed bundle — bundle looks corrupt; run scripts/restart-mac.sh"
fi

cd "$MACOS_DIR"

log "Building $PRODUCT ($BUILD_CONFIG, build path: $BUILD_PATH)"
# Must mirror scripts/package-mac-app.sh: embed the Frameworks/ rpath at link
# time, otherwise the swapped binary can't find Sparkle.framework /
# libswiftCompatibilitySpan.dylib inside the installed .app and dyld will
# abort the process before main() runs with:
#   "Library not loaded: @rpath/Sparkle.framework/Versions/B/Sparkle"
swift build -c "$BUILD_CONFIG" --product "$PRODUCT" --build-path "$BUILD_PATH" \
  -Xlinker -rpath -Xlinker @executable_path/../Frameworks

FRESH_BIN="$MACOS_DIR/$BIN_REL"
if [[ ! -f "$FRESH_BIN" ]]; then
  fail "swift build succeeded but $FRESH_BIN not found"
fi

log "Stopping existing $PRODUCT"
# Prefer graceful quit so menu-bar / SMAppService stays tidy.
osascript -e 'tell application "OpenClaw" to quit' 2>/dev/null || true
sleep 1
pkill -9 -f "$APP_PATH/Contents/MacOS/$PRODUCT" 2>/dev/null || true
pkill -x "$PRODUCT" 2>/dev/null || true

log "Swapping fresh binary into $APP_BIN"
cp -f "$FRESH_BIN" "$APP_BIN"

# Must re-seal after a binary swap — otherwise macOS rejects launch with:
#   "OpenClaw can't be opened because the signed resource has changed"
# Ad-hoc sign (-s -) is fine for local dev; for distribution use the real
# identity via restart-mac.sh's package flow.
log "Ad-hoc re-codesigning bundle"
# Re-sign the main binary first (with its embedded entitlements if any, so
# TCC permission prompts still work). Then re-seal the outer bundle so the
# CodeResources manifest picks up the new binary hash.
codesign --force --sign - --preserve-metadata=entitlements,requirements \
  "$APP_BIN" >/dev/null 2>&1 \
  || codesign --force --sign - "$APP_BIN"
codesign --force --sign - "$APP_PATH"

log "Launching $APP_PATH"
open "$APP_PATH"

sleep 2
if pgrep -f "$APP_PATH/Contents/MacOS/$PRODUCT" >/dev/null 2>&1; then
  PID="$(pgrep -f "$APP_PATH/Contents/MacOS/$PRODUCT" | head -1)"
  log "$PRODUCT is running ✓ (PID $PID)"
else
  fail "$PRODUCT failed to start — check Console.app or: log stream --predicate 'process == \"OpenClaw\"' --style compact"
fi
