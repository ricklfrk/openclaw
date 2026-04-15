#!/usr/bin/env bash
# Quick restart: kill everything → incremental build → package → install → launch.
# Skips clean dist (incremental) for faster turnaround than rebuild-mac.sh.
# Usage: scripts/restart-mac.sh [--clean]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load shell env (e.g. SIGN_IDENTITY from ~/.zshrc) when present.
# Disable all error flags before sourcing: zsh-specific syntax in .zshrc (e.g. bun/openclaw
# completion files) causes bash parser errors at parse-time, which exit the script before
# `|| true` can suppress them. Restoring flags immediately after.
if [[ -f "$HOME/.zshrc" ]] && [[ -z "${REBUILD_MAC_SKIP_ZSHRC:-}" ]]; then
  set +euo pipefail
  # shellcheck source=/dev/null
  source "$HOME/.zshrc" 2>/dev/null || true
  set -euo pipefail
fi

log()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    --clean) CLEAN=1 ;;
    --help|-h)
      log "Usage: $(basename "$0") [--clean]"
      log "  --clean  Remove dist/ before building (same as rebuild-mac.sh)"
      exit 0
      ;;
    *) ;;
  esac
done

# 1) Kill OpenClaw app + gateway + signal-cli
log "Killing OpenClaw and signal-cli"
osascript -e 'tell application "OpenClaw" to quit' 2>/dev/null || true
sleep 2
pkill -9 -f "OpenClaw.app/Contents/MacOS/OpenClaw" 2>/dev/null || true
pkill -x "OpenClaw" 2>/dev/null || true
pkill -9 -f "openclaw-gateway" 2>/dev/null || true
pkill -9 -f "signal-cli" 2>/dev/null || true
launchctl bootout gui/"$UID"/ai.openclaw.gateway 2>/dev/null || true
launchctl bootout gui/"$UID"/ai.openclaw.node 2>/dev/null || true
sleep 1
log "All processes killed"

# 2) Optionally clean dist
if [ "$CLEAN" -eq 1 ]; then
  log "Cleaning dist/"
  rm -rf dist/
fi
log "Removing /Applications/OpenClaw.app"
rm -rf /Applications/OpenClaw.app

# 3) TypeScript build (incremental unless --clean)
log "Running pnpm build"
if ! pnpm build; then
  fail "pnpm build failed"
fi
log "Build succeeded"

# 3b) Sync dev build to installed CLI shim
INSTALLED_CLI_DIR="$HOME/.openclaw/lib/node_modules/openclaw"
if [ -d "$INSTALLED_CLI_DIR" ]; then
  log "Syncing dev build to installed CLI ($INSTALLED_CLI_DIR)"
  rsync -a --delete dist/ "$INSTALLED_CLI_DIR/dist/"
  cp package.json "$INSTALLED_CLI_DIR/package.json"
  rsync -a --exclude='node_modules' extensions/ "$INSTALLED_CLI_DIR/extensions/" --ignore-existing
  log "Installed CLI updated"
fi

# 4) Package Mac app (use SIGN_IDENTITY if set; else allow ad-hoc for dev)
log "Packaging Mac app"
if [[ -z "${SIGN_IDENTITY:-}" ]]; then
  export ALLOW_ADHOC_SIGNING=1
fi
if ! scripts/package-mac-app.sh; then
  fail "package-mac-app.sh failed"
fi
log "Package succeeded: dist/OpenClaw.app"

# 5) Install to /Applications
log "Installing to /Applications"
cp -R dist/OpenClaw.app /Applications/OpenClaw.app
log "Installed"

# 6) Launch Mac app first — it manages the gateway lifecycle.
#    Only fall back to daemon install if the app doesn't bring up the gateway.
log "Launching OpenClaw"
open /Applications/OpenClaw.app

# 7) Verify app
sleep 3
if pgrep -f "OpenClaw.app/Contents/MacOS/OpenClaw" >/dev/null 2>&1; then
  log "OpenClaw is running ✓"
else
  fail "OpenClaw failed to start — check Console.app"
fi

# 8) Wait for gateway — the Mac app should start it
GATEWAY_PLIST="$HOME/Library/LaunchAgents/ai.openclaw.gateway.plist"
GATEWAY_PORT=18789

log "Waiting for gateway on port $GATEWAY_PORT..."
WAITED=0
while [ "$WAITED" -lt 20 ]; do
  if lsof -nP -i :"$GATEWAY_PORT" 2>/dev/null | grep -q LISTEN; then
    log "Gateway listening on port $GATEWAY_PORT ✓"
    break
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done

# 9) Fallback: if Mac app didn't start the gateway, use daemon install
if [ "$WAITED" -ge 20 ]; then
  log "Gateway not started by Mac app; installing daemon as fallback"
  if ! node openclaw.mjs daemon install --force --runtime node; then
    log "Warning: daemon install also failed"
  fi

  # Wait again for daemon-started gateway
  WAITED2=0
  while [ "$WAITED2" -lt 15 ]; do
    if lsof -nP -i :"$GATEWAY_PORT" 2>/dev/null | grep -q LISTEN; then
      log "Gateway listening on port $GATEWAY_PORT ✓ (via daemon)"
      break
    fi
    sleep 1
    WAITED2=$((WAITED2 + 1))
  done

  if [ "$WAITED2" -ge 15 ]; then
    log "Warning: gateway not listening on port $GATEWAY_PORT after fallback"
    log "  Try: tail -30 ~/.openclaw/logs/gateway.err.log"
  fi
fi

# 10) Final stability check — verify gateway survives after 5s
sleep 5
if lsof -nP -i :"$GATEWAY_PORT" 2>/dev/null | grep -q LISTEN; then
  log "Gateway stable ✓"
else
  log "Warning: gateway died after initial startup — restarting"
  if [ -f "$GATEWAY_PLIST" ]; then
    launchctl bootout gui/"$UID"/ai.openclaw.gateway 2>/dev/null || true
    sleep 1
    launchctl bootstrap gui/"$UID" "$GATEWAY_PLIST" 2>/dev/null || true
  else
    node openclaw.mjs daemon install --force --runtime node 2>/dev/null || true
  fi
  sleep 5
  if lsof -nP -i :"$GATEWAY_PORT" 2>/dev/null | grep -q LISTEN; then
    log "Gateway recovered ✓"
  else
    log "Warning: gateway still not running. Manual start: pnpm openclaw gateway run --bind loopback --port 18789 --force"
  fi
fi
