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
# Belt-and-suspenders: whatever is still listening on the gateway port, kill it.
# Prevents a stale listener from confusing the app's attach-existing path.
if command -v lsof >/dev/null 2>&1; then
  STALE_PIDS="$(lsof -iTCP:18789 -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "$STALE_PIDS" ]; then
    # shellcheck disable=SC2086
    kill -9 $STALE_PIDS 2>/dev/null || true
    log "Killed stale listener(s) on 18789: $STALE_PIDS"
  fi
fi
log "All processes killed"

# 2) Optionally clean dist
if [ "$CLEAN" -eq 1 ]; then
  log "Cleaning dist/"
  rm -rf dist/
fi
log "Removing /Applications/OpenClaw.app"
rm -rf /Applications/OpenClaw.app

# 3) TypeScript + Control UI build (incremental unless --clean).
# `pnpm build` runs the full profile in scripts/build-all.mjs, which now includes
# the `ui:build` step — so `dist/control-ui/` is rebuilt every restart. Without
# this the gateway reports "Control UI assets not found" after rsync --delete
# below wipes the installed CLI's stale control-ui dir.
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
  # No --ignore-existing: user-editable data files now live in ~/.openclaw/extensions/,
  # so code-dir copies can be safely overwritten on restart. With --ignore-existing
  # modified extension sources would silently stay stale in the installed copy.
  rsync -a --exclude='node_modules' extensions/ "$INSTALLED_CLI_DIR/extensions/"
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

# 8) Wait for gateway — the Mac app should start it.
# Uses an HTTP probe instead of `lsof LISTEN` because the Mac app's launchd
# bootout+bootstrap cycle during early startup can briefly free the port, which
# an lsof-only check mistakes for a dead gateway.
GATEWAY_PLIST="$HOME/Library/LaunchAgents/ai.openclaw.gateway.plist"
GATEWAY_PORT=18789
GATEWAY_PROBE_URL="http://127.0.0.1:$GATEWAY_PORT/"

# Single best-effort HTTP probe with a short timeout. Any 2xx/3xx/4xx response
# from the server means "something is listening and replying", which is what
# we actually care about for stability.
probe_gateway() {
  curl -sS -o /dev/null -m 2 -w "%{http_code}" "$GATEWAY_PROBE_URL" 2>/dev/null \
    | grep -Eq '^[1-5][0-9][0-9]$'
}

log "Waiting for gateway on port $GATEWAY_PORT..."
WAITED=0
while [ "$WAITED" -lt 20 ]; do
  if probe_gateway; then
    log "Gateway responding on port $GATEWAY_PORT ✓"
    break
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done

# 9) Fallback: if Mac app didn't start the gateway, use daemon install.
if [ "$WAITED" -ge 20 ]; then
  log "Gateway not started by Mac app; installing daemon as fallback"
  if ! node openclaw.mjs daemon install --force --runtime node; then
    log "Warning: daemon install also failed"
  fi

  WAITED2=0
  while [ "$WAITED2" -lt 15 ]; do
    if probe_gateway; then
      log "Gateway responding on port $GATEWAY_PORT ✓ (via daemon)"
      break
    fi
    sleep 1
    WAITED2=$((WAITED2 + 1))
  done

  if [ "$WAITED2" -ge 15 ]; then
    log "Warning: gateway not responding on port $GATEWAY_PORT after fallback"
    log "  Try: tail -30 ~/.openclaw/logs/gateway.err.log"
  fi
fi

# 10) Final stability check — verify gateway survives the Mac app's launchd
# bootout+bootstrap cycle. Wait longer (the app may recycle the gateway once
# during first-boot) and retry a few times before concluding it is dead,
# otherwise the previous "1-shot lsof" check raced the app's own restart and
# falsely triggered the bootout/bootstrap fallback below, which then killed a
# perfectly healthy gateway.
sleep 10
STABLE=0
for _ in 1 2 3; do
  if probe_gateway; then
    STABLE=1
    break
  fi
  sleep 2
done

if [ "$STABLE" -eq 1 ]; then
  log "Gateway stable ✓"
elif pgrep -f "OpenClaw.app/Contents/MacOS/OpenClaw" >/dev/null 2>&1; then
  # Mac app is the gateway owner when running — it will enable + bootstrap
  # ai.openclaw.gateway on its own via GatewayLaunchAgentManager and watches
  # the job with its own supervisor. If this script *also* bootouts+bootstraps
  # the same plist concurrently, the two actors race and tear down each other's
  # bootstrap — that was the SIGTERM crash-loop source. Probe may also just be
  # timing out within the app's normal first-boot bootout+bootstrap cycle.
  # Wait a bit more, then trust the app supervisor.
  log "Gateway probe timed out but Mac app is running; trusting app supervisor"
  RECOVERED=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    if probe_gateway; then
      RECOVERED=1
      break
    fi
  done
  if [ "$RECOVERED" -eq 1 ]; then
    log "Gateway came up under app supervisor ✓"
  else
    log "Warning: app is running but gateway still not responding."
    log "  Try: tail -30 ~/.openclaw/logs/gateway.err.log"
    log "  Or quit OpenClaw.app and rerun this script (it will fall back to launchd)."
  fi
else
  log "Warning: gateway not responding and Mac app is down — attempting recovery"
  if [ -f "$GATEWAY_PLIST" ]; then
    launchctl bootout gui/"$UID"/ai.openclaw.gateway 2>/dev/null || true
    sleep 1
    launchctl bootstrap gui/"$UID" "$GATEWAY_PLIST" 2>/dev/null || true
  else
    node openclaw.mjs daemon install --force --runtime node 2>/dev/null || true
  fi
  RECOVERED=0
  for _ in 1 2 3 4 5; do
    sleep 2
    if probe_gateway; then
      RECOVERED=1
      break
    fi
  done
  if [ "$RECOVERED" -eq 1 ]; then
    log "Gateway recovered ✓"
  else
    log "Warning: gateway still not responding. Manual start: pnpm openclaw gateway run --bind loopback --port 18789 --force"
  fi
fi

# 11) Bootstrap node launch agent — we bootout'd it in step 1 for a clean kill,
# but the Mac app only manages the gateway, so nothing re-bootstraps the node
# agent on its own. Without this the node daemon stays down until next login,
# and `openclaw nodes list` will show the paired local node as offline.
# Ensure it is enabled (removes from the user's disabled list, persistent across
# reboots) before bootstrap — a disabled service silently refuses to load.
NODE_PLIST="$HOME/Library/LaunchAgents/ai.openclaw.node.plist"
if [ -f "$NODE_PLIST" ]; then
  log "Bootstrapping node launch agent"
  launchctl enable "gui/$UID/ai.openclaw.node" 2>/dev/null || true
  if launchctl bootstrap gui/"$UID" "$NODE_PLIST" 2>/dev/null; then
    log "Node launch agent bootstrapped ✓"
  else
    # Already loaded (5: I/O error) is fine; anything else: try kickstart.
    launchctl kickstart -k "gui/$UID/ai.openclaw.node" 2>/dev/null || true
    log "Node launch agent already loaded (kickstarted)"
  fi
fi
