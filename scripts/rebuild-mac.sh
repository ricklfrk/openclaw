#!/usr/bin/env bash
# Clean rebuild: thin wrapper over restart-mac.sh --clean.
# Kept for muscle memory; restart-mac.sh --clean is the canonical path.
# Usage: scripts/rebuild-mac.sh [extra args passed through]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/restart-mac.sh" --clean "$@"
