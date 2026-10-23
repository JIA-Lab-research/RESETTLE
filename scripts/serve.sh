#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RESETTLE_CHECKPOINT:?Set RESETTLE_CHECKPOINT to the guarded K=1 latest.pt}"
: "${RESETTLE_BANK:?Set RESETTLE_BANK to the matching LIBERO expert goal bank}"
cd "$ROOT"
exec "${PYTHON:-python}" -m app.recovery_server \
    --checkpoint "$RESETTLE_CHECKPOINT" --bank-dir "$RESETTLE_BANK" "$@"