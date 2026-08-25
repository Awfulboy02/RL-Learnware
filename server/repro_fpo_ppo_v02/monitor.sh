#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT=""
SESSION=""
PYTHON_BIN="python3"
FOLLOW=0
SHOW_JSON=0

usage() {
  printf '%s\n' \
    "Usage: $0 --runs-root PATH --session NAME [--python PATH] [--json] [--follow]"
}

while (($#)); do
  case "$1" in
    --runs-root|--session|--python)
      if (($# < 2)) || [[ -z "${2:-}" ]]; then
        printf 'missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --runs-root) RUNS_ROOT="$2" ;;
        --session) SESSION="$2" ;;
        --python) PYTHON_BIN="$2" ;;
      esac
      shift 2
      ;;
    --json) SHOW_JSON=1; shift ;;
    --follow) FOLLOW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUNS_ROOT" || -z "$SESSION" ]]; then
  printf '%s\n' '--runs-root and --session are required' >&2
  usage >&2
  exit 2
fi
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "=${SESSION}" 2>/dev/null; then
  printf 'tmux=%s state=alive\n' "$SESSION"
else
  printf 'tmux=%s state=absent\n' "$SESSION"
fi

STATUS_FILE="${RUNS_ROOT}/queue_status.json"
MASTER_LOG="${RUNS_ROOT}/master.log"
if [[ -f "$STATUS_FILE" ]]; then
  if [[ "$SHOW_JSON" -eq 1 ]]; then
    "$PYTHON_BIN" -m json.tool "$STATUS_FILE"
  else
    "$PYTHON_BIN" -c '
import json
import sys

def reject(value):
    raise ValueError(f"non-finite status constant: {value}")

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle, parse_constant=reject)
counts = ", ".join(
    f"{key}={value}" for key, value in sorted(data.get("counts", {}).items())
) or "none"
print(
    f"queue={data.get('"'"'state'"'"')} updated={data.get('"'"'updated_at'"'"')} "
    f"plan_digest={data.get('"'"'plan_digest'"'"')}"
)
print(f"counts: {counts}")
running = data.get("running", [])
if not running:
    print("running: none")
for item in running:
    print(
        "running: "
        f"gpu={item['"'"'gpu'"'"']} job={item['"'"'job_id'"'"']} pid={item['"'"'pid'"'"']} "
        f"elapsed={item['"'"'elapsed_seconds'"'"']}s attempt={item['"'"'attempt_dir'"'"']}"
    )
' "$STATUS_FILE"
  fi
else
  printf 'queue status not created yet: %s\n' "$STATUS_FILE"
fi

if [[ -f "$MASTER_LOG" ]]; then
  printf 'recent master log:\n'
  tail -n 12 "$MASTER_LOG"
fi

if [[ "$FOLLOW" -eq 1 ]]; then
  if [[ ! -f "$MASTER_LOG" ]]; then
    printf 'cannot follow missing master log: %s\n' "$MASTER_LOG" >&2
    exit 1
  fi
  exec tail -F "$MASTER_LOG"
fi
