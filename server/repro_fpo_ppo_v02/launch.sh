#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAN=""
RUNS_ROOT=""
GPUS=""
MAX_ATTEMPTS=""
SESSION=""
FPO_ROOT=""
PYTHON_BIN=""
VENDOR_DIR=""
LEGACY_POLICY_IO=""
RUNNER="${SCRIPT_DIR}/runner.py"
POLL_SECONDS="1"
TERMINATE_GRACE_SECONDS="20"
ALLOW_NON_GPU=0
WAIT_FOR_IDLE_GPUS=0
IDLE_MAX_MEMORY_USED_MIB="512"
IDLE_MAX_UTILIZATION_PERCENT="5"
RESOURCE_POLL_SECONDS="15"
EXECUTION_PURPOSE=""

usage() {
  printf '%s\n' \
    "Usage: $0 --plan PATH --execution-purpose PURPOSE --runs-root PATH --gpus CSV --max-attempts N" \
    "          --session NAME --fpo-root PATH --python PATH --vendor-dir PATH" \
    "          --legacy-policy-io PATH [options]" \
    "" \
    "All scientific choices must already be frozen in the anchor/protocol/plan digests." \
    "" \
    "Required:" \
    "  --plan PATH                  immutable v0.2 training plan" \
    "  --execution-purpose PURPOSE  audit_smoke, development_discovery, or v02_freeze_ready" \
    "  --runs-root PATH             dedicated artifact root for this plan" \
    "  --gpus CSV                   explicit physical GPU indices" \
    "  --max-attempts N             explicit total attempts per semantic job" \
    "  --session NAME               new tmux session (duplicate launch is refused)" \
    "  --fpo-root PATH              clean FPO checkout pinned by each anchor" \
    "  --python PATH                Python executable for queue and runner" \
    "  --vendor-dir PATH            pinned legacy dependencies for runner PYTHONPATH" \
    "  --legacy-policy-io PATH      exact read-only legacy policy_io.py exporter" \
    "" \
    "Engineering options:" \
    "  --runner PATH                anchor-aware runner (default: ${RUNNER})" \
    "  --poll-seconds N             child polling interval (default: 1)" \
    "  --terminate-grace-seconds N  process-group shutdown grace (default: 20)" \
    "  --wait-for-idle-gpus         wait for two consecutive idle resource probes" \
    "  --idle-max-memory-used-mib N idle gate memory threshold (default: 512)" \
    "  --idle-max-utilization-percent N idle gate utilization threshold (default: 5)" \
    "  --resource-poll-seconds N    idle gate polling interval (default: 15)" \
    "  --allow-non-gpu              synthetic/debug use only; never formal evidence" \
    "  -h, --help                   show this help"
}

need_value() {
  if (($# < 2)) || [[ -z "${2:-}" ]]; then
    printf 'missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --plan|--execution-purpose|--runs-root|--gpus|--max-attempts|--session|--fpo-root|--python|--vendor-dir|--legacy-policy-io|--runner|--poll-seconds|--terminate-grace-seconds|--idle-max-memory-used-mib|--idle-max-utilization-percent|--resource-poll-seconds)
      need_value "$@"
      case "$1" in
        --plan) PLAN="$2" ;;
        --execution-purpose) EXECUTION_PURPOSE="$2" ;;
        --runs-root) RUNS_ROOT="$2" ;;
        --gpus) GPUS="$2" ;;
        --max-attempts) MAX_ATTEMPTS="$2" ;;
        --session) SESSION="$2" ;;
        --fpo-root) FPO_ROOT="$2" ;;
        --python) PYTHON_BIN="$2" ;;
        --vendor-dir) VENDOR_DIR="$2" ;;
        --legacy-policy-io) LEGACY_POLICY_IO="$2" ;;
        --runner) RUNNER="$2" ;;
        --poll-seconds) POLL_SECONDS="$2" ;;
        --terminate-grace-seconds) TERMINATE_GRACE_SECONDS="$2" ;;
        --idle-max-memory-used-mib) IDLE_MAX_MEMORY_USED_MIB="$2" ;;
        --idle-max-utilization-percent) IDLE_MAX_UTILIZATION_PERCENT="$2" ;;
        --resource-poll-seconds) RESOURCE_POLL_SECONDS="$2" ;;
      esac
      shift 2
      ;;
    --allow-non-gpu)
      ALLOW_NON_GPU=1
      shift
      ;;
    --wait-for-idle-gpus)
      WAIT_FOR_IDLE_GPUS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required in PLAN EXECUTION_PURPOSE RUNS_ROOT GPUS MAX_ATTEMPTS SESSION FPO_ROOT PYTHON_BIN VENDOR_DIR LEGACY_POLICY_IO; do
  if [[ -z "${!required}" ]]; then
    printf 'required option was not supplied: %s\n' "$required" >&2
    usage >&2
    exit 2
  fi
done

case "$EXECUTION_PURPOSE" in
  audit_smoke|development_discovery|v02_freeze_ready) ;;
  *)
    printf 'invalid execution purpose: %s\n' "$EXECUTION_PURPOSE" >&2
    exit 2
    ;;
esac
if [[ "$EXECUTION_PURPOSE" == "audit_smoke" && "$ALLOW_NON_GPU" -ne 1 ]]; then
  printf 'audit_smoke requires --allow-non-gpu\n' >&2
  exit 2
fi
if [[ "$EXECUTION_PURPOSE" != "audit_smoke" && "$ALLOW_NON_GPU" -eq 1 ]]; then
  printf '%s cannot use --allow-non-gpu\n' "$EXECUTION_PURPOSE" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required but was not found\n' >&2
  exit 2
fi
if [[ ! -f "$PLAN" ]]; then
  printf 'immutable training plan does not exist: %s\n' "$PLAN" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  printf 'anchor-aware runner does not exist: %s\n' "$RUNNER" >&2
  exit 2
fi
if [[ ! -d "$FPO_ROOT" ]]; then
  printf 'FPO root does not exist: %s\n' "$FPO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$VENDOR_DIR" ]]; then
  printf 'vendor dependency directory does not exist: %s\n' "$VENDOR_DIR" >&2
  exit 2
fi
if [[ ! -f "$LEGACY_POLICY_IO" ]]; then
  printf 'legacy policy exporter does not exist: %s\n' "$LEGACY_POLICY_IO" >&2
  exit 2
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python executable is missing or not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
  fi
  PYTHON_RESOLVED="$PYTHON_BIN"
else
  if ! PYTHON_RESOLVED="$(command -v -- "$PYTHON_BIN")"; then
    printf 'Python executable was not found: %s\n' "$PYTHON_BIN" >&2
    exit 2
  fi
fi
if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  printf 'tmux session already exists; refusing duplicate launch: %s\n' "$SESSION" >&2
  exit 3
fi

COMMAND_ARGS=(
  "$PYTHON_RESOLVED" -u "$SCRIPT_DIR/queue_master.py"
  --plan "$PLAN"
  --execution-purpose "$EXECUTION_PURPOSE"
  --runner "$RUNNER"
  --fpo-root "$FPO_ROOT"
  --runs-root "$RUNS_ROOT"
  --gpus "$GPUS"
  --python "$PYTHON_RESOLVED"
  --vendor-dir "$VENDOR_DIR"
  --legacy-policy-io "$LEGACY_POLICY_IO"
  --max-attempts "$MAX_ATTEMPTS"
  --poll-seconds "$POLL_SECONDS"
  --terminate-grace-seconds "$TERMINATE_GRACE_SECONDS"
)
if [[ "$WAIT_FOR_IDLE_GPUS" -eq 1 ]]; then
  COMMAND_ARGS+=(
    --wait-for-idle-gpus
    --idle-max-memory-used-mib "$IDLE_MAX_MEMORY_USED_MIB"
    --idle-max-utilization-percent "$IDLE_MAX_UTILIZATION_PERCENT"
    --resource-poll-seconds "$RESOURCE_POLL_SECONDS"
  )
fi
if [[ "$ALLOW_NON_GPU" -eq 1 ]]; then
  COMMAND_ARGS+=(--allow-non-gpu)
  printf 'WARNING: --allow-non-gpu output is synthetic/debug evidence only.\n' >&2
fi

printf -v COMMAND_STRING '%q ' "${COMMAND_ARGS[@]}"
tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" "$COMMAND_STRING"

printf 'launched v0.2 queue session: %s\n' "$SESSION"
printf 'monitor: %q --runs-root %q --session %q --python %q\n' \
  "$SCRIPT_DIR/monitor.sh" "$RUNS_ROOT" "$SESSION" "$PYTHON_RESOLVED"
printf 'attach: tmux attach -t %q\n' "$SESSION"
