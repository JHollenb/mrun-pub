#!/usr/bin/env bash
# Safe launchd/systemd bridge for one native inference process. The argument file contains one
# literal argv value per line; it is never sourced or evaluated.
set -euo pipefail

MRUN_INFERENCE_BIN="${MRUN_INFERENCE_BIN:?MRUN_INFERENCE_BIN is required}"
MRUN_INFERENCE_ARGS_FILE="${MRUN_INFERENCE_ARGS_FILE:?MRUN_INFERENCE_ARGS_FILE is required}"
MRUN_INFERENCE_KEY_FILE="${MRUN_INFERENCE_KEY_FILE:?MRUN_INFERENCE_KEY_FILE is required}"

[ -x "$MRUN_INFERENCE_BIN" ] || {
  echo "native inference executable is not executable: $MRUN_INFERENCE_BIN" >&2
  exit 1
}
[ -f "$MRUN_INFERENCE_ARGS_FILE" ] || {
  echo "native inference argument file is not regular: $MRUN_INFERENCE_ARGS_FILE" >&2
  exit 1
}
[ -f "$MRUN_INFERENCE_KEY_FILE" ] || {
  echo "native inference key file is not regular: $MRUN_INFERENCE_KEY_FILE" >&2
  exit 1
}

arguments=()
while IFS= read -r value || [ -n "$value" ]; do
  case "$value" in
    "" | \#*) continue ;;
    --api-key-file | --api-key-file=* | --api-key-env | --api-key-env=* | \
    --allow-unauthenticated | --allow-unauthenticated-nonloopback)
      echo "native inference argument file cannot override supervisor authentication" >&2
      exit 2
      ;;
  esac
  arguments+=("$value")
done < "$MRUN_INFERENCE_ARGS_FILE"

[ "${#arguments[@]}" -gt 0 ] || {
  echo "native inference argument file contains no arguments" >&2
  exit 1
}

exec "$MRUN_INFERENCE_BIN" inference serve "${arguments[@]}" \
  --api-key-file "$MRUN_INFERENCE_KEY_FILE"
