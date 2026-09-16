#!/bin/bash
# Launch the calibrated-decision chat assistant (local model, interactive).
#
# macOS: double-click this file in Finder and it opens Terminal and starts the
# assistant:
#   - one forward pass decides "general" or "arithmetic" for each turn, with a
#     calibrated confidence that gates whether the turn is answered at all
#   - arithmetic goes to a real calculator, so the number is exact
#   - everything else is generated prose by the model below
#
# From a terminal, any flag passes through:
#   ./Desktop/Assistant.command --report
#   ./Desktop/Assistant.command --bench --no-tool
#
# On Linux, run it directly:  bash Desktop/Assistant.command

set -uo pipefail

# Resolve this script's own location, so the launcher works from wherever the
# repo was cloned and from a symlink -- which is how it usually gets onto a
# Desktop. Nothing here assumes a home directory or a fixed path.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  LINK_DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
  SOURCE="$(readlink -- "$SOURCE")"
  case "$SOURCE" in
    /*) ;;                        # already absolute
    *) SOURCE="$LINK_DIR/$SOURCE" # relative to the symlink's directory
  esac
done
SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
PROJECT="$(cd -P -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT/.venv/bin/python"

# Pin a different model here, or export MODEL=... before launching. Empty means
# the script default (Qwen/Qwen2.5-1.5B-Instruct). Bigger models give better
# prose and slower replies:
#   MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL="${MODEL:-}"

if [ ! -x "$PYTHON" ]; then
  echo "No virtualenv found at:"
  echo "  $PROJECT/.venv"
  echo
  echo "Create it once, then run this again:"
  echo
  echo "  cd \"$PROJECT\""
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e '.[local]'"
  echo
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

# First run downloads the weights, which is slow enough to look like a hang.
# Say so before it starts. CACHE mirrors what transformers does: HF_HOME wins,
# otherwise the default cache directory.
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -z "$MODEL" ] && [ ! -d "$CACHE/models--Qwen--Qwen2.5-1.5B-Instruct" ]; then
  echo "First run: Qwen/Qwen2.5-1.5B-Instruct is not cached yet."
  echo "Downloading ~2.9 GB into:"
  echo "  $CACHE"
  echo "One time only, and shared with any other project using the same model."
  echo
fi

cd "$PROJECT" || exit 1

MODEL_FLAG=""
[ -n "$MODEL" ] && MODEL_FLAG="--model $MODEL"

# MODEL_FLAG is deliberately unquoted so an empty value contributes no argument.
# shellcheck disable=SC2086
"$PYTHON" examples/assistant.py $MODEL_FLAG "$@"
status=$?

# The pause is inside the script, so a failure stays on screen no matter how
# Terminal is configured to behave when the shell exits.
if [ "$status" -ne 0 ]; then
  echo
  echo "The assistant exited with status $status."
  read -n 1 -s -r -p "Press any key to close..."
fi

exit "$status"
