#!/usr/bin/env bash
# PostToolUse hook: after any Python edit, run the test suite and tell Claude
# if it broke something. Exit 2 surfaces stderr to Claude as feedback; it
# cannot undo the edit, but Claude sees the failure immediately and fixes it.

input=$(cat)
case "$input" in
  *'.py"'*) ;;          # only react to Python files
  *) exit 0 ;;
esac

cd "$CLAUDE_PROJECT_DIR" || exit 0
if ! out=$(python3 -m pytest -q -x 2>&1); then
  echo "Tests failed after your edit:" >&2
  echo "$out" | tail -25 >&2
  exit 2
fi
exit 0
