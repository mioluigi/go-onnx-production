#!/usr/bin/env bash
# Copies results/tables.md into README.md between the lines
#   <!-- tables:start -->   and   <!-- tables:end -->
# so the README's tables always come from the measured results, never typed by
# hand (Constitution 5). Running it twice gives the same README.
#
# Usage, from any directory:
#   scripts/readme.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -s results/tables.md ]]; then
  echo "readme.sh: results/tables.md is missing or empty (run scripts/tables.sh > results/tables.md)" >&2
  exit 1
fi
if [[ "$(grep -c -e '^<!-- tables:start -->$' -e '^<!-- tables:end -->$' README.md)" != 2 ]]; then
  echo "readme.sh: README.md must contain each table marker line exactly once" >&2
  exit 1
fi

awk '
  /^<!-- tables:start -->$/ { print; while ((getline line < "results/tables.md") > 0) print line; skipping = 1; next }
  /^<!-- tables:end -->$/   { skipping = 0 }
  !skipping
' README.md > README.md.new
mv README.md.new README.md
echo "readme.sh: README.md tables updated from results/tables.md"
