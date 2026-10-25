#!/usr/bin/env bash
# Runs every measurement for the README (task T15, plan section 8) and saves the
# raw output in results/. Nothing in the README is typed by hand: scripts/tables.sh
# turns these files into tables.
#
#   results/machine.txt          machine, operating system, Go and Python versions
#   results/python.txt           python/bench_python.py: PyTorch and ONNX Runtime timings
#   results/go-xla.txt           predict-digits -backend xla: parity check and timings
#   results/go-go.txt            predict-digits -backend go: parity check and timings
#   results/go-bench-xla.txt     Monte Carlo benchmark, XLA backend (go test -bench, -count runs)
#   results/go-bench-go.txt      Monte Carlo benchmark, pure-Go backend
#   results/cpu-profile.txt      CPU profile of 10,000-image predictions (pure-Go backend)
#   results/memory-profile.txt   memory-allocation profile of the same
#   results/whole-program.txt    complete programs, start-up included: median time and peak memory
#
# Run on macOS or Linux from the repository root, with the Python virtual
# environment active, the computer plugged in, and other programs closed:
#   scripts/measure.sh
# COUNT (benchmark repeats, default 5) and RUNS (whole-program repeats, default 5)
# can be set, e.g. COUNT=10 RUNS=11 scripts/measure.sh

set -euo pipefail

COUNT="${COUNT:-5}"
RUNS="${RUNS:-5}"
cd "$(dirname "$0")/.."

for setting in COUNT RUNS; do
  if ! [[ "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "measure.sh: $setting must be a positive whole number, got '${!setting}'" >&2
    exit 2
  fi
done
if ! python -c "import torch, onnxruntime" 2>/dev/null; then
  echo "measure.sh: activate the Python virtual environment first (source .venv/bin/activate)" >&2
  exit 2
fi

step() { echo "measure.sh: $*" >&2; }
trap 'echo "measure.sh: STOPPED because the step above failed; results/ is incomplete, so do not run tables.sh" >&2' ERR

mkdir -p results dist
machine="$(uname -sm)"
if [[ "$(uname)" == "Darwin" ]]; then
  machine="$machine · $(sysctl -n machdep.cpu.brand_string) · $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1 / 1073741824}') · macOS $(sw_vers -productVersion)"
fi
{
  echo "machine: $machine"
  echo "go: $(go version | cut -d' ' -f3)"
  echo "python: $(python --version | cut -d' ' -f2)"
  echo "measured: $(date '+%Y-%m-%d %H:%M %Z')"
} > results/machine.txt

step "building the programs"
go build -o dist/predict-digits ./cmd/predict-digits
go build -o dist/measure ./cmd/measure

step "Python timings (PyTorch, ONNX Runtime)"
python python/bench_python.py > results/python.txt

for backend in xla go; do
  step "Go timings, $backend backend (parity is checked first; a FAIL stops here)"
  dist/predict-digits -backend "$backend" > "results/go-$backend.txt"

  step "Monte Carlo benchmark, $backend backend, $COUNT runs"
  GOMLX_BACKEND="$backend" go test -run='^$' -bench=Predict -benchmem -count="$COUNT" ./inference/ > "results/go-bench-$backend.txt"
done

# The pure-Go backend is profiled because its work is Go code that pprof can see;
# XLA's work happens inside its C++ plugin, which Go's profiler cannot look into.
step "CPU and memory profiles, pure-Go backend"
GOMLX_BACKEND=go go test -run='^$' -bench='Predict/batch=10000$' -benchtime=20x \
  -o dist/inference.test -cpuprofile dist/cpu.prof -memprofile dist/memory.prof ./inference/ > /dev/null
go tool pprof -top -nodecount=25 dist/inference.test dist/cpu.prof > results/cpu-profile.txt 2>/dev/null
go tool pprof -top -nodecount=25 -sample_index=alloc_space dist/inference.test dist/memory.prof > results/memory-profile.txt 2>/dev/null

step "whole programs, $RUNS runs each"
dist/measure -runs "$RUNS" \
  "go-xla=dist/predict-digits -backend xla -runs 0" \
  "go-go=dist/predict-digits -backend go -runs 0" \
  "python-pytorch=python python/bench_python.py --runtime pytorch --predict-only" \
  "python-onnxruntime=python python/bench_python.py --runtime onnxruntime --predict-only" \
  > results/whole-program.txt

step "done; now run scripts/tables.sh > results/tables.md"
