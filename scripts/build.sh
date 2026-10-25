#!/usr/bin/env bash
# Builds every executable into dist/ (spec AC-8, plan section 9):
#   dist/predict-digits      command-line program for this computer (XLA and pure Go on macOS/Linux)
#   dist/predict-digits.exe  Windows program (64-bit Intel/AMD), cross-compiled, pure-Go backend only
#   dist/PredictDigits.app   macOS application (only when run on an Apple Silicon Mac)
#
# Go cross-compiles by setting GOOS and GOARCH, so no Windows machine is needed.
# XLA needs CGo, which cross-compiling turns off, so the .exe leaves XLA out
# automatically (cmd/predict-digits/backend_xla.go has a build constraint) and
# uses GoMLX's pure-Go backend. The model and test set are embedded, so each
# executable runs on its own with no other files.
#
# Usage, from any directory:
#   scripts/build.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist

go build -o dist/predict-digits ./cmd/predict-digits
echo "Built dist/predict-digits"

CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o dist/predict-digits.exe ./cmd/predict-digits
echo "Built dist/predict-digits.exe (pure-Go backend)"

if [[ "$(uname -sm)" == "Darwin arm64" ]]; then
  scripts/build-macos-app.sh
else
  echo "Skipped PredictDigits.app (Apple Silicon Macs only)"
fi
