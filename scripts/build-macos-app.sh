#!/usr/bin/env bash
# Builds dist/PredictDigits.app, a macOS application bundle for the rubric's
# ".app" requirement (decision Q8). Adapted from the Week 4 project.
#
# The program is a command-line tool, and a double-clicked app has no terminal
# to show its output. So the bundle is a thin wrapper: its launcher opens
# Terminal, runs the program there (parity check, then timings), and waits for
# Return so the results stay on screen. The program itself is unchanged.
#
# Apple Silicon only: GoMLX's XLA backend has no build for Intel Macs, and it
# needs CGo, so the program is built here, natively, rather than cross-compiled.
#
# Usage, from any directory on an Apple Silicon Mac:
#   scripts/build-macos-app.sh            then double-click dist/PredictDigits.app

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  echo "build-macos-app.sh: run this on an Apple Silicon Mac" >&2
  exit 2
fi

app=dist/PredictDigits.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

# 1. The program, with XLA and the pure-Go backend.
go build -o "$app/Contents/Resources/predict-digits" ./cmd/predict-digits

# 2. A script Terminal runs: show the results, then wait so the window stays readable.
#    Logs (standard error) go to the usual place for macOS app logs, so the window
#    shows only the results; an error message is still shown if one occurs.
cat >"$app/Contents/Resources/run-in-terminal.command" <<'SCRIPT'
#!/bin/bash
clear
log_file="$HOME/Library/Logs/PredictDigits.log"
mkdir -p "$(dirname "$log_file")"
echo "Checking the model against PyTorch on 10,000 test images, then timing it..."
echo "(The first run downloads GoMLX's XLA plugin, about a minute.)"
echo
if ! "$(dirname "$0")/predict-digits" 2>>"$log_file"; then
  echo "The program could not finish. Details: $log_file"
  tail -n 3 "$log_file"
fi
echo
read -r -p "Press Return to close this window. " _
SCRIPT

# 3. The app's launcher: hand the script to Terminal.
cat >"$app/Contents/MacOS/PredictDigits" <<'SCRIPT'
#!/bin/bash
open -a Terminal "$(dirname "$0")/../Resources/run-in-terminal.command"
SCRIPT
chmod +x "$app/Contents/MacOS/PredictDigits" "$app/Contents/Resources/run-in-terminal.command"

# 4. The metadata macOS needs to treat the folder as an application.
cat >"$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>               <string>PredictDigits</string>
  <key>CFBundleDisplayName</key>        <string>Predict Digits</string>
  <key>CFBundleIdentifier</key>         <string>com.github.mioluigi.predictdigits</string>
  <key>CFBundleExecutable</key>         <string>PredictDigits</string>
  <key>CFBundlePackageType</key>        <string>APPL</string>
  <key>CFBundleShortVersionString</key> <string>1.0</string>
  <key>CFBundleVersion</key>            <string>1</string>
  <key>LSMinimumSystemVersion</key>     <string>13.0</string>
  <key>LSArchitecturePriority</key>     <array><string>arm64</string></array>
</dict>
</plist>
PLIST

echo "Built $app — double-click it in Finder, or run: open $app"
