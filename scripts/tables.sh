#!/usr/bin/env bash
# Builds the README's results tables from the files scripts/measure.sh saved, so
# no number is ever typed by hand (Constitution 5). Times are shown in
# milliseconds (ms) or microseconds (µs) and memory in megabytes (MB); each
# table states its conversions.
#
# Usage, from the repository root:
#   scripts/tables.sh > results/tables.md
# RESULTS names another directory of result files (used to test this script).
#
# Uses only POSIX awk features, so it runs with the awk shipped with macOS.

set -euo pipefail
cd "$(dirname "$0")/.."
RESULTS="${RESULTS:-results}"

files=(machine.txt python.txt go-xla.txt go-go.txt go-bench-xla.txt go-bench-go.txt whole-program.txt)
for file in "${files[@]}"; do
  if [[ ! -s "$RESULTS/$file" ]]; then
    echo "tables.sh: missing or empty $RESULTS/$file (run scripts/measure.sh first)" >&2
    exit 1
  fi
done
gomlx_version="$(awk '$1 == "github.com/gomlx/gomlx" { print $2 }' go.mod)"

awk -v gomlx="$gomlx_version" '
# median of values[1..count], sorted in place (insertion sort; counts are small)
function median(values, count,    i, j, held) {
  for (i = 2; i <= count; i++) {
    held = values[i]
    for (j = i - 1; j >= 1 && values[j] > held; j--) values[j + 1] = values[j]
    values[j + 1] = held
  }
  if (count % 2) return values[(count + 1) / 2]
  return (values[count / 2] + values[count / 2 + 1]) / 2
}

# field returns the value of key=value in the current line
function field(key,    i, pair) {
  for (i = 1; i <= NF; i++) {
    split($i, pair, "=")
    if (pair[1] == key) return pair[2]
  }
  return ""
}

# after returns the text following prefix in line, up to the first character in stops
function after(line, prefix, stops,    rest, i) {
  i = index(line, prefix)
  if (i == 0) return ""
  rest = substr(line, i + length(prefix))
  for (i = 1; i <= length(rest); i++) if (index(stops, substr(rest, i, 1))) return substr(rest, 1, i - 1)
  return rest
}

function addRow(key, label) { rows[++rowCount] = key; rowLabel[key] = label }

FILENAME ~ /machine\.txt$/ { split($0, kv, ": "); machine[kv[1]] = substr($0, length(kv[1]) + 3) }

# Python: a "<Runtime> <version>: accuracy …; parity …" line, then its benchmark line.
FILENAME ~ /python\.txt$/ && /: accuracy / {
  pythonVersion = after($0, "", ":")
  pythonAccuracy = after($0, "accuracy ", ";") * 100
  pythonLabels = after($0, "(labels ", ",")
  pythonDifference = after($0, "largest difference ", ")")
}
FILENAME ~ /python\.txt$/ && /^benchmark / {
  key = "python-" field("runtime")
  addRow(key, "Python | " pythonVersion)
  accuracy[key] = pythonAccuracy; labels[key] = pythonLabels; difference[key] = pythonDifference
  bench[key] = $0
}

# Go: "Accuracy: 97.11 % (…)", "Parity with PyTorch: PASS: labels 10000/10000 identical, largest probability difference 1.04e-07 (…", benchmark line.
FILENAME ~ /go-(xla|go)\.txt$/ && /^Accuracy: / { goAccuracy = after($0, "Accuracy: ", " ") }
FILENAME ~ /go-(xla|go)\.txt$/ && /^Parity with PyTorch: / {
  goLabels = after($0, "labels ", " "); goDifference = after($0, "largest probability difference ", " ")
}
FILENAME ~ /go-(xla|go)\.txt$/ && /^benchmark / {
  backend = field("runtime"); sub(/^gomlx-/, "", backend)
  key = "go-" backend
  addRow(key, "Go | GoMLX " gomlx ", " (backend == "xla" ? "XLA backend" : "pure-Go backend"))
  accuracy[key] = goAccuracy; labels[key] = goLabels; difference[key] = goDifference
  bench[key] = $0
}

# Monte Carlo: BenchmarkPredict/batch=N-PROCS  iterations  value unit  value unit …
FILENAME ~ /go-bench-(xla|go)\.txt$/ && /^BenchmarkPredict\/batch=/ {
  backend = FILENAME; sub(/.*go-bench-/, "", backend); sub(/\.txt$/, "", backend)
  size = $1; sub(/^BenchmarkPredict\/batch=/, "", size); sub(/-[0-9]+$/, "", size)
  if (!((backend, size) in mcRuns)) { if (backend == "xla") sizes[++sizeCount] = size }
  n = ++mcRuns[backend, size]
  for (i = 3; i < NF; i++) {
    if ($(i + 1) == "ns/op") mcNs[backend, size, n] = $i
    if ($(i + 1) == "B/op") mcBytes[backend, size, n] = $i
  }
}

FILENAME ~ /whole-program\.txt$/ && /^wholeprogram / { whole[++wholeCount] = $0 }

END {
  print "Measured on " machine["machine"] " · Go " machine["go"] " · Python " machine["python"] " · " machine["measured"]
  print ""

  print "### Do the predictions match? (all 10,000 MNIST test images)"
  print ""
  print "| Language | Runtime | Accuracy (%) | Same digit as PyTorch | Largest probability difference from PyTorch |"
  print "|---|---|---:|---:|---:|"
  for (r = 1; r <= rowCount; r++) {
    key = rows[r]
    printf "| %s | %.2f | %s | %s |\n", rowLabel[key], accuracy[key], labels[key], (key == "python-pytorch" ? "reference" : difference[key])
  }
  print ""
  print "Parity requires the same digit for every image and probabilities within 1 × 10⁻⁵ of PyTorch'"'"'s (spec Q5)."
  print ""

  print "### Prediction speed"
  print ""
  print "| Language | Runtime | Load model (ms) | First prediction (ms) | 10,000 images in one call (ms) | Relative to PyTorch | One image, median (µs) | One image, 95th percentile (µs) | Relative to PyTorch |"
  print "|---|---|---:|---:|---:|---:|---:|---:|---:|"
  $0 = bench["python-pytorch"]; baseBulk = field("bulk_ms"); baseLatency = field("latency_p50_us")
  for (r = 1; r <= rowCount; r++) {
    $0 = bench[rows[r]]
    printf "| %s | %.2f | %.2f | %.2f | %.2f× | %.1f | %.1f | %.2f× |\n", rowLabel[rows[r]], field("load_ms"), field("first_ms"),
      field("bulk_ms"), field("bulk_ms") / baseBulk, field("latency_p50_us"), field("latency_p95_us"), field("latency_p50_us") / baseLatency
    bulkRuns = field("bulk_runs"); latencyCalls = field("latency_calls")
  }
  print ""
  printf "10,000 images: median of %d calls. One image: %d calls after 20 warm-up calls. ", bulkRuns, latencyCalls
  print "Relative to PyTorch: time ÷ PyTorch'"'"'s time, so below 1× is faster. "
  print "The first prediction includes compiling the model for that batch size (GoMLX) or preparing it (Python). "
  print "1 ms = 1,000 µs."
  print ""

  print "### Monte Carlo benchmark (Go): random inputs, increasing batch size"
  print ""
  print "| Batch size | XLA: ms per batch | XLA: µs per image | XLA: MB allocated | Pure Go: ms per batch | Pure Go: µs per image | Pure Go: MB allocated |"
  print "|---:|---:|---:|---:|---:|---:|---:|"
  for (s = 1; s <= sizeCount; s++) {
    size = sizes[s]
    printf "| %d", size
    split("xla go", backends, " ")
    for (b = 1; b <= 2; b++) {
      backend = backends[b]
      for (i = 1; i <= mcRuns[backend, size]; i++) { t[i] = mcNs[backend, size, i]; m[i] = mcBytes[backend, size, i] }
      ns = median(t, mcRuns[backend, size]); allocated = median(m, mcRuns[backend, size])
      printf " | %.3f | %.2f | %.2f", ns / 1e6, ns / 1e3 / size, allocated / 1048576
    }
    print " |"
  }
  print ""
  printf "Median of %d `go test -bench` runs per backend; ns/op ÷ 1,000,000 = ms; ÷ batch size ÷ 1,000 = µs per image; ", mcRuns["xla", sizes[1]]
  print "B/op ÷ 1,048,576 = MB. MB allocated counts Go memory only; XLA'"'"'s own C++ memory is not included."
  print ""

  print "### Whole program, start-up included: load, predict all 10,000 images once, check parity"
  print ""
  print "| Program | Median wall time (ms) | Peak memory, max RSS (MB) |"
  print "|---|---:|---:|"
  label["go-xla"] = "Go, GoMLX XLA backend (`predict-digits -runs 0`)"
  label["go-go"] = "Go, GoMLX pure-Go backend (`predict-digits -runs 0`)"
  label["python-pytorch"] = "Python, PyTorch (`bench_python.py --predict-only`)"
  label["python-onnxruntime"] = "Python, ONNX Runtime (`bench_python.py --predict-only`)"
  for (w = 1; w <= wholeCount; w++) {
    $0 = whole[w]
    printf "| %s | %.1f | %.1f |\n", label[field("language")], field("median_wall_ms"), field("median_maxrss_mb")
    wholeRuns = field("runs")
  }
  print ""
  printf "Median of %d runs measured by `cmd/measure`: wall time from Go'"'"'s monotonic clock, peak memory from the operating system'"'"'s rusage report. ", wholeRuns
  print "Includes starting the program (and for Python, importing its libraries). The Go timings exclude XLA'"'"'s one-time plugin download."
}
' "${files[@]/#/$RESULTS/}"
