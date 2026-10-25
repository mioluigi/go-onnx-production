package main

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/mioluigi/go-onnx-production/assets"
	"github.com/mioluigi/go-onnx-production/idx"
	"github.com/mioluigi/go-onnx-production/parity"
)

// binaryPath is the program built once by TestMain and run by the end-to-end
// tests, so they exercise exactly what a user runs, including exit codes.
var binaryPath string

// coverDirVariable names a directory for coverage of the end-to-end runs. Go's
// "go test -cover" cannot see code running in a separate process, so when this
// variable is set the program is built with -cover and each run writes its
// coverage there (Go's GOCOVERDIR mechanism); "go tool covdata percent" reads it.
const coverDirVariable = "PREDICT_DIGITS_COVERDIR"

func TestMain(m *testing.M) {
	buildDir, err := os.MkdirTemp("", "predict-digits-test")
	if err != nil {
		fmt.Fprintln(os.Stderr, "creating build directory:", err)
		os.Exit(1)
	}
	binaryPath = filepath.Join(buildDir, "predict-digits")
	buildArgs := []string{"build", "-o", binaryPath}
	if coverDir := os.Getenv(coverDirVariable); coverDir != "" {
		buildArgs = append(buildArgs, "-cover")
		os.Setenv("GOCOVERDIR", coverDir) // inherited by every run of the program
	}
	build := exec.Command("go", append(buildArgs, ".")...)
	if output, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "building program: %v\n%s", err, output)
		os.RemoveAll(buildDir)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(buildDir)
	os.Exit(code)
}

// runResult is what a user would see after running the program once.
type runResult struct {
	exitCode       int
	stdout, stderr string
}

func runProgram(t *testing.T, args ...string) runResult {
	t.Helper()
	var stdout, stderr bytes.Buffer
	command := exec.Command(binaryPath, args...)
	command.Stdout, command.Stderr = &stdout, &stderr
	err := command.Run()

	var exitErr *exec.ExitError
	exitCode := 0
	if errors.As(err, &exitErr) {
		exitCode = exitErr.ExitCode()
	} else if err != nil {
		t.Fatalf("could not run program: %v", err)
	}
	return runResult{exitCode, stdout.String(), stderr.String()}
}

func (r runResult) String() string {
	return fmt.Sprintf("exit code %d\n--- stdout:\n%s--- stderr:\n%s", r.exitCode, r.stdout, r.stderr)
}

func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// benchmarkFields parses the "benchmark key=value …" line into a map.
func benchmarkFields(t *testing.T, stdout string) map[string]string {
	t.Helper()
	for line := range strings.Lines(stdout) {
		if rest, ok := strings.CutPrefix(line, "benchmark "); ok {
			fields := map[string]string{}
			for _, pair := range strings.Fields(rest) {
				key, value, _ := strings.Cut(pair, "=")
				fields[key] = value
			}
			return fields
		}
	}
	return nil
}

// TestNoArguments is the default run (FR-11, AC-7): parity check, then timings.
func TestNoArguments(t *testing.T) {
	result := runProgram(t)
	if result.exitCode != exitOK {
		t.Fatalf("want exit code %d\n%s", exitOK, result)
	}
	for _, want := range []string{"Accuracy", "Parity with PyTorch: PASS", "Load model", "First prediction", "10,000 images", "One image"} {
		if !strings.Contains(result.stdout, want) {
			t.Errorf("stdout lacks %q\n%s", want, result)
		}
	}
	fields := benchmarkFields(t, result.stdout)
	if fields == nil {
		t.Fatalf("no benchmark line\n%s", result)
	}
	if fields["language"] != "go" || !strings.HasPrefix(fields["runtime"], "gomlx-") {
		t.Errorf("benchmark language=%q runtime=%q; want go, gomlx-<backend>", fields["language"], fields["runtime"])
	}
	for _, key := range []string{"load_ms", "first_ms", "bulk_ms", "latency_p50_us", "latency_p95_us"} {
		if value, err := strconv.ParseFloat(fields[key], 64); err != nil || value <= 0 {
			t.Errorf("benchmark %s=%q; want a positive number", key, fields[key])
		}
	}
	if fields["bulk_runs"] != "20" || fields["latency_calls"] != "1000" {
		t.Errorf("bulk_runs=%s latency_calls=%s; want 20 and 1000, as in python/bench_python.py", fields["bulk_runs"], fields["latency_calls"])
	}
	if strings.Contains(result.stdout, "level=") || !strings.Contains(result.stderr, "level=INFO") {
		t.Errorf("logs must go to stderr only (FR-9)\n%s", result)
	}
}

// TestParityOnAllTestImages is AC-1: Go matches PyTorch on all 10,000 test images
// with the pure-Go backend, which every build has.
func TestParityOnAllTestImages(t *testing.T) {
	result := runProgram(t, "-backend", "go", "-runs", "0")
	if result.exitCode != exitOK || !strings.Contains(result.stdout, "PASS: labels 10000/10000 identical") {
		t.Fatalf("want PASS on 10000 images and exit code %d\n%s", exitOK, result)
	}
	if benchmarkFields(t, result.stdout) != nil {
		t.Errorf("-runs 0 must skip the timings\n%s", result)
	}
}

func TestVerboseShowsDebugLogs(t *testing.T) {
	result := runProgram(t, "-runs", "0", "-verbose")
	if result.exitCode != exitOK || !strings.Contains(result.stderr, "level=DEBUG") {
		t.Errorf("want exit 0 and DEBUG logs on stderr\n%s", result)
	}
}

// testImageRows returns the first n embedded test images as CSV lines of pixels,
// and the digits PyTorch predicted for them.
func testImageRows(t *testing.T, n int) (lines []string, digits []int) {
	t.Helper()
	images, err := idx.Read(bytes.NewReader(assets.TestImages))
	if err != nil {
		t.Fatal(err)
	}
	reference, err := idx.Read(bytes.NewReader(assets.PyTorchProbabilities))
	if err != nil {
		t.Fatal(err)
	}
	probabilities, err := reference.Rows()
	if err != nil {
		t.Fatal(err)
	}
	for image := range n {
		pixels := images.Bytes[image*pixelsPerImage : (image+1)*pixelsPerImage]
		values := make([]string, len(pixels))
		for i, pixel := range pixels {
			values[i] = strconv.Itoa(int(pixel))
		}
		lines = append(lines, strings.Join(values, ","))
	}
	return lines, parity.Labels(probabilities[:n])
}

// TestInputCSV predicts a user's images (FR-11): one output row per input row.
func TestInputCSV(t *testing.T) {
	lines, want := testImageRows(t, 5)
	result := runProgram(t, "-input", writeFile(t, "digits.csv", strings.Join(lines, "\n")+"\n"))
	if result.exitCode != exitOK {
		t.Fatalf("want exit code %d\n%s", exitOK, result)
	}
	output := strings.Split(strings.TrimSpace(result.stdout), "\n")
	if len(output) != len(want)+1 || output[0] != "row,digit,probability" {
		t.Fatalf("want a header and %d rows\n%s", len(want), result)
	}
	for i, line := range output[1:] {
		fields := strings.Split(line, ",")
		probability, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if len(fields) != 3 || fields[0] != strconv.Itoa(i+1) || fields[1] != strconv.Itoa(want[i]) ||
			err != nil || probability < 0 || probability > 1 {
			t.Errorf("output row %q; want row %d, digit %d, a probability", line, i+1, want[i])
		}
	}
}

// TestInvalidInput covers FR-12: a clear message, exit code 1, and no results.
func TestInvalidInput(t *testing.T) {
	lines, _ := testImageRows(t, 2)
	tests := []struct {
		name, content, wantMessage string
	}{
		{"empty file", "", "no rows"},
		{"too few values", lines[0] + "\n1,2,3\n", "line 2 has 3 values; want 784"},
		{"not a number", "0,0,abc" + strings.Repeat(",0", 781) + "\n", `line 1, value 3 "abc"`},
		{"above 255", lines[0] + "\n256" + strings.Repeat(",0", 783) + "\n", `line 2, value 1 "256"`},
		{"negative", "-1" + strings.Repeat(",0", 783) + "\n", `line 1, value 1 "-1"`},
		{"not whole", "0.5" + strings.Repeat(",0", 783) + "\n", `line 1, value 1 "0.5"`},
		{"NaN", "NaN" + strings.Repeat(",0", 783) + "\n", `line 1, value 1 "NaN"`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := runProgram(t, "-input", writeFile(t, "digits.csv", tc.content))
			if result.exitCode != exitDataError || !strings.Contains(result.stderr, tc.wantMessage) || result.stdout != "" {
				t.Errorf("want exit code %d, %q on stderr, nothing on stdout\n%s", exitDataError, tc.wantMessage, result)
			}
		})
	}
}

func TestInvalidModel(t *testing.T) {
	tests := []struct {
		name, path, wantMessage string
	}{
		{"missing file", filepath.Join(t.TempDir(), "absent.onnx"), "no such file"},
		{"not ONNX", writeFile(t, "model.onnx", "hello"), "not a readable ONNX model"},
		{"wrong number of inputs", "../../inference/testdata/softmax_3.onnx", "3 input values; want 784"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := runProgram(t, "-model", tc.path, "-runs", "0")
			if result.exitCode != exitDataError || !strings.Contains(result.stderr, tc.wantMessage) {
				t.Errorf("want exit code %d and %q on stderr\n%s", exitDataError, tc.wantMessage, result)
			}
		})
	}
}

// TestParityFailure: a model that disagrees with PyTorch is reported as FAIL,
// exits 1, and gets no timings (Constitution 1: parity before speed).
func TestParityFailure(t *testing.T) {
	result := runProgram(t, "-model", "../../inference/testdata/always_zero_784.onnx", "-backend", "go")
	if result.exitCode != exitDataError || !strings.Contains(result.stdout, "Parity with PyTorch: FAIL") ||
		benchmarkFields(t, result.stdout) != nil || !strings.Contains(result.stderr, "do not match") {
		t.Errorf("want FAIL, exit code %d, no timings\n%s", exitDataError, result)
	}
}

// TestChosenBackendUnavailable: a backend named with -backend that cannot start
// is an error (exit 1), never a silent fallback.
func TestChosenBackendUnavailable(t *testing.T) {
	result := runProgram(t, "-backend", "xla:no-such-plugin", "-runs", "0")
	if result.exitCode != exitDataError || !strings.Contains(result.stderr, "backend unavailable") || result.stdout != "" {
		t.Errorf("want exit code %d and %q on stderr\n%s", exitDataError, "backend unavailable", result)
	}
}

func TestCommandLineMistakes(t *testing.T) {
	for _, args := range [][]string{
		{"-no-such-flag"},
		{"extra-argument"},
		{"-runs", "-1"},
		{"-runs", "many"},
		{"-backend", "gpu-please"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			result := runProgram(t, args...)
			if result.exitCode != exitUsage || result.stdout != "" || !strings.Contains(result.stderr, "Usage") {
				t.Errorf("want exit code %d, usage on stderr, nothing on stdout\n%s", exitUsage, result)
			}
		})
	}
}

func TestHelp(t *testing.T) {
	result := runProgram(t, "-h")
	if result.exitCode != exitOK || !strings.Contains(result.stderr, "Usage") {
		t.Errorf("want exit code %d and usage\n%s", exitOK, result)
	}
}

// TestToModelInput checks the input contract (FR-3a): pixel ÷ 255, as float32,
// exactly as python/mnist_io.py computes it.
func TestToModelInput(t *testing.T) {
	got := toModelInput([]uint8{0, 51, 128, 255})
	want := []float32{0, float32(51) / 255, float32(128) / 255, 1}
	if len(got) != len(want) {
		t.Fatalf("toModelInput returned %d values, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("value %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestPercentile checks the interpolation NumPy's percentile uses by default,
// so the Go and Python latency percentiles mean the same thing.
func TestPercentile(t *testing.T) {
	tests := []struct {
		sorted []float64
		p      float64
		want   float64
	}{
		{[]float64{1, 2, 3, 4, 5}, 50, 3},
		{[]float64{1, 2, 3, 4, 5}, 95, 4.8}, // numpy.percentile([1,2,3,4,5], 95) == 4.8
		{[]float64{1, 2, 3, 4}, 50, 2.5},
		{[]float64{7}, 95, 7},
		{[]float64{10, 20}, 0, 10},
		{[]float64{10, 20}, 100, 20},
	}
	for _, tc := range tests {
		if got := percentile(tc.sorted, tc.p); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("percentile(%v, %v) = %v, want %v", tc.sorted, tc.p, got, tc.want)
		}
	}
}
