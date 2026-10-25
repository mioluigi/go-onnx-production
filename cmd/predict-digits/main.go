// Command predict-digits runs a handwritten-digit classifier, trained in
// PyTorch and exported to ONNX, in Go with GoMLX.
//
// With no arguments it checks that Go's predictions on the 10,000 MNIST test
// images match PyTorch's, then times loading and prediction the same way
// python/bench_python.py times Python. With -input it predicts the digit in
// each row of a CSV file. The model and test set are embedded, so the program
// needs no other files. Results go to standard output, logs to standard error.
package main

import (
	"bytes"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/mioluigi/go-onnx-production/assets"
	"github.com/mioluigi/go-onnx-production/idx"
	"github.com/mioluigi/go-onnx-production/inference"
	"github.com/mioluigi/go-onnx-production/parity"
)

// Exit codes tell scripts what happened without parsing messages.
const (
	exitOK        = 0 // results printed
	exitDataError = 1 // a data or model problem, including a parity FAIL; the message explains why
	exitUsage     = 2 // the command line was wrong; usage is printed
)

const (
	pixelsPerImage  = 28 * 28
	parityTolerance = 1e-5 // spec Q5
	builtInModel    = "built-in mnist_mlp.onnx"

	// Timing repetitions, the same as python/bench_python.py's defaults.
	defaultRuns = 1000 // single-image calls timed
	maxBulkRuns = 20   // 10,000-image predictions timed
	warmUpCalls = 20   // untimed single-image calls before the latency timing
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// options holds the parsed command line.
type options struct {
	inputPath, modelPath, backend string
	runs                          int
	verbose                       bool
	backendChosen                 bool // -backend given: no fallback to the pure-Go backend
}

// run carries out one invocation and returns the exit code. Keeping it apart
// from main makes the whole program testable without terminating the test.
func run(args []string, stdout, stderr io.Writer) int {
	opts, err := parseOptions(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return exitOK
	}
	if err != nil {
		return exitUsage
	}
	logger := newLogger(stderr, opts.verbose)
	started := time.Now()
	logger.Debug("options", "input", opts.inputPath, "model", opts.modelPath, "backend", opts.backend, "runs", opts.runs)

	if err := execute(opts, stdout, logger); err != nil {
		fmt.Fprintf(stderr, "predict-digits: %v\n", err)
		return exitDataError
	}
	logger.Info("finished", "elapsed", time.Since(started))
	return exitOK
}

// parseOptions reads and checks the command line. The flag package prints its
// own message and the usage text for malformed flags.
func parseOptions(args []string, stderr io.Writer) (options, error) {
	opts := options{backend: defaultBackend}
	flags := flag.NewFlagSet("predict-digits", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&opts.inputPath, "input", "", "CSV `file` of images to predict (default: check parity and time the built-in test set)")
	flags.StringVar(&opts.modelPath, "model", "", "ONNX model `file` (default: the "+builtInModel+")")
	flags.StringVar(&opts.backend, "backend", defaultBackend, "GoMLX `backend`: go (pure Go, every build) or xla (macOS and Linux builds with CGo;\nthe default there, falling back to go if XLA cannot start)")
	flags.IntVar(&opts.runs, "runs", defaultRuns, "single-image predictions to time; 0 skips the timings")
	flags.BoolVar(&opts.verbose, "verbose", false, "show detailed (debug) log messages")
	flags.Usage = func() {
		fmt.Fprint(flags.Output(), `Usage: predict-digits [-input digits.csv] [-model file.onnx] [-backend go|xla] [-runs n] [-verbose]

Runs a handwritten-digit classifier, trained in PyTorch and exported to ONNX, in Go with GoMLX.

Without -input: checks that the predictions for the 10,000 MNIST test images match
PyTorch's, then times loading the model, the first prediction, all 10,000 images in
one call (median of up to 20 runs), and one image at a time (median and 95th percentile).

With -input: predicts the digit in each row of a CSV file and prints row,digit,probability.
Each row holds the 784 pixels of one 28 x 28 image, row by row, as whole numbers
from 0 (background) to 255 (ink), with no header line.

Flags:
`)
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	flags.Visit(func(f *flag.Flag) { opts.backendChosen = opts.backendChosen || f.Name == "backend" })
	var problem string
	switch name, _, _ := strings.Cut(opts.backend, ":"); {
	case flags.NArg() > 0:
		problem = fmt.Sprintf("unexpected argument %q (use -input to name a file)", flags.Arg(0))
	case opts.runs < 0:
		problem = fmt.Sprintf("-runs must be 0 or more, not %d", opts.runs)
	case name != "go" && name != "xla":
		problem = fmt.Sprintf("unknown backend %q; choose go or xla", opts.backend)
	}
	if problem != "" {
		fmt.Fprintln(stderr, problem)
		flags.Usage()
		return options{}, errors.New(problem)
	}
	return opts, nil
}

// newLogger writes timestamped key=value logs to standard error (spec FR-9).
func newLogger(stderr io.Writer, verbose bool) *slog.Logger {
	level := slog.LevelInfo
	if verbose {
		level = slog.LevelDebug
	}
	return slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: level}))
}

// execute does the work of one run: CSV prediction, or parity check and timings.
func execute(opts options, stdout io.Writer, logger *slog.Logger) error {
	var userImages [][]float32
	if opts.inputPath != "" {
		var err error
		if userImages, err = readImagesCSV(opts.inputPath); err != nil {
			return err
		}
		logger.Info("input read", "file", opts.inputPath, "images", len(userImages))
	}

	m, source, loadTime, err := loadModel(opts, logger)
	if err != nil {
		return err
	}
	defer m.Close()
	logger.Info("model loaded", "model", source, "backend", m.BackendName(), "elapsed", loadTime)

	if userImages != nil {
		return printPredictions(stdout, m, userImages)
	}
	return checkAndTime(stdout, logger, m, source, loadTime, opts.runs)
}

// loadModel reads the chosen model and prepares it with the chosen backend,
// returning the time taken: the "load" measure (plan section 8). When the
// default backend (XLA) is unavailable, for example because its plugin cannot
// be downloaded, it falls back to the pure-Go backend so that a run with no
// arguments still works; a backend chosen with -backend never falls back.
func loadModel(opts options, logger *slog.Logger) (m *inference.Model, source string, loadTime time.Duration, err error) {
	start := time.Now()
	onnxBytes, source := assets.ModelONNX, builtInModel
	if opts.modelPath != "" {
		if onnxBytes, err = os.ReadFile(opts.modelPath); err != nil {
			return nil, "", 0, fmt.Errorf("reading model: %w", err)
		}
		source = opts.modelPath
	}
	readTime := time.Since(start)

	start = time.Now()
	m, err = inference.Load(onnxBytes, inference.Options{Backend: opts.backend})
	if errors.Is(err, inference.ErrBackend) && !opts.backendChosen && opts.backend != "go" {
		logger.Warn("backend unavailable; using the pure-Go backend instead", "backend", opts.backend, "reason", err.Error())
		start = time.Now()
		m, err = inference.Load(onnxBytes, inference.Options{Backend: "go"})
	}
	loadTime = readTime + time.Since(start)
	if err != nil {
		return nil, "", 0, fmt.Errorf("loading %s: %w", source, err)
	}
	if m.NumFeatures() != pixelsPerImage {
		m.Close()
		return nil, "", 0, fmt.Errorf("model %s takes %d input values; want %d (one 28 x 28 image)", source, m.NumFeatures(), pixelsPerImage)
	}
	return m, source, loadTime, nil
}

// toModelInput applies the input contract (spec FR-3a), exactly as
// python/mnist_io.py does: pixels in row order, as float32, divided by 255.
func toModelInput(pixels []uint8) []float32 {
	values := make([]float32, len(pixels))
	for i, pixel := range pixels {
		values[i] = float32(pixel) / 255
	}
	return values
}

// readImagesCSV reads one image per row: 784 whole numbers from 0 to 255.
// Errors name the line and value so the user can fix the file (spec FR-12).
func readImagesCSV(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening input: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // checked below, with a clearer message
	reader.ReuseRecord = true
	var images [][]float32
	pixels := make([]uint8, pixelsPerImage)
	for line := 1; ; line++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if len(record) != pixelsPerImage {
			return nil, fmt.Errorf("%s: line %d has %d values; want %d (one 28 x 28 image)", path, line, len(record), pixelsPerImage)
		}
		for i, field := range record {
			value, err := strconv.ParseUint(strings.TrimSpace(field), 10, 8)
			if err != nil {
				return nil, fmt.Errorf("%s: line %d, value %d %q is not a whole number from 0 to 255", path, line, i+1, field)
			}
			pixels[i] = uint8(value)
		}
		images = append(images, toModelInput(pixels))
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("%s: no rows to predict", path)
	}
	return images, nil
}

// printPredictions writes row,digit,probability for each image, as CSV.
func printPredictions(stdout io.Writer, m *inference.Model, images [][]float32) error {
	probabilities, err := m.Predict(images)
	if err != nil {
		return err
	}
	var out bytes.Buffer
	out.WriteString("row,digit,probability\n")
	for row, digit := range parity.Labels(probabilities) {
		fmt.Fprintf(&out, "%d,%d,%.6f\n", row+1, digit, probabilities[row][digit])
	}
	_, err = stdout.Write(out.Bytes())
	return err
}

// testSet is the embedded MNIST test set and PyTorch's predictions for it.
type testSet struct {
	inputs    [][]float32
	labels    []int
	reference [][]float32
}

// loadTestSet decodes the embedded files and applies the input contract.
func loadTestSet() (testSet, error) {
	images, err := idx.Read(bytes.NewReader(assets.TestImages))
	if err != nil {
		return testSet{}, fmt.Errorf("test images: %w", err)
	}
	labels, err := idx.Read(bytes.NewReader(assets.TestLabels))
	if err != nil {
		return testSet{}, fmt.Errorf("test labels: %w", err)
	}
	reference, err := idx.Read(bytes.NewReader(assets.PyTorchProbabilities))
	if err != nil {
		return testSet{}, fmt.Errorf("PyTorch probabilities: %w", err)
	}
	referenceRows, err := reference.Rows()
	if err != nil {
		return testSet{}, fmt.Errorf("PyTorch probabilities: %w", err)
	}
	count := len(referenceRows)
	if !slices.Equal(images.Shape, []int{count, 28, 28}) || !slices.Equal(labels.Shape, []int{count}) {
		return testSet{}, fmt.Errorf("embedded test set does not fit together: images %v, labels %v, probabilities %d rows",
			images.Shape, labels.Shape, count)
	}

	set := testSet{inputs: make([][]float32, count), labels: make([]int, count), reference: referenceRows}
	for i := range count {
		set.inputs[i] = toModelInput(images.Bytes[i*pixelsPerImage : (i+1)*pixelsPerImage])
		set.labels[i] = int(labels.Bytes[i])
	}
	return set, nil
}

// checkAndTime checks parity with PyTorch on the whole test set and, if it
// passes and runs > 0, times the model (Constitution 1: parity before speed).
func checkAndTime(stdout io.Writer, logger *slog.Logger, m *inference.Model, source string, loadTime time.Duration, runs int) error {
	set, err := loadTestSet()
	if err != nil {
		return err
	}
	count := len(set.inputs)

	// The first prediction includes compiling the graph for a batch of one.
	firstStart := time.Now()
	if _, err := m.Predict(set.inputs[:1]); err != nil {
		return err
	}
	firstTime := time.Since(firstStart)

	probabilities, err := m.Predict(set.inputs)
	if err != nil {
		return err
	}
	report, err := parity.Compare(probabilities, set.reference, parityTolerance)
	if err != nil {
		return err
	}
	correct := 0
	for i, label := range parity.Labels(probabilities) {
		if label == set.labels[i] {
			correct++
		}
	}
	logger.Info("parity checked", "pass", report.Pass(), "max_abs_diff", report.MaxAbsDiff,
		"row", report.MaxDiffRow, "class", report.MaxDiffClass)

	fmt.Fprintf(stdout, "MNIST digit classifier: PyTorch model run in Go with GoMLX\n")
	fmt.Fprintf(stdout, "Model:    %s (%d inputs, %d classes)\n", source, m.NumFeatures(), m.NumClasses())
	fmt.Fprintf(stdout, "Backend:  %s\n", m.BackendName())
	fmt.Fprintf(stdout, "Test set: %s MNIST test images\n\n", thousands(count))
	fmt.Fprintf(stdout, "Accuracy: %.2f %% (%s of %s correct)\n", 100*float64(correct)/float64(count), thousands(correct), thousands(count))
	fmt.Fprintf(stdout, "Parity with PyTorch: %s\n", report)
	if !report.Pass() {
		return errors.New("Go's predictions do not match PyTorch's, so no timings are reported")
	}
	if runs == 0 {
		return nil
	}

	bulkRuns := min(maxBulkRuns, runs)
	bulk, err := medianBulkTime(m, set.inputs, bulkRuns)
	if err != nil {
		return err
	}
	p50, p95, err := latency(m, set.inputs, runs)
	if err != nil {
		return err
	}
	logger.Info("timings done", "bulk_runs", bulkRuns, "latency_calls", runs)

	fmt.Fprintf(stdout, "\nTimings (%s backend)\n", m.BackendName())
	fmt.Fprintf(stdout, "  Load model           %10.2f ms\n", milliseconds(loadTime))
	fmt.Fprintf(stdout, "  First prediction     %10.2f ms   one image, includes compiling the graph\n", milliseconds(firstTime))
	fmt.Fprintf(stdout, "  %s images         %10.2f ms   one call, median of %d runs (%.2f µs per image)\n",
		thousands(count), milliseconds(bulk), bulkRuns, microseconds(bulk)/float64(count))
	fmt.Fprintf(stdout, "  One image, median    %10.1f µs   %d calls after %d warm-up calls\n", p50, runs, warmUpCalls)
	fmt.Fprintf(stdout, "  One image, 95th pct  %10.1f µs\n\n", p95)
	fmt.Fprintf(stdout, "benchmark language=go runtime=gomlx-%s load_ms=%.2f first_ms=%.2f bulk_ms=%.2f latency_p50_us=%.1f latency_p95_us=%.1f bulk_runs=%d latency_calls=%d\n",
		m.BackendName(), milliseconds(loadTime), milliseconds(firstTime), milliseconds(bulk), p50, p95, bulkRuns, runs)
	return nil
}

// medianBulkTime predicts all inputs in one call, runs times, and returns the median time.
func medianBulkTime(m *inference.Model, inputs [][]float32, runs int) (time.Duration, error) {
	times := make([]float64, runs)
	for i := range times {
		start := time.Now()
		if _, err := m.Predict(inputs); err != nil {
			return 0, err
		}
		times[i] = float64(time.Since(start))
	}
	slices.Sort(times)
	return time.Duration(percentile(times, 50)), nil
}

// latency times calls predictions of one image each, after warm-up, cycling
// through the inputs, and returns the median and 95th percentile in microseconds.
func latency(m *inference.Model, inputs [][]float32, calls int) (p50, p95 float64, err error) {
	for i := range warmUpCalls {
		if _, err := m.Predict(inputs[i%len(inputs) : i%len(inputs)+1]); err != nil {
			return 0, 0, err
		}
	}
	times := make([]float64, calls)
	for i := range times {
		row := i % len(inputs)
		start := time.Now()
		if _, err := m.Predict(inputs[row : row+1]); err != nil {
			return 0, 0, err
		}
		times[i] = microseconds(time.Since(start))
	}
	slices.Sort(times)
	return percentile(times, 50), percentile(times, 95), nil
}

// percentile returns the p-th percentile (0–100) of sorted values, interpolating
// linearly between neighbors as NumPy's percentile does by default, so Go and
// Python report the same statistic.
func percentile(sorted []float64, p float64) float64 {
	rank := p / 100 * float64(len(sorted)-1)
	below := int(rank)
	if below >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	return sorted[below] + (rank-float64(below))*(sorted[below+1]-sorted[below])
}

func milliseconds(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func microseconds(d time.Duration) float64 { return float64(d) / float64(time.Microsecond) }

// thousands formats a count with comma separators, as in "10,000".
func thousands(n int) string {
	digits := strconv.Itoa(n)
	var out strings.Builder
	for i, digit := range digits {
		if i > 0 && (len(digits)-i)%3 == 0 {
			out.WriteByte(',')
		}
		out.WriteRune(digit)
	}
	return out.String()
}
