// Package inference runs a classifier exported to ONNX (for example from PyTorch)
// in Go, using GoMLX (github.com/gomlx/gomlx) and its ONNX converter
// (github.com/gomlx/onnx-gomlx).
//
// It handles one-input, one-output classifiers: the input is a batch of feature
// rows (batch × features, float32) and the output a batch of class probabilities
// (batch × classes). The package performs no file or console I/O, so any Go
// program can reuse it.
//
// The pure-Go backend is always available. A program can add GoMLX's faster XLA
// backend (macOS and Linux, requires CGo) by importing
// github.com/gomlx/go-xla/compute/xla/autoinstall and choosing Options.Backend "xla".
package inference

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend" // pure-Go backend: portable, no CGo
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

// DefaultOutputName is the output Load looks for when Options.OutputName is empty.
const DefaultOutputName = "probabilities"

// Sentinel errors, for errors.Is.
var (
	ErrNotONNX       = errors.New("not a readable ONNX model")
	ErrInputShape    = errors.New("input shape not supported")
	ErrMissingOutput = errors.New("model has no such output")
	ErrBackend       = errors.New("GoMLX backend unavailable")
	ErrEmptyBatch    = errors.New("no input rows")
	ErrNotFinite     = errors.New("value is NaN or infinite")
	ErrRun           = errors.New("GoMLX could not run the model")
	ErrClosed        = errors.New("model is closed")
)

// Options adjust how a model is loaded. The zero value suits PyTorch exports.
type Options struct {
	// Backend names the GoMLX backend: "go", "xla", or "xla:cpu". Empty selects
	// GoMLX's default (the $GOMLX_BACKEND variable, else the best available).
	Backend string
	// OutputName is the output holding class probabilities; empty means DefaultOutputName.
	OutputName string
	// DisableFusion turns off GoMLX's fused dense layers. Needed for models whose
	// bias has shape [1, n], as scikit-learn exports (spike 001, finding 2).
	DisableFusion bool
}

// Model is an ONNX classifier prepared to run with GoMLX. Predict may be called
// from several goroutines at once; Close waits for running predictions.
type Model struct {
	mu          sync.RWMutex // guards closed against Close running alongside Predict
	closed      bool
	onnxModel   onnx.Model
	backend     compute.Backend
	backendName string
	executor    *model.Exec
	numFeatures int
	numClasses  int
}

// Load parses an ONNX model, checks it is a one-input classifier, loads its
// weights, and prepares an executor. GoMLX compiles the graph on the first
// Predict call for each batch size and reuses it afterwards.
func Load(onnxBytes []byte, opts Options) (*Model, error) {
	onnxModel, err := parseModel(onnxBytes)
	if err != nil {
		return nil, err
	}
	m, err := prepare(onnxModel, opts)
	if err != nil {
		onnxModel.Close()
		return nil, err
	}
	return m, nil
}

// NumFeatures is the number of values each input row must have.
func (m *Model) NumFeatures() int { return m.numFeatures }

// NumClasses is the number of probabilities in each output row.
func (m *Model) NumClasses() int { return m.numClasses }

// BackendName is the name the GoMLX backend was chosen by, such as "go" or "xla".
func (m *Model) BackendName() string { return m.backendName }

// Predict returns one row of class probabilities for each input row.
func (m *Model) Predict(inputs [][]float32) ([][]float32, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, ErrClosed
	}
	flatInputs, err := m.flatten(inputs)
	if err != nil {
		return nil, err
	}
	input := tensors.FromFlatDataAndDimensions(flatInputs, len(inputs), m.numFeatures)
	defer input.FinalizeAll()

	output, err := m.run(input)
	if err != nil {
		return nil, err
	}
	defer output.FinalizeAll()
	return m.probabilityRows(output, len(inputs))
}

// Close releases the compiled model and its backend. It is safe to call twice.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	m.closed = true
	m.executor.Finalize()
	m.backend.Finalize()
	m.onnxModel.Close()
}

// parseModel reads ONNX bytes, turning any parser failure into ErrNotONNX.
func parseModel(onnxBytes []byte) (onnx.Model, error) {
	if len(onnxBytes) == 0 {
		return nil, fmt.Errorf("%w: the model is empty", ErrNotONNX)
	}
	onnxModel, err := parser.Parse(onnxBytes)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrNotONNX, firstLine(err))
	}
	return onnxModel, nil
}

// prepare validates the model's interface, then builds the backend and executor.
func prepare(onnxModel onnx.Model, opts Options) (*Model, error) {
	inputName, numFeatures, err := singleInput(onnxModel)
	if err != nil {
		return nil, err
	}
	outputName := opts.OutputName
	if outputName == "" {
		outputName = DefaultOutputName
	}
	numClasses, err := outputClasses(onnxModel, outputName)
	if err != nil {
		return nil, err
	}
	if opts.DisableFusion {
		onnxModel.DisableFusion()
	}

	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		return nil, fmt.Errorf("%w: loading weights: %s", ErrNotONNX, firstLine(err))
	}
	backend, err := newBackend(opts.Backend)
	if err != nil {
		return nil, err
	}
	executor, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		inputs := map[string]*graph.Node{inputName: input}
		return onnxModel.CallGraph(scope, input.Graph(), inputs, outputName)[0]
	})
	if err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("%w: %s", ErrRun, firstLine(err))
	}
	return &Model{onnxModel: onnxModel, backend: backend, backendName: chosenName(opts.Backend, backend),
		executor: executor, numFeatures: numFeatures, numClasses: numClasses}, nil
}

// singleInput returns the name and feature count of the model's only input,
// which must be a matrix (batch × features) with a fixed number of features.
func singleInput(onnxModel onnx.Model) (name string, numFeatures int, err error) {
	names, shapes := onnxModel.Inputs()
	if len(names) != 1 {
		return "", 0, fmt.Errorf("%w: the model has %d inputs; one is supported", ErrInputShape, len(names))
	}
	dimensions := shapes[0].Dimensions
	if len(dimensions) != 2 || dimensions[1] <= 0 {
		return "", 0, fmt.Errorf("%w: input %q has shape %v; want (batch, features)", ErrInputShape, names[0], dimensions)
	}
	return names[0], dimensions[1], nil
}

// outputClasses returns the class count of the named output, a (batch × classes) matrix.
func outputClasses(onnxModel onnx.Model, outputName string) (int, error) {
	names, shapes := onnxModel.Outputs()
	for i, name := range names {
		if name != outputName {
			continue
		}
		dimensions := shapes[i].Dimensions
		if len(dimensions) != 2 || dimensions[1] <= 0 {
			return 0, fmt.Errorf("%w: output %q has shape %v; want (batch, classes)", ErrMissingOutput, name, dimensions)
		}
		return dimensions[1], nil
	}
	return 0, fmt.Errorf("%w: %q (the model's outputs are %v)", ErrMissingOutput, outputName, names)
}

// newBackend creates the GoMLX backend named by config, or by $GOMLX_BACKEND
// when config is empty. compute.NewWithConfig declares an error result but
// panics on an unknown backend name (found by TestLoadRejectsUnsuitableModels),
// so the name is checked against the compiled-in backends first. The recover
// remains as a backstop for panics inside a backend's own constructor, so a
// program using this package never crashes while loading a model.
func newBackend(config string) (backend compute.Backend, err error) {
	requested := config
	if requested == "" {
		requested = os.Getenv(compute.ConfigEnvVar)
	}
	if name, _, _ := strings.Cut(requested, ":"); name != "" && !slices.Contains(compute.List(), name) {
		available := compute.List()
		slices.Sort(available)
		return nil, fmt.Errorf("%w: %q is not in this build (available: %s)", ErrBackend, name, strings.Join(available, ", "))
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			backend, err = nil, fmt.Errorf("%w: %s", ErrBackend, firstLine(fmt.Errorf("%v", recovered)))
		}
	}()
	if config == "" {
		backend, err = compute.New()
	} else {
		backend, err = compute.NewWithConfig(config)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %q: %s", ErrBackend, config, firstLine(err))
	}
	return backend, nil
}

// chosenName returns the registered name of the backend ("go", "xla"), from the
// configuration or $GOMLX_BACKEND. Backend.Name is only a fallback: the pure-Go
// backend's Name returns "Go Backend", not its registered name "go" (found by
// TestLoadDescribesTheModel).
func chosenName(config string, backend compute.Backend) string {
	if config == "" {
		config = os.Getenv(compute.ConfigEnvVar)
	}
	if name, _, _ := strings.Cut(config, ":"); name != "" {
		return name
	}
	return backend.Name()
}

// flatten checks the batch and copies it into one contiguous slice, the layout
// GoMLX tensors use.
func (m *Model) flatten(inputs [][]float32) ([]float32, error) {
	if len(inputs) == 0 {
		return nil, ErrEmptyBatch
	}
	flat := make([]float32, 0, len(inputs)*m.numFeatures)
	for row, values := range inputs {
		if len(values) != m.numFeatures {
			return nil, fmt.Errorf("%w: row %d has %d values; the model expects %d", ErrInputShape, row, len(values), m.numFeatures)
		}
		if column := firstNonFinite(values); column >= 0 {
			return nil, fmt.Errorf("%w: input row %d, column %d is %v", ErrNotFinite, row, column, values[column])
		}
		flat = append(flat, values...)
	}
	return flat, nil
}

// run executes the compiled model. The converter panics on problems such as an
// unsupported ONNX operation, but Exec.Call1 already turns panics raised while
// building the graph into errors (shown by a mutation check in T08). The
// recover below is a last line of defense for a panic during execution itself,
// so that a server using this package never crashes on one bad request.
func (m *Model) run(input *tensors.Tensor) (output *tensors.Tensor, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			output, err = nil, fmt.Errorf("%w: %s", ErrRun, firstLine(fmt.Errorf("%v", recovered)))
		}
	}()
	output, err = m.executor.Call1(input)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrRun, firstLine(err))
	}
	return output, nil
}

// probabilityRows copies the output tensor into one slice per row, checking its
// shape and that every probability is a finite number.
func (m *Model) probabilityRows(output *tensors.Tensor, numRows int) ([][]float32, error) {
	flat, err := tensors.CopyFlatData[float32](output)
	if err != nil {
		return nil, fmt.Errorf("%w: reading the output: %s", ErrRun, firstLine(err))
	}
	if len(flat) != numRows*m.numClasses {
		return nil, fmt.Errorf("%w: output has %d values; want %d rows × %d classes", ErrRun, len(flat), numRows, m.numClasses)
	}
	rows := make([][]float32, numRows)
	for row := range rows {
		rows[row] = flat[row*m.numClasses : (row+1)*m.numClasses : (row+1)*m.numClasses]
		if column := firstNonFinite(rows[row]); column >= 0 {
			return nil, fmt.Errorf("%w: output row %d, class %d is %v", ErrNotFinite, row, column, rows[row][column])
		}
	}
	return rows, nil
}

// firstNonFinite returns the index of the first NaN or infinite value, or -1.
func firstNonFinite(values []float32) int {
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return i
		}
	}
	return -1
}

// firstLine keeps the first line of a GoMLX error; the rest is usually a stack
// trace that would bury the message for the person reading it.
func firstLine(err error) string {
	message, _, _ := strings.Cut(err.Error(), "\n")
	return message
}
