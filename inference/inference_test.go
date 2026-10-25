package inference

import (
	"errors"
	"math"
	"os"
	"testing"

	"github.com/mioluigi/go-onnx-production/idx"
	"github.com/mioluigi/go-onnx-production/parity"
)

// exactTolerance is for answers computed exactly in the test; float32 arithmetic
// in the model leaves differences around 1e-7.
const exactTolerance = 1e-6

// loadTestModel loads a model from testdata/ with the given options, failing the test on error.
func loadTestModel(t *testing.T, fileName string, opts Options) *Model {
	t.Helper()
	m, err := Load(readTestFile(t, fileName), opts)
	if err != nil {
		t.Fatalf("Load(%s): %v", fileName, err)
	}
	t.Cleanup(m.Close)
	return m
}

func readTestFile(t *testing.T, fileName string) []byte {
	t.Helper()
	contents, err := os.ReadFile("testdata/" + fileName)
	if err != nil {
		t.Fatal(err)
	}
	return contents
}

func readTestMatrix(t *testing.T, fileName string) [][]float32 {
	t.Helper()
	file, err := os.Open("testdata/" + fileName)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	array, err := idx.Read(file)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := array.Rows()
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

// softmax is the reference answer for softmax_3.onnx, computed independently of the model.
func softmax(values []float32) []float32 {
	var total float64
	exponentials := make([]float64, len(values))
	for i, value := range values {
		exponentials[i] = math.Exp(float64(value))
		total += exponentials[i]
	}
	result := make([]float32, len(values))
	for i := range values {
		result[i] = float32(exponentials[i] / total)
	}
	return result
}

func assertClose(t *testing.T, got, want [][]float32, tolerance float64) {
	t.Helper()
	report, err := parity.Compare(got, want, tolerance)
	if err != nil {
		t.Fatalf("comparing outputs: %v", err)
	}
	if !report.Pass() {
		t.Errorf("outputs differ: %s\ngot  %v\nwant %v", report, got, want)
	}
}

func TestLoadDescribesTheModel(t *testing.T) {
	m := loadTestModel(t, "two_layer.onnx", Options{})
	if m.NumFeatures() != 8 || m.NumClasses() != 4 {
		t.Errorf("NumFeatures, NumClasses = %d, %d; want 8, 4", m.NumFeatures(), m.NumClasses())
	}
}

func TestPredictKnownAnswers(t *testing.T) {
	t.Run("softmax of the input itself", func(t *testing.T) {
		inputs := [][]float32{{1, 2, 3}, {0, 0, 0}, {-1, 5, 0.5}}
		got, err := loadTestModel(t, "softmax_3.onnx", Options{}).Predict(inputs)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		want := [][]float32{softmax(inputs[0]), softmax(inputs[1]), softmax(inputs[2])}
		assertClose(t, got, want, exactTolerance)
	})

	t.Run("a model that always prefers class 2", func(t *testing.T) {
		inputs := [][]float32{{9, -9, 0, 1}, {0, 0, 0, 0}, {-3, 7, 2, 2}}
		got, err := loadTestModel(t, "prefers_class_2.onnx", Options{}).Predict(inputs)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		for row, label := range parity.Labels(got) {
			if label != 2 {
				t.Errorf("row %d: label %d, want 2 (probabilities %v)", row, label, got[row])
			}
		}
	})

	t.Run("two-layer network matches ONNX Runtime", func(t *testing.T) {
		got, err := loadTestModel(t, "two_layer.onnx", Options{}).Predict(readTestMatrix(t, "two_layer_inputs-idx2-float.gz"))
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		assertClose(t, got, readTestMatrix(t, "two_layer_expected-idx2-float.gz"), exactTolerance)
	})

	t.Run("output with a custom name", func(t *testing.T) {
		got, err := loadTestModel(t, "named_logits.onnx", Options{OutputName: "logits"}).Predict([][]float32{{1, 2, 3}})
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		assertClose(t, got, [][]float32{softmax([]float32{1, 2, 3})}, exactTolerance)
	})
}

// TestPredictScikitLearnStyleBias covers spike finding 2: GoMLX's fusion fails on
// a 2-D bias, and must be turned off for such models.
func TestPredictScikitLearnStyleBias(t *testing.T) {
	inputs := readTestMatrix(t, "two_layer_inputs-idx2-float.gz")
	want := readTestMatrix(t, "two_layer_expected-idx2-float.gz")

	withFusion := loadTestModel(t, "sklearn_bias.onnx", Options{})
	if _, err := withFusion.Predict(inputs); !errors.Is(err, ErrRun) {
		t.Errorf("with fusion: error = %v, want errors.Is(ErrRun)", err)
	}

	got, err := loadTestModel(t, "sklearn_bias.onnx", Options{DisableFusion: true}).Predict(inputs)
	if err != nil {
		t.Fatalf("without fusion: Predict: %v", err)
	}
	assertClose(t, got, want, exactTolerance)
}

// TestPredictReusesTheCompiledModel checks that repeated predictions, including
// different batch sizes, keep working and give the same answers.
func TestPredictReusesTheCompiledModel(t *testing.T) {
	m := loadTestModel(t, "softmax_3.onnx", Options{})
	for _, inputs := range [][][]float32{{{1, 2, 3}}, {{1, 2, 3}, {3, 2, 1}}, {{1, 2, 3}}} {
		got, err := m.Predict(inputs)
		if err != nil {
			t.Fatalf("Predict(%d rows): %v", len(inputs), err)
		}
		assertClose(t, got[:1], [][]float32{softmax(inputs[0])}, exactTolerance)
	}
}

func TestLoadRejectsUnsuitableModels(t *testing.T) {
	tests := []struct {
		name     string
		contents []byte
		opts     Options
		wantErr  error
	}{
		{"not an ONNX file", []byte("this is not a model"), Options{}, ErrNotONNX},
		{"empty file", nil, Options{}, ErrNotONNX},
		{"no output named probabilities", readTestFile(t, "named_logits.onnx"), Options{}, ErrMissingOutput},
		{"two inputs", readTestFile(t, "two_inputs.onnx"), Options{}, ErrInputShape},
		{"unknown backend", readTestFile(t, "softmax_3.onnx"), Options{Backend: "no-such-backend"}, ErrBackend},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := Load(tc.contents, tc.opts)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Load error = %v, want errors.Is(%v)", err, tc.wantErr)
			}
			if m != nil {
				m.Close()
				t.Error("Load returned a model together with an error")
			}
		})
	}
}

func TestPredictRejectsBadInput(t *testing.T) {
	m := loadTestModel(t, "softmax_3.onnx", Options{})
	tests := []struct {
		name    string
		inputs  [][]float32
		wantErr error
	}{
		{"no rows", nil, ErrEmptyBatch},
		{"too few features", [][]float32{{1, 2}}, ErrInputShape},
		{"rows of different lengths", [][]float32{{1, 2, 3}, {1, 2, 3, 4}}, ErrInputShape},
		{"NaN", [][]float32{{1, float32(math.NaN()), 3}}, ErrNotFinite},
		{"infinity", [][]float32{{float32(math.Inf(-1)), 2, 3}}, ErrNotFinite},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := m.Predict(tc.inputs); !errors.Is(err, tc.wantErr) {
				t.Errorf("Predict error = %v, want errors.Is(%v)", err, tc.wantErr)
			}
		})
	}
}

// TestPredictReportsProblemsAsErrors covers failures that happen inside GoMLX:
// they must come back as errors, never as a crash.
func TestPredictReportsProblemsAsErrors(t *testing.T) {
	t.Run("operation GoMLX does not implement", func(t *testing.T) {
		m := loadTestModel(t, "unsupported_op.onnx", Options{})
		if _, err := m.Predict([][]float32{{1, 2, 3}}); !errors.Is(err, ErrRun) {
			t.Errorf("Predict error = %v, want errors.Is(ErrRun)", err)
		}
	})
	t.Run("output contains NaN", func(t *testing.T) {
		m := loadTestModel(t, "sqrt_output.onnx", Options{OutputName: "probabilities"})
		if _, err := m.Predict([][]float32{{4, -1}}); !errors.Is(err, ErrNotFinite) {
			t.Errorf("Predict error = %v, want errors.Is(ErrNotFinite)", err)
		}
	})
}

func TestCloseTwiceIsSafe(t *testing.T) {
	m, err := Load(readTestFile(t, "softmax_3.onnx"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	m.Close()
	m.Close()
	if _, err := m.Predict([][]float32{{1, 2, 3}}); !errors.Is(err, ErrClosed) {
		t.Errorf("Predict after Close: error = %v, want errors.Is(ErrClosed)", err)
	}
}
