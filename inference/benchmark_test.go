package inference

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
)

// benchmarkModelPath is the trained MNIST model written by python/train_export.py.
const benchmarkModelPath = "../assets/mnist_mlp.onnx"

// benchmarkPoolSize is how many random input rows the benchmarks draw batches from.
const benchmarkPoolSize = 10_000

// BenchmarkPredict is the Monte Carlo benchmark (spec FR-8): for each batch size,
// every iteration predicts a batch drawn at a random position from a pool of
// random inputs (pixel values in [0, 1), fixed seed, so runs are repeatable).
// The backend is GoMLX's default; choose one with GOMLX_BACKEND=go or xla.
//
//	go test -run='^$' -bench=Predict -benchmem ./inference/
func BenchmarkPredict(b *testing.B) {
	onnxBytes, err := os.ReadFile(benchmarkModelPath)
	if err != nil {
		b.Skipf("needs the trained model (run python/train_export.py): %v", err)
	}
	m, err := Load(onnxBytes, Options{})
	if err != nil {
		b.Fatalf("Load: %v", err)
	}
	defer m.Close()

	rng := rand.New(rand.NewPCG(431, 5))
	pool := randomRows(rng, benchmarkPoolSize, m.NumFeatures())
	for _, batchSize := range []int{1, 10, 100, 1_000, 10_000} {
		b.Run(fmt.Sprintf("batch=%d", batchSize), func(b *testing.B) {
			// The first call compiles the graph for this batch size; keep it out of the timing.
			if _, err := m.Predict(pool[:batchSize]); err != nil {
				b.Fatalf("Predict: %v", err)
			}
			b.ReportAllocs()
			for b.Loop() {
				start := rng.IntN(len(pool) - batchSize + 1)
				if _, err := m.Predict(pool[start : start+batchSize]); err != nil {
					b.Fatalf("Predict: %v", err)
				}
			}
			images := float64(b.N) * float64(batchSize)
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/images, "ns/image")
			b.ReportMetric(images/b.Elapsed().Seconds(), "images/s")
		})
	}
}

// randomRows returns rows × features values drawn uniformly from [0, 1), the
// range of MNIST inputs after the input contract divides pixels by 255.
func randomRows(rng *rand.Rand, rows, features int) [][]float32 {
	result := make([][]float32, rows)
	for i := range result {
		result[i] = make([]float32, features)
		for j := range result[i] {
			result[i][j] = rng.Float32()
		}
	}
	return result
}
