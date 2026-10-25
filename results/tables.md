Measured on Darwin arm64 · Apple M1 Max · 64 GB · macOS 27.0.1 · Go go1.27.2 · Python 3.14.2 · 2026-10-25 16:04 EDT

### Do the predictions match? (all 10,000 MNIST test images)

| Language | Runtime | Accuracy (%) | Same digit as PyTorch | Largest probability difference from PyTorch |
|---|---|---:|---:|---:|
| Python | PyTorch 2.14.1 | 97.11 | 10000/10000 | reference |
| Python | ONNX Runtime 1.31.0 | 97.11 | 10000/10000 | 1.49e-06 |
| Go | GoMLX v0.28.17, XLA backend | 97.11 | 10000/10000 | 7.75e-07 |
| Go | GoMLX v0.28.17, pure-Go backend | 97.11 | 10000/10000 | 1.37e-06 |

Parity requires the same digit for every image and probabilities within 1 × 10⁻⁵ of PyTorch's (spec Q5).

### Prediction speed

| Language | Runtime | Load model (ms) | First prediction (ms) | 10,000 images in one call (ms) | Relative to PyTorch | One image, median (µs) | One image, 95th percentile (µs) | Relative to PyTorch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Python | PyTorch 2.14.1 | 988.90 | 3.97 | 3.71 | 1.00× | 27.7 | 29.6 | 1.00× |
| Python | ONNX Runtime 1.31.0 | 71.62 | 0.68 | 5.17 | 1.39× | 10.2 | 10.4 | 0.37× |
| Go | GoMLX v0.28.17, XLA backend | 77.71 | 177.44 | 15.31 | 4.13× | 52.2 | 70.5 | 1.88× |
| Go | GoMLX v0.28.17, pure-Go backend | 0.53 | 2.67 | 47.22 | 12.73× | 46.6 | 50.0 | 1.68× |

10,000 images: median of 20 calls. One image: 1000 calls after 20 warm-up calls. Relative to PyTorch: time ÷ PyTorch's time, so below 1× is faster. 
The first prediction includes compiling the model for that batch size (GoMLX) or preparing it (Python). 
1 ms = 1,000 µs.

### Monte Carlo benchmark (Go): random inputs, increasing batch size

| Batch size | XLA: ms per batch | XLA: µs per image | XLA: MB allocated | Pure Go: ms per batch | Pure Go: µs per image | Pure Go: MB allocated |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.039 | 38.95 | 0.01 | 0.046 | 46.45 | 0.01 |
| 10 | 0.114 | 11.40 | 0.06 | 0.247 | 24.71 | 0.10 |
| 100 | 0.286 | 2.86 | 0.62 | 0.748 | 7.48 | 1.12 |
| 1000 | 1.878 | 1.88 | 6.05 | 5.028 | 5.03 | 9.90 |
| 10000 | 14.909 | 1.49 | 60.45 | 46.151 | 4.62 | 108.59 |

Median of 5 `go test -bench` runs per backend; ns/op ÷ 1,000,000 = ms; ÷ batch size ÷ 1,000 = µs per image; B/op ÷ 1,048,576 = MB. MB allocated counts Go memory only; XLA's own C++ memory is not included.

### Whole program, start-up included: load, predict all 10,000 images once, check parity

| Program | Median wall time (ms) | Peak memory, max RSS (MB) |
|---|---:|---:|
| Go, GoMLX XLA backend (`predict-digits -runs 0`) | 158.3 | 267.4 |
| Go, GoMLX pure-Go backend (`predict-digits -runs 0`) | 112.5 | 154.0 |
| Python, PyTorch (`bench_python.py --predict-only`) | 1473.9 | 252.9 |
| Python, ONNX Runtime (`bench_python.py --predict-only`) | 185.1 | 102.0 |

Median of 5 runs measured by `cmd/measure`: wall time from Go's monotonic clock, peak memory from the operating system's rusage report. Includes starting the program (and for Python, importing its libraries). The Go timings exclude XLA's one-time plugin download.
