# Go for ML/AI Production: a PyTorch model served in Go through ONNX and GoMLX

Can a startup's data scientists **develop models in Python** and **run them in production with Go**? This
repository trains a handwritten-digit classifier in PyTorch, exports it to ONNX (Open Neural Network
Exchange), runs it in Go with [GoMLX](https://github.com/gomlx/gomlx), proves the Go predictions match
Python's on all 10,000 MNIST test images, compares speed and memory, and makes a recommendation to
management.

**Short answer:** the exported model gives **the same prediction for every one of the 10,000 test
images** in Go, with probabilities within about one millionth of PyTorch's. Go starts faster and uses less
memory than PyTorch, and ships as one self-contained executable. But Python's optimized runtimes
(PyTorch, ONNX Runtime) **predict faster** than GoMLX on this model. **Develop in Python, serve in Go** is
sound when the service is built in Go, with a parity check as a release gate; it is not a way to make
predictions faster. See [Recommendation](#recommendation-to-management).

**Contents:** [Recommendation](#recommendation-to-management) · [Results](#results) ·
[Concerns](#concerns-for-data-scientists-and-engineers) · [Quick start](#quick-start) ·
[Using the program](#using-the-program) · [Testing](#testing) ·
[Reproducing the measurements](#reproducing-the-measurements) · [Building executables](#building-the-executables) ·
[Repository map](#repository-map) · [How this was built](#how-this-project-was-built) · [References](#references)

---

## Recommendation to management

> **Yes: develop in Python and serve in Go, with an automated parity check before every release. But
> choose Go for its operational benefits, not for prediction speed.**

1. **Correctness is not a barrier.** The PyTorch model, exported to ONNX and run in Go, predicted the same
   digit as PyTorch for all 10,000 MNIST test images, with every probability within 1.4 × 10⁻⁶ of
   PyTorch's, on both GoMLX backends. Accuracy is identical (97.11 %). The check is automated in this
   repository and can gate every model release.
2. **Prediction speed favors Python's runtimes.** These are C++ engines tuned for exactly this work. For
   10,000 images in one call, GoMLX with its XLA backend took about 4 times as long as PyTorch, and its
   pure-Go backend about 13 times. For one image at a time, ONNX Runtime (about 10 µs) was about 5 times
   faster than Go (about 50 µs). Moving to Go with GoMLX does not, by itself, make predictions faster.
3. **Operations favor Go.** A complete Go run (start, load, predict 10,000 images, check) took about a
   tenth of a second, against about 1.5 seconds for PyTorch, most of which is importing PyTorch. That
   matters when servers scale up and down with demand. Go ships as one executable with the model inside,
   with no Python environment to install or patch, and it cross-compiles to Windows. Python with ONNX
   Runtime is competitive here too (about 0.2 s and the least memory), so the advantage over a lean Python
   deployment is smaller than over PyTorch.
4. **Capacity for "thousands of users".** At about 50 µs per single-image prediction, one thread could in
   principle serve on the order of 20,000 predictions per second for a model this size. Network handling
   and many simultaneous requests were **not** measured here, so this is an upper bound, not a forecast.
5. **Suggested path.**
   - (a) Keep model development in PyTorch; export to ONNX with pinned versions; make this repository's
     parity check (`parity` package) a required step before deployment.
   - (b) Where services are written in Go, serve with GoMLX and its XLA backend: compile the model at
     start-up (the first prediction includes compilation) and pre-install XLA's plugin in the server image.
   - (c) Before committing for latency-critical, high-traffic endpoints, benchmark ONNX Runtime's Go
     bindings and a concurrent-load test on the production servers (Linux). Neither was tested here;
     all measurements come from one laptop.

*Evidence for each point is in the sections below. The tables are generated from the files in
[`results/`](results/); figures quoted in prose are rounded from them.*

---

## Results

**The model.** A small neural network (multilayer perceptron): 784 pixel inputs → 128 → 64 → 10 digit
probabilities, with ReLU activations and a final softmax. Trained in PyTorch for 5 epochs on the 60,000
MNIST training images (Adam, learning rate 0.001, batch 128, seed 42; about 3 seconds on the Mac), then
exported to ONNX (IR version 10, opset 18). The exported graph uses only `Gemm`, `Relu` and `Softmax`.

**The test set.** The official MNIST test files, 10,000 images, unchanged (their MD5 checksums match the
published ones). Every runtime receives exactly the same inputs: each 28 × 28 image flattened row by row to
784 values, as 32-bit floats, divided by 255 (the *input contract*, applied identically in Python and Go
and tested in both).

**How the timings compare.** Both languages measure the same four things in the same way: loading the
model, the first prediction (one image), all 10,000 images in one call, and one image at a time. Python
is timed by [`python/bench_python.py`](python/bench_python.py), Go by the program itself.

<!-- tables:start -->
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
<!-- tables:end -->

**Reading the tables.**

- **Parity.** The Go backends differ from PyTorch by less than ONNX Runtime does, and all four runtimes
  agree on every digit.
- **Load model.** PyTorch's load time is mostly `import torch`. GoMLX's pure-Go backend has almost nothing
  to load; the XLA backend starts its plugin.
- **First prediction.** GoMLX compiles the model for each new batch size on its first call. With XLA this
  costs much more than a steady-state prediction, which is why a Go service should warm up at start-up.
- **10,000 images.** Python's runtimes lead. The CPU profile of the pure-Go backend
  ([`results/cpu-profile.txt`](results/cpu-profile.txt)) shows about three quarters of its time in a
  matrix multiplication that does not use the processor's vector (SIMD) instructions on this Mac.
- **Monte Carlo benchmark.** Per-image cost falls steeply as batches grow, for both backends; batching
  requests is the most effective speed-up available in Go.
- **Whole program.** Includes everything a user waits for. Go's peak memory includes the embedded test set
  decoded into memory.

---

## Concerns for data scientists and engineers

- **Not every model converts.** GoMLX's ONNX converter implements a subset of ONNX. In the preparatory
  experiment ([`docs/spikes/001-gomlx-onnx.md`](docs/spikes/001-gomlx-onnx.md)), a scikit-learn export used
  `ArrayFeatureExtractor`, which GoMLX does not implement, and a two-dimensional bias that needed GoMLX's
  layer fusion turned off (`inference.Options.DisableFusion`). Every new model needs the parity check.
- **Export settings matter.** In the spike, the newest `onnx` package wrote IR version 14, which the ONNX
  Runtime version installed there could not read; the export pins IR version 10, which both runtimes read.
- **Young libraries.** GoMLX and its ONNX converter are pre-1.0, so their APIs may still change, and they
  require Go 1.27. Pin versions (done in `go.mod` and `python/requirements.txt`).
- **XLA is not pure Go.** The XLA backend needs CGo, runs on macOS (Apple Silicon) and Linux only, and
  downloads its plugin from GitHub on first use. Production servers should have it pre-installed. The
  Windows executable therefore uses the slower pure-Go backend. If XLA cannot start, the program falls
  back to pure Go with a warning, unless `-backend xla` was requested explicitly.
- **Compilation on first use.** GoMLX compiles once per batch size. A service receiving many batch sizes
  pays that cost repeatedly unless it pads requests to a few fixed sizes.
- **Pre-processing must be duplicated exactly.** Any scaling done in Python before the model must be
  re-implemented in Go. Here it is one rule (pixel ÷ 255), written once in each language and tested in both.
- **Training and experimentation stay in Python.** Go runs the finished model only.

---

## Quick start

Requires **Go 1.27.2 or later** and Git. Python is needed only to retrain or to repeat the Python
measurements. Five commands, from clone to results:

```bash
git clone https://github.com/mioluigi/go-onnx-production.git
cd go-onnx-production
go test ./...                       # all tests, including parity on all 10,000 test images
go run ./cmd/predict-digits         # parity check, then timings, on the built-in model and test set
scripts/build.sh                    # executables in dist/ (macOS or Linux)
```

On macOS and Linux the first run with XLA downloads GoMLX's XLA plugin, once; later runs reuse it.

---

## Using the program

```text
predict-digits                        parity check and timings (XLA where available, else pure Go)
predict-digits -backend go            the same with the pure-Go backend
predict-digits -runs 0                parity check only, no timings
predict-digits -input digits.csv      predict the digit in each row of a CSV file
predict-digits -model other.onnx      use another ONNX model (784 inputs, 10 probabilities)
predict-digits -verbose               detailed (debug) logs
predict-digits -h                     help
```

**Input file (`-input`).** One image per line: the 784 pixels of a 28 × 28 image, row by row, as whole
numbers from 0 (background) to 255 (ink), separated by commas, with no header line. The output is CSV:
`row,digit,probability`. Mistakes are reported with the line and value, for example
`line 2, value 1 "256" is not a whole number from 0 to 255`.

**Output and logs.** Results go to standard output; timestamped logs (Go's `log/slog`) go to standard
error, so `predict-digits > results.txt` saves clean results. Each timing run ends with a single
`benchmark key=value …` line that `scripts/tables.sh` reads.

**Exit codes.** 0 success · 1 a data or model problem, including a parity FAIL (no timings are then
reported) · 2 a command-line mistake (usage is printed).

---

## Testing

```bash
go test ./...                                        # all tests
go test -cover ./idx/ ./parity/ ./inference/         # coverage of the reusable packages
go vet ./... && gofmt -l .                           # checks; gofmt prints nothing when tidy
go test -run='^$' -bench=Predict -benchmem ./inference/                  # Monte Carlo benchmark
GOMLX_BACKEND=xla go test -run='^$' -bench=Predict -benchmem ./inference/  # the same with XLA
```

| Package | What its tests check | Coverage |
|---|---|---:|
| `idx` | Reading MNIST's IDX format: official headers, gzip and plain files, truncated or damaged data, unsupported types, size limits | 96.7 % |
| `parity` | Comparing predictions: identical inputs, differences just inside and outside the tolerance, label disagreements, NaN and infinity | 100 % |
| `inference` | Small synthetic ONNX models with known answers (softmax, a model that always picks one class, a two-layer network checked against ONNX Runtime, a scikit-learn-style bias), invalid models, invalid inputs, NaN outputs, closing twice | 92.1 % |
| `cmd/predict-digits` | The built program, run as a user would: no arguments, parity on all 10,000 images, CSV input and each kind of input error, invalid models, a model that fails parity, command-line mistakes, help; plus the input contract and percentile calculation | end to end ¹ |
| `cmd/measure` | The whole-program measuring tool (from the Week 4 project) | 93.0 % |

¹ These tests run the compiled program in a separate process, which Go's coverage tool does not count.

**Tests were written before the code** they test, and important tests were checked by deliberately breaking
the code to confirm a test fails (for example: dividing pixels by 256 instead of 255, ignoring a parity
FAIL, skipping the row-length check). The synthetic test models are generated by
[`python/make_test_models.py`](python/make_test_models.py).

---

## Reproducing the measurements

All numbers above come from one run of `scripts/measure.sh` on an Apple M1 Max, plugged in, with other
programs closed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
python python/train_export.py        # optional: retrain and re-export (writes assets/)
scripts/measure.sh                   # every measurement, saved in results/ (about 5 minutes)
scripts/tables.sh > results/tables.md
scripts/readme.sh                    # copy results/tables.md into this README
```

Python 3.14.2 was used. If downloading MNIST fails with `CERTIFICATE_VERIFY_FAILED` on macOS, run
`pip install certifi` and `export SSL_CERT_FILE="$(python -m certifi)"` first.

| File in `results/` | Produced by |
|---|---|
| `python.txt` | `python/bench_python.py` (PyTorch and ONNX Runtime) |
| `go-xla.txt`, `go-go.txt` | `predict-digits -backend xla` / `-backend go` |
| `go-bench-xla.txt`, `go-bench-go.txt` | Monte Carlo benchmark, 5 runs per backend |
| `cpu-profile.txt`, `memory-profile.txt` | `go tool pprof -top` on 10,000-image predictions (pure-Go backend; XLA's work happens in C++, which Go's profiler cannot see) |
| `whole-program.txt` | `cmd/measure`: median wall time and peak memory of 5 complete runs of each program |
| `machine.txt`, `tables.md` | machine details; the tables in this README |

---

## Building the executables

```bash
scripts/build.sh
```

| File in `dist/` | What it is | How to run it |
|---|---|---|
| `predict-digits` | Command-line program for this computer, with XLA and pure Go | `dist/predict-digits` |
| `predict-digits.exe` | Windows program (64-bit), cross-compiled with `CGO_ENABLED=0 GOOS=windows GOARCH=amd64`; pure-Go backend only | In Command Prompt: `predict-digits.exe` |
| `PredictDigits.app` | macOS application for Apple Silicon (built on a Mac only) | Double-click in Finder, or `open dist/PredictDigits.app` |

The model and test set are embedded in each executable (Go's `embed`), so each runs with no other files.
**About the `.app`:** the program is a command-line tool, so the app is a thin wrapper that opens Terminal,
runs the program there, and waits for Return so the results stay on screen. Logs go to
`~/Library/Logs/PredictDigits.log`.

---

## Repository map

| Path | Role |
|---|---|
| [`cmd/predict-digits/`](cmd/predict-digits/) | The program: parity check, timings, or predictions for a CSV file. `backend_xla.go` adds XLA where CGo is available. |
| [`inference/`](inference/) | Reusable package: loads any one-input ONNX classifier and predicts with GoMLX. No file or console I/O. |
| [`parity/`](parity/) | Reusable package: compares two sets of predictions (labels and probabilities). |
| [`idx/`](idx/) | Reusable package: reads the IDX format MNIST is published in. |
| [`assets/`](assets/) | The trained model (`.onnx`, and PyTorch weights `.pt`), the official MNIST test files, PyTorch's predictions; embedded by `assets.go`. |
| [`cmd/measure/`](cmd/measure/) | Whole-program time and peak-memory tool (macOS, Linux). |
| [`python/`](python/) | Training and export (`train_export.py`, `mnist_model.py`), shared helpers (`mnist_io.py`), Python timings (`bench_python.py`), synthetic test models (`make_test_models.py`), pinned `requirements.txt`. |
| [`scripts/`](scripts/) | Measurement (`measure.sh`, `tables.sh`, `readme.sh`) and builds (`build.sh`, `build-macos-app.sh`). |
| [`results/`](results/) | Every measured output used in this README. |
| [`docs/`](docs/), [`specs/`](specs/) | How the project was specified, planned, reviewed and decided (next section). |

---

## How this project was built

The project followed **spec-driven development**: requirements and design were written and reviewed
before code, and every result is backed by a file in the repository.

1. **Constitution** ([`docs/constitution.md`](docs/constitution.md)): rules derived from the assignment's
   grading rubric ([`docs/assignment-and-rubric.md`](docs/assignment-and-rubric.md)).
2. **Specification** ([`spec.md`](specs/001-onnx-go-inference/spec.md)): what and why, acceptance criteria,
   and the decisions made with the author.
3. **Spike** ([`docs/spikes/001-gomlx-onnx.md`](docs/spikes/001-gomlx-onnx.md)): a throwaway experiment
   proving the Python → ONNX → GoMLX path before planning.
4. **Plan** ([`plan.md`](specs/001-onnx-go-inference/plan.md)) and **tasks**
   ([`tasks.md`](specs/001-onnx-go-inference/tasks.md)): design with reasons, then small test-first steps,
   one commit each. The riskiest step (the real export, checked in Go) came first.
5. **Reviews and decisions** ([`docs/reviews/`](docs/reviews/), [`docs/decisions.md`](docs/decisions.md)):
   every review finding accepted or rejected, and every later change, with its reason.

**AI assistance, disclosed.** Claude (Anthropic, used through Claude Code) drafted the specification,
plan, code and tests under the author's direction; Gemini (Google, on the web) reviewed the specification
and the repository independently; the author approved every document, ran the training and all
measurements on his own Mac, and reviewed every change. The dialogs are in
[`docs/ai-dialogs/`](docs/ai-dialogs/). **Service tiers:** ⚠️ TO CONFIRM BY THE AUTHOR BEFORE SUBMISSION.

---

## References

- Bodner, Jon. 2024. *Learning Go: An Idiomatic Approach to Real-World Go Programming*, 2nd ed. O'Reilly.
- GoMLX authors. GoMLX v0.28.17 and onnx-gomlx v0.5.14. https://github.com/gomlx/gomlx, https://github.com/gomlx/onnx-gomlx
- LeCun, Yann, Corinna Cortes, and Christopher J. C. Burges. "The MNIST Database of Handwritten Digits."
- McConnell, Steve. 2004. *Code Complete*, 2nd ed. Microsoft Press.
- ONNX Runtime. https://onnxruntime.ai
- PyTorch. "Export a PyTorch model to ONNX" (`torch.onnx.export` with `dynamo=True`). https://pytorch.org/docs/stable/onnx.html
- Vivien, Vladimir. 2025. "Go in the AI/ML Landscape: A Practical Guide." *Medium*.
