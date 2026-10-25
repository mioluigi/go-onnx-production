# Feature Spec 001 — Serve a Python-trained neural network from Go via ONNX and GoMLX

| Field | Value |
|---|---|
| Status | **DRAFT v1 — awaiting Gemini review and Luis's answers to section 11** |
| Owner | Luis Varela |
| Course deliverable | Week 5 Assignment: Go for ML/AI Production (100 pts) |
| Governing rules | `docs/constitution.md` · requirements: `docs/assignment-and-rubric.md` |
| Evidence so far | `docs/spikes/001-gomlx-onnx.md` |

> **Spec rule:** this document says **what** and **why**. **How** belongs in `plan.md`.

---

## 1. Problem (why)

Management fears that serving Python deep learning models to **thousands of users** will be slow and
costly. They ask whether data scientists can keep **developing in Python** while production **runs the
same model in Go**, using ONNX as the hand-off format and GoMLX as the Go runtime. The answer is only
useful if two things are proven: Go gives **the same predictions** as Python, and it is **faster or cheaper**
in the ways that matter for many users — per-request response time, bulk throughput, start-up, and memory.

## 2. Users and what they need

| User | Needs |
|---|---|
| **Management** | A clear recommendation: is Python-develop / Go-serve viable, at what cost, with what risks? |
| **Data scientists** | Keep their Python workflow; hand over one `.onnx` file; proof the Go version predicts identically. |
| **Production engineers** | A reusable Go package that loads an ONNX model and predicts, with known limits. |
| **Grader / any engineer** | Clone, build, run, test, and benchmark from the README alone. |

## 3. User stories

- **US-1 Equivalence.** As a data scientist, I want Go's predictions compared with Python's on the full test set, so I know the production model is the model I built.
- **US-2 Response time.** As a manager, I want the time to answer **one request** (latency) in Python and in Go, because thousands of interactive users experience latency, not bulk throughput.
- **US-3 Throughput and cost.** As a manager, I want bulk prediction time, start-up time, and memory for both languages, as a proxy for server cost.
- **US-4 Hand-off workflow.** As a data scientist, I want a documented, repeatable path: train in Python → export ONNX → verify in Go.
- **US-5 Usable program.** As any user, I want one executable that runs the bundled model on the bundled test set, or on my own CSV, and explains bad input.
- **US-6 Decision.** As a manager, I want a recommendation that includes the concerns engineers and data scientists would raise.

## 4. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | A Python program trains the model (framework: Q1; model and data: Q2), saves the **`.onnx` model**, the **test set**, and **Python's predictions** on it as versioned files. |
| FR-2 | The test set is **large**: at least 10,000 examples (Q2). |
| FR-3 | A reusable Go package loads the `.onnx` file with **GoMLX** (onnx-gomlx) and returns class **probabilities** and **predicted labels** for a batch of inputs. |
| FR-4 | **Parity check:** Go's predictions are compared with Python's on the **entire** test set: label agreement and largest probability difference, against the tolerance in Q5. Reported by a program, not by hand. |
| FR-5 | **Python timing:** bulk prediction on the full test set, and single-example latency (median and 95th percentile), from repeated runs after warm-up. |
| FR-6 | **Go timing:** the same two measurements on the same machine, for each GoMLX backend chosen in Q3. |
| FR-7 | **Start-up cost** is reported separately: loading the model and preparing it to run (GoMLX compiles a graph on first use; this matters for production start-up and autoscaling). |
| FR-8 | **Monte Carlo benchmark:** Go inference time and memory for batch sizes from 1 to 10,000, on synthetic random inputs with a fixed seed. |
| FR-9 | **Logging:** timestamped `log/slog` logs to standard error; `-verbose` shows detail. |
| FR-10 | **Whole-program metrics:** wall time and peak memory for the complete Python and Go programs (reusing the Week 4 measurement approach). |
| FR-11 | The program runs with **no arguments** on the bundled model and test set; optional flags select another model file or a CSV of inputs. |
| FR-12 | **Invalid input** produces a clear message and a non-zero exit code: missing file, not a valid ONNX model, a model without the expected input or `probabilities` output, CSV with the wrong number of features, non-numeric or non-finite values, empty input. |
| FR-13 | The README contains: purpose, roles of programs and data, build/run/test/benchmark instructions, parity and timing results, concerns, and the **recommendation to management**. |
| FR-14 | **AI dialogs** (Claude, Gemini) used in the project are included in the repository as plain text or PDF (Q7). |

## 5. Reference expectations

Unlike Week 4, there is no published answer to match: **Python's own predictions are the reference**
(FR-1, FR-4). The spike (`docs/spikes/001-gomlx-onnx.md`) shows what agreement to expect:

| Measure | Spike result (10 rows) | Required on full test set (Q5) |
|---|---|---|
| Predicted label agreement | 10 / 10 | 100 % |
| Largest absolute probability difference | 3.6 × 10⁻⁷ | ≤ 1 × 10⁻⁵ |

For **unit tests**, synthetic ONNX models with hand-chosen weights give exact known answers (for example,
a layer whose weights make the output equal a known transformation of the input).

## 6. Key findings that shape the plan ⚠️ (from Spike 001)

1. **GoMLX v0.28 requires Go ≥ 1.27.**
2. **Operation fusion breaks on scikit-learn exports** (2-D bias shape) and must be disabled there with `DisableFusion()`; PyTorch-style exports work with fusion on (spike addendum).
6. **Python ONNX tools can disagree:** the newest `onnx` writes a file version the newest ONNX Runtime rejects; exports must pin a compatible version.
3. **Classic-ML ONNX operations are unsupported** (e.g. `ArrayFeatureExtractor`); Go must request the
   neural network's `probabilities` output and compute the label itself.
4. **The library's README example is out of date;** pin versions and rely on tested code.
5. **Backends differ:** XLA (fast; macOS/Linux) vs. pure Go (portable; the only choice for Windows).

Each is also a point for the recommendation: the path works, but it needs engineering care.

## 7. Non-functional requirements

| ID | Requirement | Rubric |
|---|---|---|
| NFR-1 | Code follows Constitution Article 2. | Coding |
| NFR-2 | Test coverage of the reusable Go packages ≥ 90 %. | Testing |
| NFR-3 | CPU and memory profiles of Go inference generated and kept. | Testing |
| NFR-4 | Inference logic is a reusable package, independent of the program's input/output. | Design |
| NFR-5 | From `git clone` to results in ≤ 5 documented commands; Python retraining is optional, not required. | Documentation / Application |
| NFR-6 | Every file in the repository is under 50 MB (GitHub warns above that). | Documentation |
| NFR-7 | Dependency versions pinned in Go and Python. | Design |

## 8. Acceptance criteria (definition of done)

- [ ] AC-1 The parity check over the full test set meets the Q5 tolerance, and its output is saved in `results/`.
- [ ] AC-2 `go test ./...` passes, including synthetic known-answer models and invalid-input cases.
- [ ] AC-3 Python and Go timings (bulk, single-example latency, start-up) exist on the same machine for each chosen backend.
- [ ] AC-4 The Monte Carlo benchmark reports time and memory for batch sizes 1 to 10,000.
- [ ] AC-5 CPU and memory profiles are saved and summarized.
- [ ] AC-6 Whole-program wall time and peak memory are reported for Python and Go.
- [ ] AC-7 The executable runs with no arguments; each FR-12 case has a test and a clear message.
- [ ] AC-8 Build commands produce a macOS binary, a Windows `.exe`, and a macOS `.app` (Q8).
- [ ] AC-9 README contains every FR-13 item, with tables generated from `results/`.
- [ ] AC-10 AI dialog files are in the repository (FR-14).
- [ ] AC-11 Repository URL submitted in Canvas.

## 9. Rubric traceability

| Rubric criterion (20 pts each) | Satisfied by |
|---|---|
| Coding rules, organization, aesthetics | NFR-1, Constitution Art. 2 |
| Testing and software metrics | FR-4–FR-10, NFR-2, NFR-3, AC-1–AC-6 |
| Design and development | FR-1, FR-3, NFR-4, NFR-7 |
| Documentation | FR-13, FR-14, NFR-5, NFR-6, AC-9, AC-10 |
| Application | FR-11, FR-12, AC-7, AC-8 |

## 10. Out of scope

- Training in Go or fine-tuning (inference only).
- A network service (HTTP/gRPC). Latency per request is measured directly; a server is a possible next step named in the recommendation.
- GPU execution (CPU only, on Luis's Mac).

## 11. Open questions — **Luis decides before the plan step**

| # | Question | Recommendation |
|---|---|---|
| Q1 | Python framework: PyTorch, scikit-learn, or Keras? | **PyTorch.** Management asked about *deep learning*; PyTorch is the most common choice, and both of its export styles ran correctly in GoMLX in the spike addendum. scikit-learn is the fallback (already proven by the spike). Keras needs an extra converter (tf2onnx). |
| Q2 | Model and data? | **MNIST handwritten digits with a small multilayer network** (784 → 128 → 64 → 10): a real, widely known deep learning benchmark with a standard **10,000-image test set** (FR-2). A small convolutional network is a stretch goal once the core works. |
| Q3 | Which GoMLX backends to measure? | **Both:** XLA (fastest, macOS) and pure Go (portable, used for the Windows `.exe`). |
| Q4 | Also time **ONNX Runtime in Python**? | **Yes.** It separates "Go vs. Python" from "PyTorch vs. a lean runtime", which management needs to make a fair decision. |
| Q5 | Parity tolerance? | **100 % label agreement and ≤ 1 × 10⁻⁵ largest probability difference** (spike: 3.6 × 10⁻⁷). |
| Q6 | GitHub repository name? | e.g. `go-onnx-production` |
| Q7 | AI dialogs and the **free-versions** constraint: which tools' conversations go in the repo, and do the tools you used meet the "free versions" requirement? | Export this Claude conversation and the Gemini reviews to `docs/ai-dialogs/` as PDF. **You need to confirm** whether your Claude plan counts as a free version under the assignment's wording; if not, consider noting it in the README or asking the instructor. |
| Q8 | Executables? | Same as Week 4: Windows `.exe` (pure-Go backend) and macOS `.app` (thin Terminal wrapper). |
