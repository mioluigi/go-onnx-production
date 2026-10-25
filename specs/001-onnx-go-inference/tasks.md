# Tasks 001 — Ordered checklist to implement Plan 001

| Field | Value |
|---|---|
| Status | **APPROVED — 2026-10-25** |
| Implements | `plan.md` → `spec.md` (APPROVED v2) |

**How to work this list:** one task = one commit (message starts with the task ID). 🔴 tasks write tests that
must fail first; 🟢 tasks make them pass. 🤖 Claude writes and pushes · 👤 Luis runs on the Mac (anything
involving PyTorch, XLA, or official timings). After every 🤖 task: `git pull`, read the diff, run "Verify".

Ordering puts the **riskiest step first** (T03, the real PyTorch export checked in Go) so a surprise is found
early, while the Go packages are built in parallel on synthetic models.

---

## Phase 0 — Setup
- [ ] **T00 👤 Python environment on the Mac.** `.venv` with torch, torchvision, onnx, onnxruntime, onnxscript; send versions.
  Done when: versions known for `requirements.txt`.
- [x] **T01 🤖 Scaffolding.** `CLAUDE.md`, `.gitignore` (`.venv/`, `dist/`, `*.out`, MNIST download cache), `docs/decisions.md`, `go.mod`. *(GoMLX version pins are added in T08 with the first code that imports them, then `go mod tidy` on the Mac; `python/requirements.txt` is written in T03 from the Mac's installed versions.)*
  Verify: `go mod verify`.

## Phase 1 — The model (riskiest first)
- [x] **T02 🤖 `python/train_export.py`.** Train the 784 → 128 → 64 → 10 network (fixed seed), export ONNX (`input` → `probabilities`, dynamic batch, compatible IR version), copy the official test files, write PyTorch's test-set probabilities in IDX format, verify the export with ONNX Runtime before saving.
- [ ] **T03 👤 Run it on the Mac; commit `assets/`.** Then run Claude's quick check (`go run ./internal/checkexport`, a throwaway parity check) — **the go/no-go gate** for the rest of the plan.
  Done when: test accuracy ~97–98 %; Go and PyTorch agree on all 10,000 images.

## Phase 2 — Reusable packages (built on synthetic data; can overlap with T02–T03)
- [x] **T04 🤖 `python/make_test_models.py` + `inference/testdata/`.** Tiny ONNX models with known answers.
- [x] **T05 🔴🟢 `idx` package.** Tests, then implementation. Verify: `go test -cover ./idx/` ≥ 90 %.
- [x] **T06 🔴🟢 `parity` package.** Tests (incl. NaN), then implementation. Verify: ≥ 90 %.
- [x] **T07 🔴 `inference` tests.** Known answers, fusion on/off, invalid model and input cases, `Labels`.
- [x] **T08 🟢 `inference` implementation.** Verify: `go test -cover ./inference/` ≥ 90 %, `go vet` clean.
- [x] **T09 🤖 Monte Carlo benchmarks** (`BenchmarkPredict/batch=1…10000`).

## Phase 3 — The program
- [x] **T10 🔴 End-to-end tests** for `cmd/predict-digits` (no args → PASS; CSV errors; flags; stderr logs).
- [x] **T11 🟢 `cmd/predict-digits`** with embedded assets, backend selection, timings, CSV mode.
- [ ] **T12 👤 XLA on the Mac.** `go run ./cmd/predict-digits -backend xla` (first run downloads the XLA plugin); confirm PASS.

## Phase 4 — Python timing and measurement
- [x] **T13 🤖 `python/bench_python.py`** (PyTorch and ONNX Runtime; bulk, latency, load, first prediction; `--predict-only`).
- [ ] **T14 🤖 `cmd/measure` (from Week 4), `scripts/measure.sh`, `scripts/tables.sh`.**
- [ ] **T15 👤 Run all measurements on the Mac** (Python, Go × 2 backends, Monte Carlo, profiles, whole program); commit `results/`.

## Phase 5 — Executables
- [ ] **T16 🤖 `scripts/build.sh`, `scripts/build-macos-app.sh`** (`.exe` pure Go; `.app` Apple Silicon).
- [ ] **T17 👤 Build all three; open the `.app`.**

## Phase 6 — Documentation and submission
- [ ] **T18 👤 AI dialogs.** Export this Claude conversation and the Gemini reviews to `docs/ai-dialogs/` (PDF).
- [ ] **T19 🤖 README** (tables by script; concerns; recommendation; AI tools statement for Luis to confirm).
- [ ] **T20 👤 Gemini review of the repository** against the rubric; triage into `docs/reviews/`.
- [ ] **T21 👤🤖 Final acceptance** (fresh clone, every AC) **and submit** the URL in Canvas.

**Stretch (only if time remains):** small CNN model; concurrent-request throughput.

## Traceability

| Acceptance criterion | Tasks |
|---|---|
| AC-1 parity on full test set | T03, T10, T11 |
| AC-2 tests pass | T05–T08, T10 |
| AC-3 timings both languages, both backends | T11–T13, T15 |
| AC-4 Monte Carlo | T09, T15 |
| AC-5 profiles | T15 |
| AC-6 whole program | T14, T15 |
| AC-7 runs with no arguments; input errors | T10, T11 |
| AC-8 executables | T16, T17 |
| AC-9 README | T19 |
| AC-10 AI dialogs | T18 |
| AC-11 submitted | T21 |
