# CLAUDE.md — project context

Go for ML/AI Production (Week 5): train an MNIST classifier in PyTorch, export it to ONNX, run it in Go with
GoMLX, prove the predictions match Python's on all 10,000 test images, compare speed and memory, and recommend
to management whether to develop in Python and serve in Go.

## Read these first (in order)

1. `docs/constitution.md` — non-negotiable rules (rubric-derived)
2. `specs/001-onnx-go-inference/spec.md` — what and why (APPROVED)
3. `specs/001-onnx-go-inference/plan.md` — how (APPROVED)
4. `specs/001-onnx-go-inference/tasks.md` — ordered checklist; work the next unchecked task
5. `docs/spikes/001-gomlx-onnx.md` — what was proven before planning

## Working rules

- One task per commit; message starts with the task ID (`T05: …`).
- Test first: write the failing test, run it, then implement; check important tests by breaking the code.
- If implementation shows the spec or plan is wrong, update the document first, then the code.
- Record changes after approval in `docs/decisions.md`; review findings in `docs/reviews/`.
- Never hand-type results into the README; generate them from `results/`.
- Performance numbers come only from Luis's Mac, and only after the parity check passes.

## Key commands

```bash
go test ./...                                          # all tests
go test -cover ./idx/ ./parity/ ./inference/           # coverage (≥ 90 %)
go vet ./... && gofmt -l .                             # lint (gofmt -l prints nothing)
go run ./cmd/predict-digits                            # parity check + timings, built-in model and data
go run ./cmd/predict-digits -backend xla               # same with the XLA backend (macOS/Linux)
python python/train_export.py                          # retrain and re-export (writes assets/)
```

## Layout

`idx/` IDX file reader · `inference/` ONNX model → GoMLX predictions · `parity/` compares predictions ·
`assets/` model, MNIST test set, PyTorch predictions (embedded) · `cmd/predict-digits/` the program ·
`cmd/measure/` whole-program time and memory · `python/` training, export, Python timings ·
`scripts/` measurement, tables, builds · `results/` measured outputs · `docs/`, `specs/` the process record.
