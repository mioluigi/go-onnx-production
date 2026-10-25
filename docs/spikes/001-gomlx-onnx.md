# Spike 001 — Can GoMLX load a Python-exported ONNX model and match its predictions?

| Field | Value |
|---|---|
| Date | 2026-10-24 |
| Question | The riskiest assumption in the assignment: does the Python → ONNX → GoMLX path work, and do predictions match? |
| Answer | **Yes**, with three adjustments (below). |
| Status | Throwaway experiment. Code is not part of the deliverable; findings feed the spec. |

## Setup

- **Python:** scikit-learn 1.9.1 `MLPClassifier` (two hidden layers, 64 and 32 units, ReLU) on a synthetic
  3-class problem (20 features); exported with skl2onnx 1.20.0, ONNX opset 17, `zipmap` disabled.
- **Go:** Go 1.27.2, GoMLX v0.28.17, onnx-gomlx v0.5.14, pure-Go backend (`GOMLX_BACKEND=go`), Linux x86-64.
- PyTorch was not tested here because its download site is blocked from the spike environment; it is
  expected to export the same core operations (MatMul/Gemm, Add, Relu, Softmax).

## Results

| Check | Result |
|---|---|
| Parse the `.onnx` file | ✅ inputs `X` (batch × 20), outputs `label`, `probabilities` (batch × 3) |
| Load weights into GoMLX | ✅ |
| Predicted labels, 10 test rows | ✅ identical to Python: `[2 2 0 0 0 1 1 2 0 2]` |
| Largest probability difference | ✅ **3.6 × 10⁻⁷** (float32 rounding) |

## Findings that shape the spec

1. **GoMLX v0.28 requires Go ≥ 1.27.** (Luis's Mac has Go 1.27.2.)
2. **Fusion must be disabled.** GoMLX fuses `MatMul + Add` into one dense operation, but skl2onnx stores
   biases with shape `[1, 64]` rather than `[64]`, and fusion fails with
   `FusedDense: bias shape (Float32)[1, 64] does not match output features dimensions [64]`.
   Calling `model.DisableFusion()` before building the graph fixes it.
3. **Classic-ML ONNX operations are not implemented.** Requesting the `label` output fails with
   `unimplemented ONNX op "ArrayFeatureExtractor"`. The neural network itself runs, so Go requests only
   `probabilities` and takes the most probable class (arg max), which is what `label` computes.
4. **Documentation lags the code.** The onnx-gomlx README example (`onnx.ReadFile`, `backends.New()`) no
   longer matches v0.5.14; the working API is `parser.ParseFile` → `VariablesToScope` → `CallGraph`, with
   backends from `github.com/gomlx/compute`. Pin versions.
5. **Two backends matter.** XLA (fast; macOS arm64 and Linux, CPU) and the pure-Go backend (portable,
   no C dependencies — the only option for a Windows `.exe`).

## Working call sequence (for the plan)

```go
onnxModel, err := parser.ParseFile(path)   // github.com/gomlx/onnx-gomlx/onnx/parser
onnxModel.DisableFusion()                  // finding 2
store := model.NewStore()                  // github.com/gomlx/gomlx/ml/model
err = onnxModel.VariablesToScope(store.RootScope())
backend, err := compute.New()              // github.com/gomlx/compute; GOMLX_BACKEND selects xla or go
outputs := model.MustCallOnceN(backend, store, func(scope *model.Scope, inputs []*Node) []*Node {
	return onnxModel.CallGraph(scope, inputs[0].Graph(), map[string]*Node{"X": inputs[0]}, "probabilities")
}, batch)
```
