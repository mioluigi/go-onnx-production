# Review 002 — Repository reviewed by Gemini (task T20)

| Field | Value |
|---|---|
| Artifact reviewed | The whole repository at commit `5b3aa56`, imported into Gemini from GitHub |
| Reviewer | Gemini (web), prompted as a strict grader and senior Go/ML engineer |
| Checked against | `docs/assignment-and-rubric.md`; README claims against `results/tables.md` |
| Date | 2026-10-25 |
| Full text | [`docs/ai-dialogs/gemini-t20-repository-review.pdf`](../ai-dialogs/gemini-t20-repository-review.pdf) |
| Outcome | 2 accepted, 2 accepted with changes, 1 rejected, 1 documented as a limitation by the author's decision |

| # | Finding (summarized) | Decision | Reason / change made |
|---|---|---|---|
| 1 | The README still contains the placeholder "TO CONFIRM BY THE AUTHOR BEFORE SUBMISSION". | **Accept** | Replaced with the author's statement of the AI service tiers used (Claude Pro, Gemini Ultra). |
| 2 | Prose says probabilities are within 1.4 × 10⁻⁶ of PyTorch's, but the table shows 1.49 × 10⁻⁶. | **Adjust** | The sentence referred to the two Go backends (1.37 × 10⁻⁶ and 7.75 × 10⁻⁷); 1.49 × 10⁻⁶ is ONNX Runtime's. The claim was correct, but because it could be misread, the sentence now names the Go backends and gives ONNX Runtime's figure separately. |
| 3 | The program's code (`cmd/predict-digits`) is excluded from coverage, because its tests run the built program in a separate process. | **Accept** | The end-to-end tests can now build the program with `-cover` and collect coverage from every run (`PREDICT_DIGITS_COVERDIR`, Go's `GOCOVERDIR` mechanism). Measured coverage: **91.6 %**. Command in the README's Testing section. |
| 4 | `newBackend` recovers a panic instead of validating the backend name. | **Adjust** | The panic comes from GoMLX (`compute.NewWithConfig` panics on an unknown name although it declares an error result). The name, or `$GOMLX_BACKEND`, is now checked against the compiled-in backends (`compute.List()`) first and rejected with a plain error listing the available ones. The recover stays as a backstop for panics inside a backend's constructor, so a program using the package cannot crash while loading a model. `TestLoadRejectsUnsuitableModels/unknown_backend` still passes. |
| 5 | The fallback from XLA to pure Go might leak resources from the failed attempt. | **Reject** | When `newBackend` fails, no backend exists to release, and `Load` closes the parsed ONNX model on every error from `prepare` (`inference.go`, `Load`). When the executor cannot be built, `prepare` finalizes the backend before returning. Nothing from the failed attempt outlives it. |
| 6 | The recommendation for "thousands of users" lacks a concurrency benchmark (many simultaneous requests; Go goroutines versus Python's global interpreter lock). | **Valid; decided by the author** | The README already states that concurrent load was not measured and recommends testing it before committing. **The author chose to document only:** no concurrency benchmark before submission; the README's recommendation names it as the next measurement (point 5c). |
