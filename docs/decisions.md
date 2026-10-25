# Decision log

Decisions made **after** a document was approved, newest first: date, decision, reason, documents updated.

| Date | Decision | Reason | Updated |
|---|---|---|---|
| 2026-10-25 | Claude's workspace builds Go code with a git-ignored module file (`go.local.mod`) that maps blocked import hosts (`golang.org/x`, `google.golang.org`, `gonum.org`, `k8s.io`) to their GitHub mirrors. The committed `go.mod` never contains these replacements; `go.sum` is produced by `go mod tidy` on Luis's Mac. | Those hosts are blocked from Claude's workspace only. Committing the replacements would make the repository depend on Claude's environment. | tasks.md T01 |
| 2026-10-25 | Gemini reviews the plan **in parallel** with the low-risk tasks T01, T02, T04 instead of before them; any accepted finding is still applied before the affected task. | Assignment due the same day. These tasks do not depend on plan details likely to change. | — |
