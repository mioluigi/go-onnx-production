# Project Constitution — Go for ML/AI Production (ONNX → GoMLX)

> **Purpose of this file.** These are the non-negotiable rules for the project. Every spec, plan, task,
> and line of code is checked against them. If a later decision conflicts with this file, either the
> decision changes or this file is amended deliberately, with a note in `docs/decisions.md`.
>
> **Source of authority.** Derived from the Week 5 assignment and its **Programming Assignment Rubric**
> (`docs/assignment-and-rubric.md`), and carried forward from the Week 4 project where the lessons
> still apply. Each rule is tagged with the rubric criterion it protects: **[C]** Coding,
> **[T]** Testing & metrics, **[D]** Design, **[Doc]** Documentation, **[A]** Application.

---

## Article 1 — Correctness before speed

1.1 Go's predictions are only trusted when they **match the Python model's predictions on the same test
set**, within a stated numeric tolerance, with the agreement computed by a program. **[T]**
1.2 Speed is only reported for a Go model that has passed 1.1 on the full test set. A fast wrong answer
is not a result. **[T]**
1.3 Every numeric claim in the README is reproducible by a command a grader can run, and is copied from
program output — never hand-typed. **[Doc]**

## Article 2 — Idiomatic, readable Go (McConnell 2004, ch. 32) **[C]**

2.1 Go modules, the Go toolchain the dependencies require (GoMLX needs Go ≥ 1.27), `gofmt` and `go vet` clean.
2.2 Meaningful, specific names; not abstract (`data`, `doIt`), not overly long.
2.3 No useless temporary variables or intermediate results.
2.4 Functions do **one thing**; large blocks are split into small, named functions.
2.5 Plain `if/else` and `for`; in complex Boolean expressions, **distribute the NOT**.
2.6 Self-documenting code: every exported identifier has a doc comment; comments explain *why*.
2.7 Errors are returned and handled, never ignored; library panics are recovered at the boundary and
turned into errors with a plain-English explanation.

## Article 3 — Test-first, measured **[T]**

3.1 Write the failing test before the implementation of every critical component, and confirm each
important test can fail by deliberately breaking the code (mutation check).
3.2 Standard library `testing`; table-driven tests; edge cases; **synthetic test data with known answers**
(for example, a tiny ONNX model whose weights make the correct output obvious).
3.3 Floating-point comparisons use a named tolerance constant, and **reject NaN explicitly**.
3.4 Benchmarks use `testing.B` with allocation reporting; CPU and memory profiles are generated and kept.
3.5 Python and Go time **the same unit of work** on **the same machine**, with printing outside timed code,
and report medians of repeated runs.

## Article 4 — Clean, reusable design **[D]**

4.1 Model loading and inference live in a **reusable Go package** with no printing, file-format
assumptions about the data, or `os.Exit`. Input/output belongs in `main`.
4.2 The Python training/export program, the exported `.onnx` model, the test set, and Python's own
predictions are all **versioned files** in the repo, so Go is checked against exactly what Python produced.
4.3 Pin every dependency version in every language (`go.mod`, `requirements.txt`).

## Article 5 — Documentation and Git hygiene **[Doc]**

5.1 One `README.md` is the complete documentation: purpose, roles of programs and data, how to build,
run, test and benchmark, results, and the **recommendation to management**.
5.2 One task per commit with clear messages; specs, plan, tasks, reviews, decisions versioned in the repo.
5.3 **Dialogs with LLM-based agents are included in the repository** as plain text or PDF (required deliverable).

## Article 6 — A finished, friendly application **[A]**

6.1 Documented build commands produce a macOS binary, a Windows `.exe`, and a macOS `.app`.
6.2 The program **runs to completion with no arguments** using bundled model and data.
6.3 User input is validated; bad input yields a clear message and a non-zero exit code — never a panic.
6.4 Output is clean and readable for a non-programmer; logs go to standard error.

## Article 7 — How we work with AI tools

7.1 **Claude Code** is the only tool that writes files in this repo during a task.
7.2 **Gemini (web)** independently reviews specs, plans, and the finished repository; it does not edit.
7.3 Luis approves every spec, plan, and task list, and reads every diff before it is committed.
7.4 If implementation reveals the spec is wrong, **update the spec first**, then the code.
7.5 Every review finding gets a recorded decision — accept, adjust, or reject — with a reason.
7.6 **Spike before planning:** the riskiest technical assumption is tested with a throwaway experiment
before the plan is written, and the result is recorded in `docs/spikes/`.
