# Assignment Requirements and Rubric (summary)

> Summary of the Week 5 assignment, *Go for ML/AI Production* (100 points, due 2026-10-25), in our own
> words, so every tool and reviewer can check the spec against it. The Canvas page is the authoritative source.

## Scenario

A startup's managers are concerned about the **cost and performance** of running Python deep learning
models in production, with potentially **thousands of users demanding fast response times**. They ask
whether a model can be **developed in Python** (scikit-learn, TensorFlow/Keras, or PyTorch), **exported to
a common format** (ONNX, the Open Neural Network Exchange), and **run in production with Go** (GoMLX, for
example). We act as the outside consultant.

## Suggested steps

1. Develop a deep learning (machine learning or AI) model in TensorFlow/Keras, scikit-learn, or PyTorch —
   our own, from a textbook, or from the web.
2. Obtain predictions from the Python model on a **large test set**; record Python's execution time.
3. Export the model in **ONNX** format.
4. Read the ONNX model into **GoMLX**.
5. Run the GoMLX model and show its predictions **match** the Python predictions on the test set.
6. Record Go's execution time.
7. In the README, **compare execution times** under Python and Go.

## Constraints

- AI code-generation and assistant tools used for the assignment should be **free versions**; no paid
  subscriptions are required.

## Deliverables

- GitHub repository URL in the Canvas comments form.
- **File(s) showing dialogs with LLM-based agents or services**, as plain text or PDF, in the repository.
- `README.md` with complete documentation for the assignment.

## Rubric (5 criteria × 20 points)

Each criterion: Outstanding 20 (all criteria met) · Meets Expectations 15 · Needs Improvement 10 · Partially Acceptable 5 · Unacceptable 0.

| Criterion | What earns full marks |
|---|---|
| **Coding rules, organization, aesthetics** | Go modules and idiomatic Go. Readable code; meaningful, specific names, not too long or short. No useless temporary variables. Clean, consistent blocks and line breaks. Large blocks split into single-purpose pieces. Simple control flow. Distribute the *not* in complex Boolean expressions. Self-documenting code with comments explaining the logic (McConnell 2004, 777–817). |
| **Testing and software metrics** | Unit tests of critical components, synthetic test data where appropriate. Program logs and profiles where appropriate. Monitor memory and processing requirements of components and the whole program. Monte Carlo performance benchmark if the requirements call for it. |
| **Design and development** | Clean, efficient, easy-to-understand design meeting all requirements and serving the use case. General-purpose, reusable modules where possible. |
| **Documentation** | Effective use of Git/GitHub; a README.md noting the roles of programs and data and explaining how to test and use the application. |
| **Application** | Instructions for creating an executable (.exe on Windows or .app on macOS). Runs to completion without issues. Validates user input and explains incorrect input. Clean UX/UI. |

## Reference

Vivien, Vladimir. 2025. "Go in the AI/ML Landscape: A Practical Guide." Medium.
