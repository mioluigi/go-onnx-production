"""Time the MNIST model in Python, the same way cmd/predict-digits times it in Go (plan section 8).

For each runtime (PyTorch, and ONNX Runtime running the exported model) it measures:
  load     read the model and prepare it to run
  first    the first prediction, one image
  bulk     all 10,000 test images in one call (median of --bulk-runs)
  latency  one image per call (median and 95th percentile of --latency-calls, after warm-up)
It also checks accuracy against the true labels and parity with PyTorch's saved probabilities.

Usage, from the repository root inside the virtual environment:
    python python/bench_python.py                       full timing, both runtimes
    python python/bench_python.py --runtime onnxruntime one runtime only
    python python/bench_python.py --predict-only        load and predict once (for scripts/measure.sh)
"""

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mnist_io import read_idx, to_model_input

REPO = Path(__file__).resolve().parent.parent
PARITY_TOLERANCE = 1e-5
WARM_UP_CALLS = 20


def load_pytorch(assets):
    import torch
    from mnist_model import DigitClassifier, WithProbabilities

    classifier = DigitClassifier()
    classifier.load_state_dict(torch.load(assets / "mnist_mlp.pt", weights_only=True))
    model = WithProbabilities(classifier).eval()

    def predict(inputs):
        with torch.no_grad():
            return model(torch.from_numpy(inputs)).numpy()

    return predict, f"PyTorch {torch.__version__}"


def load_onnxruntime(assets):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(assets / "mnist_mlp.onnx"), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(inputs):
        return session.run(None, {input_name: inputs})[0]

    return predict, f"ONNX Runtime {onnxruntime.__version__}"


LOADERS = {"pytorch": load_pytorch, "onnxruntime": load_onnxruntime}


def milliseconds_since(start_ns):
    return (time.perf_counter_ns() - start_ns) / 1e6


def time_runtime(name, assets, inputs, labels, reference, args):
    """Measure one runtime and print its results and a parseable benchmark line."""
    start = time.perf_counter_ns()
    predict, version = LOADERS[name](assets)
    load_ms = milliseconds_since(start)

    start = time.perf_counter_ns()
    predict(inputs[:1])
    first_ms = milliseconds_since(start)

    probabilities = predict(inputs)
    accuracy = float((probabilities.argmax(axis=1) == labels).mean())
    difference = float(np.abs(probabilities - reference).max())
    labels_agree = int((probabilities.argmax(axis=1) == reference.argmax(axis=1)).sum())
    verdict = "PASS" if labels_agree == len(labels) and difference <= PARITY_TOLERANCE else "FAIL"
    print(f"{version}: accuracy {accuracy:.4f}; parity with PyTorch {verdict} "
          f"(labels {labels_agree}/{len(labels)}, largest difference {difference:.2e})")
    if args.predict_only:
        return

    bulk_ms = statistics.median(timed_ms(lambda: predict(inputs)) for _ in range(args.bulk_runs))
    for row in range(WARM_UP_CALLS):
        predict(inputs[row:row + 1])
    latencies_us = [timed_ms(lambda row=row: predict(inputs[row:row + 1])) * 1000
                    for row in range(args.latency_calls)]
    p50, p95 = np.percentile(latencies_us, [50, 95])
    print(f"benchmark language=python runtime={name} load_ms={load_ms:.2f} first_ms={first_ms:.2f} "
          f"bulk_ms={bulk_ms:.2f} latency_p50_us={p50:.1f} latency_p95_us={p95:.1f} "
          f"bulk_runs={args.bulk_runs} latency_calls={args.latency_calls}")


def timed_ms(action):
    start = time.perf_counter_ns()
    action()
    return milliseconds_since(start)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", choices=["all", *LOADERS], default="all", help="which runtime to time (default all)")
    parser.add_argument("--bulk-runs", type=int, default=20, help="timed 10,000-image predictions (default 20)")
    parser.add_argument("--latency-calls", type=int, default=1000, help="timed one-image predictions (default 1000)")
    parser.add_argument("--predict-only", action="store_true", help="load and predict once; no timing loops")
    parser.add_argument("--assets", type=Path, default=REPO / "assets", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.bulk_runs < 1 or not 1 <= args.latency_calls <= 10_000:
        parser.error("--bulk-runs must be at least 1 and --latency-calls between 1 and 10000")

    inputs = to_model_input(read_idx(args.assets / "t10k-images-idx3-ubyte.gz"))
    labels = read_idx(args.assets / "t10k-labels-idx1-ubyte.gz")
    reference = read_idx(args.assets / "pytorch-probabilities-idx2-float.gz")
    print(f"Python {platform.python_version()} · {len(inputs)} test images")
    for name in LOADERS if args.runtime == "all" else [args.runtime]:
        time_runtime(name, args.assets, inputs, labels, reference, args)


if __name__ == "__main__":
    main()
