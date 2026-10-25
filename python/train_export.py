"""Train an MNIST digit classifier in PyTorch and export it to ONNX for the Go program.

Writes to assets/ (all used by Go and by python/bench_python.py):
  mnist_mlp.onnx                        the exported model: input "input" (batch x 784) -> "probabilities" (batch x 10)
  mnist_mlp.pt                          the trained PyTorch weights
  t10k-images-idx3-ubyte.gz             official MNIST test images, unchanged
  t10k-labels-idx1-ubyte.gz             official MNIST test labels, unchanged
  pytorch-probabilities-idx2-float.gz   PyTorch's probabilities for all 10,000 test images (the parity reference)

Before saving, the export is checked: ONNX Runtime must reproduce PyTorch's probabilities within 1e-5.

Usage, from the repository root inside the virtual environment:
    python python/train_export.py [--epochs 5] [--seed 42]
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

from mnist_io import (onnx_summary, onnxruntime_probabilities, read_idx, to_model_input, write_idx)

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
DOWNLOAD_CACHE = REPO / ".mnist-cache"
EXPORT_TOLERANCE = 1e-5
# ONNX Runtime may lag the newest onnx package; IR version 10 works with opset 18 (spike finding 7).
ONNX_IR_VERSION = 10
ONNX_OPSET = 18


class DigitClassifier(nn.Module):
    """Multilayer network 784 -> 128 -> 64 -> 10. Outputs logits, as CrossEntropyLoss expects for training."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class WithProbabilities(nn.Module):
    """The exported model: the classifier followed by softmax, so its output is class probabilities."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, inputs):
        return torch.softmax(self.classifier(inputs), dim=1)


def load_split(train):
    """Return (model inputs, labels) for one MNIST split, applying the shared input contract."""
    dataset = MNIST(DOWNLOAD_CACHE, train=train, download=True)
    inputs = torch.from_numpy(to_model_input(dataset.data.numpy()))
    return inputs, dataset.targets


def train(classifier, inputs, labels, epochs, batch_size=128):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    classifier.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(inputs))
        total_loss = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(classifier(inputs[batch]), labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"epoch {epoch}/{epochs}: training loss {total_loss / len(inputs):.4f}")


def export_onnx(model, onnx_path):
    """Export with a dynamic batch dimension, then pin a file version ONNX Runtime can read."""
    import onnx

    example = torch.zeros(2, 784)
    batch = torch.export.Dim("batch")
    program = torch.onnx.export(model, (example,), dynamo=True, opset_version=ONNX_OPSET,
                                input_names=["input"], output_names=["probabilities"],
                                dynamic_shapes={"inputs": {0: batch}})
    program.save(str(onnx_path))
    exported = onnx.load(str(onnx_path))
    exported.ir_version = min(exported.ir_version, ONNX_IR_VERSION)
    onnx.save(exported, str(onnx_path))


def copy_official_test_files():
    """Copy MNIST's own compressed test files, so Go reads exactly what was published."""
    raw = DOWNLOAD_CACHE / "MNIST" / "raw"
    for name in ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        shutil.copyfile(raw / name, ASSETS / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=5, help="training passes over the 60,000 images (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, for repeatable training (default 42)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    torch.manual_seed(args.seed)
    ASSETS.mkdir(exist_ok=True)
    print(f"PyTorch {torch.__version__}")

    train_inputs, train_labels = load_split(train=True)
    test_inputs, test_labels = load_split(train=False)
    classifier = DigitClassifier()
    started = time.perf_counter()
    train(classifier, train_inputs, train_labels, args.epochs)
    print(f"training time: {time.perf_counter() - started:.1f} s")

    model = WithProbabilities(classifier).eval()
    with torch.no_grad():
        pytorch_probabilities = model(test_inputs).numpy()
    accuracy = (pytorch_probabilities.argmax(axis=1) == test_labels.numpy()).mean()
    print(f"test accuracy: {accuracy:.4f} on {len(test_labels)} images")

    onnx_path = ASSETS / "mnist_mlp.onnx"
    export_onnx(model, onnx_path)
    summary = onnx_summary(onnx_path)
    print(f"ONNX export: {summary}")

    # Verify the export before anything else depends on it.
    runtime_probabilities = onnxruntime_probabilities(onnx_path, test_inputs.numpy())
    difference = float(np.abs(runtime_probabilities - pytorch_probabilities).max())
    print(f"ONNX Runtime vs PyTorch: largest probability difference {difference:.2e}")
    if not difference <= EXPORT_TOLERANCE:
        raise SystemExit(f"export check failed: difference {difference:.2e} exceeds {EXPORT_TOLERANCE:.0e}")

    torch.save(classifier.state_dict(), ASSETS / "mnist_mlp.pt")
    copy_official_test_files()
    write_idx(ASSETS / "pytorch-probabilities-idx2-float.gz", pytorch_probabilities.astype(np.float32))

    # Confirm the saved files read back as what was used here.
    images = read_idx(ASSETS / "t10k-images-idx3-ubyte.gz")
    if not np.array_equal(to_model_input(images), test_inputs.numpy()):
        raise SystemExit("saved test images do not match the images used for prediction")
    print("saved:", ", ".join(f"{path.name} ({path.stat().st_size / 1024:.0f} KB)" for path in sorted(ASSETS.iterdir())))


if __name__ == "__main__":
    main()
