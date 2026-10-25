"""Generate tiny ONNX models with known answers for the Go unit tests (inference/testdata/).

Each model exercises one behavior the Go `inference` package must handle:

  softmax_3.onnx            3 features -> softmax of the input itself (identity weights, zero bias)
  prefers_class_2.onnx      4 features, 3 classes; zero weights and a large bias, so class 2 always wins
  two_layer.onnx            8 -> 6 -> 4 with random weights, PyTorch "Gemm" style; expected outputs saved as IDX
  sklearn_bias.onnx         like two_layer but MatMul + Add with a 2-D bias [1, n], as scikit-learn exports
  named_logits.onnx         valid model whose output is named "logits", not "probabilities"
  two_inputs.onnx           a model with two inputs, which the package does not support
  unsupported_op.onnx       uses ArrayFeatureExtractor, an operation GoMLX does not implement
  sqrt_output.onnx          outputs the square root of its input, so a negative input yields NaN
  always_zero_784.onnx      MNIST-shaped (784 -> 10) but always predicts 0: a parity FAIL for cmd/predict-digits

Usage, from the repository root:  python python/make_test_models.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mnist_io import onnxruntime_probabilities, write_idx

OUT = Path(__file__).resolve().parent.parent / "inference" / "testdata"
IR_VERSION, OPSET = 10, 18  # readable by ONNX Runtime and GoMLX (spike finding 7)


def tensor_input(name, features):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", features])


def save(file_name, nodes, inputs, outputs, initializers, extra_opsets=()):
    graph = helper.make_graph(nodes, file_name.removesuffix(".onnx"), inputs, outputs, initializers)
    opsets = [helper.make_opsetid("", OPSET), *extra_opsets]
    model = helper.make_model(graph, opset_imports=opsets, ir_version=IR_VERSION)
    onnx.checker.check_model(model)
    onnx.save(model, OUT / file_name)
    return OUT / file_name


def dense_softmax(file_name, weights, bias, features, classes, output_name="probabilities"):
    """MatMul + Add + Softmax: the smallest classifier shape."""
    nodes = [helper.make_node("MatMul", ["input", "W"], ["scores"]),
             helper.make_node("Add", ["scores", "b"], ["shifted_scores"]),
             helper.make_node("Softmax", ["shifted_scores"], [output_name], axis=-1)]
    return save(file_name, nodes, [tensor_input("input", features)],
                [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch", classes])],
                [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "b")])


def two_layer(file_name, w1, b1, w2, b2, gemm_style):
    """8 -> 6 (ReLU) -> 4 (softmax), in PyTorch's Gemm style or scikit-learn's MatMul + 2-D bias style."""
    if gemm_style:
        nodes = [helper.make_node("Gemm", ["input", "W1t", "b1"], ["h"], transB=1),
                 helper.make_node("Relu", ["h"], ["r"]),
                 helper.make_node("Gemm", ["r", "W2t", "b2"], ["logits"], transB=1)]
        weights = [numpy_helper.from_array(w1.T.copy(), "W1t"), numpy_helper.from_array(b1, "b1"),
                   numpy_helper.from_array(w2.T.copy(), "W2t"), numpy_helper.from_array(b2, "b2")]
    else:
        nodes = [helper.make_node("MatMul", ["input", "W1"], ["m1"]), helper.make_node("Add", ["m1", "b1"], ["h"]),
                 helper.make_node("Relu", ["h"], ["r"]),
                 helper.make_node("MatMul", ["r", "W2"], ["m2"]), helper.make_node("Add", ["m2", "b2"], ["logits"])]
        weights = [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(b1.reshape(1, -1), "b1"),
                   numpy_helper.from_array(w2, "W2"), numpy_helper.from_array(b2.reshape(1, -1), "b2")]
    nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1))
    return save(file_name, nodes, [tensor_input("input", 8)],
                [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 4])], weights)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(431)

    dense_softmax("softmax_3.onnx", np.eye(3, dtype=np.float32), np.zeros(3, np.float32), 3, 3)
    dense_softmax("prefers_class_2.onnx", np.zeros((4, 3), np.float32), np.array([0, 0, 20], np.float32), 4, 3)
    dense_softmax("named_logits.onnx", np.eye(3, dtype=np.float32), np.zeros(3, np.float32), 3, 3,
                  output_name="logits")

    w1, b1 = rng.normal(size=(8, 6)).astype(np.float32), rng.normal(size=6).astype(np.float32)
    w2, b2 = rng.normal(size=(6, 4)).astype(np.float32), rng.normal(size=4).astype(np.float32)
    inputs = rng.normal(size=(5, 8)).astype(np.float32)
    gemm_path = two_layer("two_layer.onnx", w1, b1, w2, b2, gemm_style=True)
    sklearn_path = two_layer("sklearn_bias.onnx", w1, b1, w2, b2, gemm_style=False)
    expected = onnxruntime_probabilities(gemm_path, inputs)
    if not np.allclose(onnxruntime_probabilities(sklearn_path, inputs), expected, atol=1e-6):
        raise SystemExit("the two two-layer variants disagree; check make_test_models.py")
    write_idx(OUT / "two_layer_inputs-idx2-float.gz", inputs)
    write_idx(OUT / "two_layer_expected-idx2-float.gz", expected.astype(np.float32))

    nodes = [helper.make_node("Add", ["a", "b"], ["probabilities"])]
    save("two_inputs.onnx", nodes, [tensor_input("a", 3), tensor_input("b", 3)],
         [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 3])], [])

    nodes = [helper.make_node("Softmax", ["input"], ["scores"], axis=-1),
             helper.make_node("ArrayFeatureExtractor", ["scores", "columns"], ["probabilities"], domain="ai.onnx.ml")]
    save("unsupported_op.onnx", nodes, [tensor_input("input", 3)],
         [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 2])],
         [numpy_helper.from_array(np.array([0, 2], np.int64), "columns")],
         extra_opsets=[helper.make_opsetid("ai.onnx.ml", 3)])

    nodes = [helper.make_node("Sqrt", ["input"], ["probabilities"])]
    save("sqrt_output.onnx", nodes, [tensor_input("input", 2)],
         [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 2])], [])

    bias = np.zeros(10, np.float32)
    bias[0] = 5
    dense_softmax("always_zero_784.onnx", np.zeros((784, 10), np.float32), bias, 784, 10)

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:40s} {path.stat().st_size:6d} bytes")


if __name__ == "__main__":
    main()
