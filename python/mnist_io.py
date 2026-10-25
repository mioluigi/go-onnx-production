"""Helpers shared by the Python programs that do not need PyTorch.

* The **input contract** (spec FR-3a): how raw MNIST pixels become model inputs. Go applies the same rule.
* Reading and writing the **IDX format** MNIST is published in (big-endian header, then raw values).
* Checking an exported ONNX model with ONNX Runtime and summarizing its graph.
"""

import gzip
import struct
from pathlib import Path

import numpy as np

PIXELS_PER_IMAGE = 28 * 28
PIXEL_SCALE = np.float32(255.0)

# IDX data-type codes (http://yann.lecun.com/exdb/mnist/ "THE IDX FILE FORMAT").
IDX_UBYTE = 0x08
IDX_FLOAT32 = 0x0D
_DTYPES = {IDX_UBYTE: np.dtype(">u1"), IDX_FLOAT32: np.dtype(">f4")}


def to_model_input(images_uint8):
    """Input contract: flatten each 28x28 image row by row to 784 values, as float32, divided by 255."""
    images = np.asarray(images_uint8)
    if images.dtype != np.uint8:
        raise ValueError(f"images must be uint8 pixels, got {images.dtype}")
    return images.reshape(len(images), PIXELS_PER_IMAGE).astype(np.float32) / PIXEL_SCALE


def write_idx(path, array):
    """Write a uint8 or float32 array as a gzip-compressed IDX file, identical bytes for identical data."""
    array = np.asarray(array)
    if array.dtype == np.uint8:
        type_code = IDX_UBYTE
    elif array.dtype == np.float32:
        type_code = IDX_FLOAT32
    else:
        raise ValueError(f"IDX supports uint8 and float32 here, got {array.dtype}")
    header = struct.pack(">BBBB", 0, 0, type_code, array.ndim) + struct.pack(f">{array.ndim}I", *array.shape)
    # mtime=0 and no stored file name make the output byte-for-byte reproducible.
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
        out.write(header)
        out.write(array.astype(_DTYPES[type_code]).tobytes())


def read_idx(path):
    """Read an IDX file (gzip-compressed or plain) into a NumPy array with native byte order."""
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as source:
        zero1, zero2, type_code, ndim = struct.unpack(">BBBB", source.read(4))
        if (zero1, zero2) != (0, 0) or type_code not in _DTYPES:
            raise ValueError(f"{path}: not a supported IDX file (header {zero1}, {zero2}, {type_code:#x})")
        shape = struct.unpack(f">{ndim}I", source.read(4 * ndim))
        values = np.frombuffer(source.read(), dtype=_DTYPES[type_code])
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header says {shape}, but file holds {values.size} values")
    return values.reshape(shape).astype(_DTYPES[type_code].newbyteorder("="))


def onnx_summary(onnx_path):
    """Describe an ONNX model: IR version, opsets, operations, input and output names."""
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return {
        "ir_version": model.ir_version,
        "opsets": {opset.domain or "ai.onnx": opset.version for opset in model.opset_import},
        "operations": sorted({node.op_type for node in model.graph.node}),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def onnxruntime_probabilities(onnx_path, inputs):
    """Run an ONNX model with ONNX Runtime on CPU and return its single output."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inputs})[0]
