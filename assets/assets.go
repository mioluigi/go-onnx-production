// Package assets embeds the trained model and the MNIST test set in any Go
// program that imports it, so the program runs with no other files (spec FR-11).
//
// The files are written by python/train_export.py; see that program for how
// each one is produced.
package assets

import _ "embed"

// ModelONNX is the trained classifier exported from PyTorch: input "input"
// (batch × 784 float32, pixels ÷ 255), output "probabilities" (batch × 10).
//
//go:embed mnist_mlp.onnx
var ModelONNX []byte

// TestImages is the official MNIST test-image file (IDX format, gzip),
// 10,000 images of 28 × 28 unsigned-byte pixels, unchanged.
//
//go:embed t10k-images-idx3-ubyte.gz
var TestImages []byte

// TestLabels is the official MNIST test-label file (IDX format, gzip), unchanged.
//
//go:embed t10k-labels-idx1-ubyte.gz
var TestLabels []byte

// PyTorchProbabilities holds PyTorch's class probabilities for every test
// image (IDX format, gzip, 10,000 × 10 float32): the reference for the parity check.
//
//go:embed pytorch-probabilities-idx2-float.gz
var PyTorchProbabilities []byte
