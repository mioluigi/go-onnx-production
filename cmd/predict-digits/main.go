// Command predict-digits runs the MNIST digit classifier exported from PyTorch
// to ONNX in Go with GoMLX. (T10 stub: the tests are written first; T11 implements.)
package main

import (
	"io"
	"os"
)

const (
	exitOK        = 0
	exitDataError = 1
	exitUsage     = 2
)

const pixelsPerImage = 28 * 28

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int { return exitDataError }

func toModelInput(pixels []uint8) []float32 { return nil }

func percentile(sorted []float64, p float64) float64 { return 0 }
