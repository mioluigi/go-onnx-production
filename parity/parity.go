// Package parity checks whether two classifiers give the same answers: for example,
// a model run in Go against the same model run in Python. It compares class
// probabilities row by row and reports how many predicted labels agree and the
// largest probability difference. It has no dependency on any ML library.
package parity

import (
	"errors"
	"fmt"
	"math"
)

// Sentinel errors, for errors.Is.
var (
	ErrEmpty = errors.New("nothing to compare")
	ErrShape = errors.New("probability matrices have different shapes")
)

// Report summarizes a comparison of two probability matrices.
type Report struct {
	Rows         int     // rows (examples) compared
	LabelsAgree  int     // rows whose predicted labels match and whose values are all finite
	MaxAbsDiff   float64 // largest |got - want| over all finite values
	MaxDiffRow   int     // where MaxAbsDiff occurred
	MaxDiffClass int
	NonFinite    int     // rows containing NaN or infinity on either side
	Tolerance    float64 // largest acceptable difference
}

// Pass reports whether the two classifiers agree: every label matches, no value
// is NaN or infinite, and no probability differs by more than the tolerance.
func (r Report) Pass() bool {
	return r.LabelsAgree == r.Rows && r.NonFinite == 0 && r.MaxAbsDiff <= r.Tolerance
}

// String gives a one-line summary for logs and the program's output.
func (r Report) String() string {
	verdict := "FAIL"
	if r.Pass() {
		verdict = "PASS"
	}
	return fmt.Sprintf("%s: labels %d/%d identical, largest probability difference %.2e (tolerance %.0e), %d rows with NaN or infinity",
		verdict, r.LabelsAgree, r.Rows, r.MaxAbsDiff, r.Tolerance, r.NonFinite)
}

// Labels returns the predicted class of each row: the index of its largest
// probability. Ties go to the first class, as NumPy's argmax does.
func Labels(probabilities [][]float32) []int {
	labels := make([]int, len(probabilities))
	for row, values := range probabilities {
		best := 0
		for class, value := range values {
			if value > values[best] {
				best = class
			}
		}
		labels[row] = best
	}
	return labels
}

// Compare checks got against want, the reference, row by row.
func Compare(got, want [][]float32, tolerance float64) (Report, error) {
	if err := checkShapes(got, want); err != nil {
		return Report{}, err
	}
	report := Report{Rows: len(want), Tolerance: tolerance}
	gotLabels, wantLabels := Labels(got), Labels(want)
	for row := range want {
		if !allFinite(got[row]) || !allFinite(want[row]) {
			// NaN compares false with everything, so it must be counted explicitly
			// rather than left to slip through a "difference > tolerance" test.
			report.NonFinite++
			continue
		}
		if gotLabels[row] == wantLabels[row] {
			report.LabelsAgree++
		}
		report.recordDifferences(row, got[row], want[row])
	}
	return report, nil
}

// recordDifferences updates the largest difference seen so far with one row.
func (r *Report) recordDifferences(row int, got, want []float32) {
	for class := range want {
		difference := math.Abs(float64(got[class]) - float64(want[class]))
		if difference > r.MaxAbsDiff {
			r.MaxAbsDiff, r.MaxDiffRow, r.MaxDiffClass = difference, row, class
		}
	}
}

func checkShapes(got, want [][]float32) error {
	if len(want) == 0 {
		return ErrEmpty
	}
	if len(got) != len(want) {
		return fmt.Errorf("%w: %d rows compared with %d", ErrShape, len(got), len(want))
	}
	for row := range want {
		if len(want[row]) == 0 || len(got[row]) != len(want[row]) {
			return fmt.Errorf("%w: row %d has %d classes compared with %d", ErrShape, row, len(got[row]), len(want[row]))
		}
	}
	return nil
}

func allFinite(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}
