package parity

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

const tolerance = 1e-5

func TestLabels(t *testing.T) {
	probabilities := [][]float32{
		{0.1, 0.7, 0.2},
		{0.6, 0.3, 0.1},
		{0.2, 0.3, 0.5},
		{0.4, 0.4, 0.2}, // tie: the first class wins, as NumPy's argmax does
	}
	if got, want := Labels(probabilities), []int{1, 0, 2, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("Labels = %v, want %v", got, want)
	}
}

func TestCompare(t *testing.T) {
	reference := [][]float32{{0.9, 0.1}, {0.2, 0.8}, {0.6, 0.4}}
	nan := float32(math.NaN())

	tests := []struct {
		name        string
		got         [][]float32
		wantPass    bool
		wantAgree   int
		wantMaxDiff float64
		wantNonFin  int
	}{
		{"identical", reference, true, 3, 0, 0},
		{"just inside tolerance", [][]float32{{0.9, 0.1}, {0.2, 0.8}, {0.600009, 0.399991}}, true, 3, 9e-6, 0},
		{"just outside tolerance", [][]float32{{0.9, 0.1}, {0.2, 0.8}, {0.60002, 0.39998}}, false, 3, 2e-5, 0},
		{"one label flips", [][]float32{{0.9, 0.1}, {0.2, 0.8}, {0.4, 0.6}}, false, 2, 0.2, 0},
		{"NaN is never a match", [][]float32{{0.9, 0.1}, {nan, 0.8}, {0.6, 0.4}}, false, 2, 0, 1},
		{"infinity is never a match", [][]float32{{0.9, float32(math.Inf(1))}, {0.2, 0.8}, {0.6, 0.4}}, false, 2, 0, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			report, err := Compare(tc.got, reference, tolerance)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if report.Pass() != tc.wantPass {
				t.Errorf("Pass() = %v, want %v; report %+v", report.Pass(), tc.wantPass, report)
			}
			if report.Rows != 3 || report.LabelsAgree != tc.wantAgree || report.NonFinite != tc.wantNonFin {
				t.Errorf("rows %d, labels agree %d, non-finite %d; want 3, %d, %d",
					report.Rows, report.LabelsAgree, report.NonFinite, tc.wantAgree, tc.wantNonFin)
			}
			if math.Abs(report.MaxAbsDiff-tc.wantMaxDiff) > 1e-6 {
				t.Errorf("MaxAbsDiff = %g, want %g", report.MaxAbsDiff, tc.wantMaxDiff)
			}
		})
	}
}

func TestCompareRecordsWhereTheLargestDifferenceIs(t *testing.T) {
	want := [][]float32{{0.5, 0.5}, {0.5, 0.5}}
	got := [][]float32{{0.5, 0.5}, {0.5001, 0.4999}}
	report, err := Compare(got, want, tolerance)
	if err != nil {
		t.Fatal(err)
	}
	if report.MaxDiffRow != 1 || report.MaxDiffClass != 0 {
		t.Errorf("largest difference at row %d, class %d; want row 1, class 0", report.MaxDiffRow, report.MaxDiffClass)
	}
}

func TestCompareRejectsMismatchedShapes(t *testing.T) {
	tests := []struct {
		name      string
		got, want [][]float32
		wantErr   error
	}{
		{"no rows", nil, nil, ErrEmpty},
		{"different row counts", [][]float32{{1, 0}}, [][]float32{{1, 0}, {0, 1}}, ErrShape},
		{"different class counts", [][]float32{{1, 0}}, [][]float32{{1, 0, 0}}, ErrShape},
		{"an empty row", [][]float32{{}}, [][]float32{{}}, ErrShape},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Compare(tc.got, tc.want, tolerance); !errors.Is(err, tc.wantErr) {
				t.Errorf("Compare error = %v, want errors.Is(%v)", err, tc.wantErr)
			}
		})
	}
}

func TestReportString(t *testing.T) {
	pass := Report{Rows: 10000, LabelsAgree: 10000, MaxAbsDiff: 2.4e-7, Tolerance: 1e-5}
	if got, want := pass.String(), "PASS: labels 10000/10000 identical, largest probability difference 2.40e-07 (tolerance 1e-05), 0 rows with NaN or infinity"; got != want {
		t.Errorf("String() =\n%q\nwant\n%q", got, want)
	}
	fail := Report{Rows: 10, LabelsAgree: 9, Tolerance: 1e-5}
	if got := fail.String(); got[:4] != "FAIL" {
		t.Errorf("String() = %q, want it to start with FAIL", got)
	}
}
