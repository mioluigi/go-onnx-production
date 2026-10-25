//go:build darwin || linux

package main

import (
	"errors"
	"io"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestMedian(t *testing.T) {
	tests := []struct {
		name   string
		values []int64
		want   int64
	}{
		{"single value", []int64{7}, 7},
		{"odd count, unsorted", []int64{9, 1, 5}, 5},
		{"even count averages the middle two", []int64{4, 1, 3, 2}, 2}, // (2 + 3) / 2 rounds down
		{"does not reorder the caller's slice", []int64{3, 2, 1}, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			original := append([]int64(nil), tc.values...)
			if got := median(tc.values); got != tc.want {
				t.Errorf("median(%v) = %d, want %d", original, got, tc.want)
			}
			if !reflect.DeepEqual(tc.values, original) {
				t.Errorf("median reordered its input: %v, was %v", tc.values, original)
			}
		})
	}
}

func TestMedianDurations(t *testing.T) {
	got := median([]time.Duration{3 * time.Millisecond, time.Millisecond, 2 * time.Millisecond})
	if got != 2*time.Millisecond {
		t.Errorf("median = %v, want 2ms", got)
	}
}

func TestParseProgram(t *testing.T) {
	got, err := parseProgram("python=python3 script.py --fit-only")
	if err != nil {
		t.Fatalf("parseProgram: %v", err)
	}
	want := program{label: "python", command: []string{"python3", "script.py", "--fit-only"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseProgram = %+v, want %+v", got, want)
	}

	for _, bad := range []string{"no-equals-sign", "=command", "label=", "label=   "} {
		if _, err := parseProgram(bad); !errors.Is(err, errBadProgram) {
			t.Errorf("parseProgram(%q) error = %v, want errBadProgram", bad, err)
		}
	}
}

// TestMeasureOnce runs this test binary itself in a mode that exits at once,
// so the test needs no external program.
func TestMeasureOnce(t *testing.T) {
	if os.Getenv("MEASURE_TEST_CHILD") == "exit" {
		os.Exit(0)
	}
	t.Setenv("MEASURE_TEST_CHILD", "exit")

	got, err := measureOnce([]string{os.Args[0], "-test.run=^TestMeasureOnce$"})
	if err != nil {
		t.Fatalf("measureOnce: %v", err)
	}
	if got.wall <= 0 {
		t.Errorf("wall time = %v, want > 0", got.wall)
	}
	if got.maxRSSBytes < 1<<20 {
		t.Errorf("max RSS = %d bytes, want at least 1 MB for a running Go program", got.maxRSSBytes)
	}
}

func TestMeasureOnceReportsFailure(t *testing.T) {
	if _, err := measureOnce([]string{"no-such-program-anywhere"}); err == nil {
		t.Error("measureOnce succeeded for a program that does not exist")
	}
}

func TestRunRejectsBadArguments(t *testing.T) {
	for _, args := range [][]string{
		{},                        // no programs
		{"-runs", "0", "go=true"}, // runs must be positive
		{"go-without-equals"},     // malformed program
	} {
		if code := run(args, io.Discard, io.Discard); code != exitUsage {
			t.Errorf("run(%q) = %d, want %d", args, code, exitUsage)
		}
	}
}

// TestRunReportsMedians checks the full path and the exact output format that
// scripts/bench_table.sh parses.
func TestRunReportsMedians(t *testing.T) {
	t.Setenv("MEASURE_TEST_CHILD", "exit")
	var stdout strings.Builder
	code := run([]string{"-runs", "3", "self=" + os.Args[0] + " -test.run=^TestMeasureOnce$"}, &stdout, io.Discard)
	if code != exitOK {
		t.Fatalf("run exit code = %d, want %d", code, exitOK)
	}
	pattern := `^wholeprogram language=self runs=3 median_wall_ms=\d+\.\d{2} median_maxrss_mb=\d+\.\d\n$`
	if !regexp.MustCompile(pattern).MatchString(stdout.String()) {
		t.Errorf("output %q does not match %s", stdout.String(), pattern)
	}
}

func TestRunReportsFailingProgram(t *testing.T) {
	if code := run([]string{"missing=no-such-program-anywhere"}, io.Discard, io.Discard); code != exitFailure {
		t.Errorf("run exit code = %d, want %d", code, exitFailure)
	}
}
