//go:build darwin || linux

// Command measure runs complete programs several times and reports the median
// wall-clock time and peak memory (maximum resident set size) of each, for the
// whole-program comparison in spec FR-10. Copied unchanged in behavior from the
// Week 4 project (github.com/mioluigi/go-anscombe-stats), where it was tested.
//
// Peak memory comes from the operating system's per-process resource report
// (rusage), the same source /usr/bin/time uses. Time is measured with Go's
// monotonic clock, because /usr/bin/time reports only 10 ms steps, too coarse
// for a program that finishes in under a millisecond.
//
// Usage:
//
//	measure [-runs 5] label=command [label=command ...]
//
// Each argument is a label, "=", and a command line split on spaces (quoting is
// not supported). Program output is discarded. Example:
//
//	measure -runs 5 "go-xla=dist/predict-digits -backend xla -runs 0" "python-onnxruntime=python python/bench_python.py --runtime onnxruntime --predict-only"
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"syscall"
	"time"
)

const (
	exitOK      = 0
	exitFailure = 1 // a measured program failed to run
	exitUsage   = 2
)

var errBadProgram = errors.New(`program must be written as label=command`)

// program is one command to measure, with the label used in the report.
type program struct {
	label   string
	command []string
}

// sample is the measurement of one complete run.
type sample struct {
	wall        time.Duration
	maxRSSBytes int64
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("measure", flag.ContinueOnError)
	flags.SetOutput(stderr)
	runs := flags.Int("runs", 5, "times to run each program; the median is reported")
	flags.Usage = func() {
		fmt.Fprintln(flags.Output(), "Usage: measure [-runs 5] label=command [label=command ...]")
		flags.PrintDefaults()
	}
	if err := flags.Parse(args); err != nil {
		return exitUsage
	}
	if *runs < 1 || flags.NArg() == 0 {
		fmt.Fprintln(stderr, "measure: need -runs of at least 1 and at least one label=command")
		flags.Usage()
		return exitUsage
	}

	programs := make([]program, 0, flags.NArg())
	for _, arg := range flags.Args() {
		parsed, err := parseProgram(arg)
		if err != nil {
			fmt.Fprintf(stderr, "measure: %v\n", err)
			return exitUsage
		}
		programs = append(programs, parsed)
	}

	for _, p := range programs {
		walls, peaks, err := measureRepeatedly(p.command, *runs)
		if err != nil {
			fmt.Fprintf(stderr, "measure: %s: %v\n", p.label, err)
			return exitFailure
		}
		fmt.Fprintf(stdout, "wholeprogram language=%s runs=%d median_wall_ms=%.2f median_maxrss_mb=%.1f\n",
			p.label, *runs, float64(median(walls))/float64(time.Millisecond), float64(median(peaks))/(1<<20))
	}
	return exitOK
}

// parseProgram splits "label=command arg ..." into a label and a command.
func parseProgram(arg string) (program, error) {
	label, commandLine, found := strings.Cut(arg, "=")
	command := strings.Fields(commandLine)
	if !found || label == "" || len(command) == 0 {
		return program{}, fmt.Errorf("%w, got %q", errBadProgram, arg)
	}
	return program{label: label, command: command}, nil
}

// measureRepeatedly runs command the given number of times and returns each
// run's wall time and peak memory.
func measureRepeatedly(command []string, runs int) (walls []time.Duration, peaks []int64, err error) {
	for range runs {
		measured, runErr := measureOnce(command)
		if runErr != nil {
			return nil, nil, runErr
		}
		walls = append(walls, measured.wall)
		peaks = append(peaks, measured.maxRSSBytes)
	}
	return walls, peaks, nil
}

// measureOnce runs command to completion, discarding its output.
func measureOnce(command []string) (sample, error) {
	process := exec.Command(command[0], command[1:]...)
	started := time.Now()
	if err := process.Run(); err != nil {
		return sample{}, fmt.Errorf("running %q: %w", strings.Join(command, " "), err)
	}
	wall := time.Since(started)

	usage, ok := process.ProcessState.SysUsage().(*syscall.Rusage)
	if !ok {
		return sample{}, errors.New("this operating system does not report peak memory")
	}
	return sample{wall: wall, maxRSSBytes: maxRSSInBytes(usage.Maxrss)}, nil
}

// maxRSSInBytes converts rusage's Maxrss, which macOS reports in bytes but
// Linux reports in kilobytes.
func maxRSSInBytes(maxrss int64) int64 {
	if runtime.GOOS == "darwin" {
		return maxrss
	}
	return maxrss * 1024
}

// median returns the middle value without reordering values; for an even
// count it returns the mean of the two middle values.
func median[T time.Duration | int64](values []T) T {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}
	return (sorted[middle-1] + sorted[middle]) / 2
}
