//go:build cgo && (darwin || linux)

package inference

import (
	"slices"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/go-xla/compute/xla/autoinstall" // lets the benchmarks run with GOMLX_BACKEND=xla
)

// TestXLABackendIsLinked guards scripts/measure.sh, which runs the benchmarks
// with GOMLX_BACKEND=xla: the backend must be compiled into this package's
// test binary, not only into cmd/predict-digits (found in T15). It checks the
// registration only, so it needs no network access to XLA's plugin.
func TestXLABackendIsLinked(t *testing.T) {
	if !slices.Contains(compute.List(), "xla") {
		t.Errorf("registered backends %v; want xla too", compute.List())
	}
}
