//go:build cgo && (darwin || linux)

package main

// The XLA backend needs CGo and runs on macOS and Linux. On first use it
// downloads its plugin (from GitHub) into the user's cache; later runs reuse it.
import _ "github.com/gomlx/go-xla/compute/xla/autoinstall"

// defaultBackend is XLA where it is built in, because it is GoMLX's faster backend.
const defaultBackend = "xla"
