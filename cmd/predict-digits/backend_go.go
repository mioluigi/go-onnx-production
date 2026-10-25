//go:build !(cgo && (darwin || linux))

package main

// defaultBackend is the pure-Go backend in builds without XLA, such as the
// Windows .exe built with CGO_ENABLED=0.
const defaultBackend = "go"
