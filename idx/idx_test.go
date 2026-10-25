package idx

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

// encode builds an IDX file in memory: two zero bytes, the type code, the number of
// dimensions, each dimension as a big-endian uint32, then the values big-endian.
func encode(t *testing.T, dataType DataType, shape []uint32, values any) []byte {
	t.Helper()
	var buffer bytes.Buffer
	buffer.Write([]byte{0, 0, byte(dataType), byte(len(shape))})
	for _, size := range shape {
		binary.Write(&buffer, binary.BigEndian, size)
	}
	if err := binary.Write(&buffer, binary.BigEndian, values); err != nil {
		t.Fatalf("encoding values: %v", err)
	}
	return buffer.Bytes()
}

func compress(t *testing.T, plain []byte) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := gzip.NewWriter(&buffer)
	writer.Write(plain)
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func TestReadValid(t *testing.T) {
	images := encode(t, UnsignedByte, []uint32{2, 2, 3}, []uint8{0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255})
	probabilities := encode(t, Float32, []uint32{2, 2}, []float32{0.25, 0.75, 1, -1.5})
	labels := encode(t, UnsignedByte, []uint32{3}, []uint8{7, 2, 1})

	tests := []struct {
		name  string
		input []byte
		want  Array
	}{
		{"3-D unsigned bytes (images)", images,
			Array{Type: UnsignedByte, Shape: []int{2, 2, 3}, Bytes: []uint8{0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255}}},
		{"same file, gzip-compressed", compress(t, images),
			Array{Type: UnsignedByte, Shape: []int{2, 2, 3}, Bytes: []uint8{0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255}}},
		{"2-D float32 (probabilities)", probabilities,
			Array{Type: Float32, Shape: []int{2, 2}, Floats: []float32{0.25, 0.75, 1, -1.5}}},
		{"1-D unsigned bytes (labels)", compress(t, labels),
			Array{Type: UnsignedByte, Shape: []int{3}, Bytes: []uint8{7, 2, 1}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Read(bytes.NewReader(tc.input))
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Read = %+v\nwant   %+v", got, tc.want)
			}
		})
	}
}

func TestReadInvalid(t *testing.T) {
	valid := encode(t, UnsignedByte, []uint32{2, 2}, []uint8{1, 2, 3, 4})
	tests := []struct {
		name    string
		input   []byte
		wantErr error
	}{
		{"empty input", nil, ErrHeader},
		{"header cut short", valid[:3], ErrHeader},
		{"dimension sizes cut short", valid[:6], ErrHeader},
		{"first bytes not zero", append([]byte{1, 0}, valid[2:]...), ErrHeader},
		{"no dimensions", []byte{0, 0, byte(UnsignedByte), 0}, ErrHeader},
		{"unsupported type (int16)", encode(t, 0x0B, []uint32{2}, []int16{1, 2}), ErrUnsupportedType},
		{"fewer values than the header says", valid[:len(valid)-1], ErrLength},
		{"more values than the header says", append(append([]byte{}, valid...), 9), ErrLength},
		{"dimension of size zero", encode(t, UnsignedByte, []uint32{0, 3}, []uint8{}), ErrHeader},
		{"header claims more values than allowed", encode(t, Float32, []uint32{1 << 16, 1 << 16}, []float32{}), ErrTooLarge},
		{"damaged gzip stream", compress(t, valid)[:12], ErrHeader},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Read(bytes.NewReader(tc.input))
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Read error = %v, want errors.Is(%v); got %+v", err, tc.wantErr, got)
			}
		})
	}
}

// TestReadFloat32SpecialValues checks that NaN and infinity survive the round trip,
// so that callers (not this reader) decide whether to reject them.
func TestReadFloat32SpecialValues(t *testing.T) {
	input := encode(t, Float32, []uint32{3}, []float32{float32(math.Inf(1)), float32(math.NaN()), 0})
	got, err := Read(bytes.NewReader(input))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !math.IsInf(float64(got.Floats[0]), 1) || !math.IsNaN(float64(got.Floats[1])) {
		t.Errorf("special values not preserved: %v", got.Floats)
	}
}

// TestReadFileWrittenByPython reads a file produced by python/mnist_io.py, checking
// that the Go reader and the Python writer agree on the format.
func TestReadFileWrittenByPython(t *testing.T) {
	file, err := os.Open("../inference/testdata/two_layer_inputs-idx2-float.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	got, err := Read(file)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.Type != Float32 || !reflect.DeepEqual(got.Shape, []int{5, 8}) || len(got.Floats) != 40 {
		t.Errorf("got type %#x, shape %v, %d values; want float32 [5 8], 40 values", got.Type, got.Shape, len(got.Floats))
	}
}

func TestRows(t *testing.T) {
	matrix := Array{Type: Float32, Shape: []int{2, 3}, Floats: []float32{1, 2, 3, 4, 5, 6}}
	got, err := matrix.Rows()
	if err != nil {
		t.Fatalf("Rows: %v", err)
	}
	if want := [][]float32{{1, 2, 3}, {4, 5, 6}}; !reflect.DeepEqual(got, want) {
		t.Errorf("Rows = %v, want %v", got, want)
	}

	for _, notMatrix := range []Array{
		{Type: Float32, Shape: []int{6}, Floats: []float32{1, 2, 3, 4, 5, 6}},
		{Type: UnsignedByte, Shape: []int{2, 3}, Bytes: []uint8{1, 2, 3, 4, 5, 6}},
	} {
		if _, err := notMatrix.Rows(); !errors.Is(err, ErrNotMatrix) {
			t.Errorf("Rows on type %#x shape %v: error = %v, want ErrNotMatrix", notMatrix.Type, notMatrix.Shape, err)
		}
	}
}
