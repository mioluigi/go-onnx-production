// Package idx reads the IDX file format, in which the MNIST handwritten-digit
// data set is published (LeCun, Cortes, and Burges, "The MNIST Database").
//
// An IDX file is a 4-byte header (two zero bytes, a data-type code, the number of
// dimensions), then each dimension's size as a big-endian uint32, then the values
// in row-major order, big-endian. Files may be gzip-compressed; Read detects that.
//
// The package supports the two types this project needs: unsigned bytes (images,
// labels) and 32-bit floats (model outputs). It does no file handling of its own,
// so it can read from files, embedded data, or network streams alike.
package idx

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// DataType is the IDX code for the type of the stored values.
type DataType byte

// Supported data types.
const (
	UnsignedByte DataType = 0x08
	Float32      DataType = 0x0D
)

// MaxValues limits how many values a header may announce, so a damaged or hostile
// header cannot make Read allocate gigabytes. MNIST's largest file holds 47 million.
const MaxValues = 1 << 28

// Sentinel errors, for errors.Is.
var (
	ErrHeader          = errors.New("not a valid IDX header")
	ErrUnsupportedType = errors.New("unsupported IDX data type")
	ErrTooLarge        = errors.New("IDX file announces too many values")
	ErrLength          = errors.New("IDX data length does not match its header")
	ErrNotMatrix       = errors.New("IDX array is not a 2-D float32 matrix")
)

// gzipMagic is the first two bytes of every gzip stream.
var gzipMagic = []byte{0x1f, 0x8b}

// Array holds one IDX file's contents. Exactly one of Bytes or Floats is set,
// according to Type.
type Array struct {
	Type   DataType
	Shape  []int
	Bytes  []uint8
	Floats []float32
}

// Read parses an IDX stream, decompressing it first if it is gzip-compressed.
func Read(r io.Reader) (Array, error) {
	source, err := decompressIfGzip(r)
	if err != nil {
		return Array{}, err
	}
	dataType, shape, err := readHeader(source)
	if err != nil {
		return Array{}, err
	}
	count, err := valueCount(shape)
	if err != nil {
		return Array{}, err
	}
	return readValues(source, dataType, shape, count)
}

// Rows returns a 2-D float32 array as one slice per row, sharing its memory.
func (a Array) Rows() ([][]float32, error) {
	if a.Type != Float32 || len(a.Shape) != 2 {
		return nil, fmt.Errorf("%w: type %#x, shape %v", ErrNotMatrix, byte(a.Type), a.Shape)
	}
	rows, columns := a.Shape[0], a.Shape[1]
	matrix := make([][]float32, rows)
	for row := range matrix {
		matrix[row] = a.Floats[row*columns : (row+1)*columns : (row+1)*columns]
	}
	return matrix, nil
}

// decompressIfGzip peeks at the first bytes and wraps the stream in a gzip
// reader when they are the gzip signature.
func decompressIfGzip(r io.Reader) (io.Reader, error) {
	buffered := bufio.NewReader(r)
	signature, _ := buffered.Peek(len(gzipMagic))
	if !bytes.Equal(signature, gzipMagic) {
		return buffered, nil
	}
	decompressed, err := gzip.NewReader(buffered)
	if err != nil {
		return nil, fmt.Errorf("%w: damaged gzip stream: %v", ErrHeader, err)
	}
	return decompressed, nil
}

// readHeader reads the magic bytes, type code, and dimension sizes.
func readHeader(r io.Reader) (DataType, []int, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return 0, nil, fmt.Errorf("%w: reading the first 4 bytes: %v", ErrHeader, err)
	}
	if magic[0] != 0 || magic[1] != 0 || magic[3] == 0 {
		return 0, nil, fmt.Errorf("%w: first bytes are % x", ErrHeader, magic)
	}
	dataType := DataType(magic[2])
	if dataType != UnsignedByte && dataType != Float32 {
		return 0, nil, fmt.Errorf("%w: code %#x (supported: %#x unsigned byte, %#x float32)",
			ErrUnsupportedType, byte(dataType), byte(UnsignedByte), byte(Float32))
	}

	sizes := make([]uint32, magic[3])
	if err := binary.Read(r, binary.BigEndian, sizes); err != nil {
		return 0, nil, fmt.Errorf("%w: reading %d dimension sizes: %v", ErrHeader, len(sizes), err)
	}
	shape := make([]int, len(sizes))
	for i, size := range sizes {
		if size == 0 {
			return 0, nil, fmt.Errorf("%w: dimension %d has size 0", ErrHeader, i)
		}
		shape[i] = int(size)
	}
	return dataType, shape, nil
}

// valueCount multiplies the dimensions, refusing totals above MaxValues.
func valueCount(shape []int) (int, error) {
	count := 1
	for _, size := range shape {
		if count > math.MaxInt/size || count*size > MaxValues {
			return 0, fmt.Errorf("%w: shape %v exceeds %d values", ErrTooLarge, shape, MaxValues)
		}
		count *= size
	}
	return count, nil
}

// readValues reads exactly count values and checks that nothing follows them.
func readValues(r io.Reader, dataType DataType, shape []int, count int) (Array, error) {
	array := Array{Type: dataType, Shape: shape}
	var target any
	if dataType == UnsignedByte {
		array.Bytes = make([]uint8, count)
		target = array.Bytes
	} else {
		array.Floats = make([]float32, count)
		target = array.Floats
	}
	if err := binary.Read(r, binary.BigEndian, target); err != nil {
		return Array{}, fmt.Errorf("%w: header shape %v needs %d values: %v", ErrLength, shape, count, err)
	}
	if extra, _ := io.Copy(io.Discard, r); extra > 0 {
		return Array{}, fmt.Errorf("%w: %d unexpected bytes after %d values", ErrLength, extra, count)
	}
	return array, nil
}
