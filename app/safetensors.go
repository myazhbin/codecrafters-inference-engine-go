package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// tensor is one table of numbers from a safetensors file.
type tensor struct {
	DType  string
	Shape  []int
	Values []float32
	// Ints holds the exact values of an integer tensor (I8…I64, U8…U64,
	// BOOL), since float32 can't hold every integer past 2^24 exactly.
	// It is nil for float tensors.
	Ints []int64
}

// dtypeSizes maps every dtype the loader understands to its size in bytes.
var dtypeSizes = map[string]int{
	"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
	"I64": 8, "I32": 4, "I16": 2, "I8": 1,
	"U64": 8, "U32": 4, "U16": 2, "U8": 1,
	"BOOL": 1,
}

func isIntegerDType(dtype string) bool {
	switch dtype {
	case "F64", "F32", "F16", "BF16":
		return false
	}
	return true
}

// tensorInfo is one entry of a safetensors header.
type tensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// maxHeaderLen guards against a corrupt length making us allocate gigabytes.
const maxHeaderLen = 100 << 20

// readChunkBytes is how much of a tensor is read from disk at once. Reading
// in chunks straight into the tensor's []float32 means a 548 MB model needs
// about 548 MB of memory, not twice that.
const readChunkBytes = 1 << 20

// loadTensors reads every tensor in a safetensors file, with all of its numbers.
//
// The format is an 8-byte little-endian header length, a JSON header mapping
// each tensor name to its dtype, shape and byte range, then the raw data.
func loadTensors(path string) (map[string]tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := stat.Size()

	var lenBytes [8]byte
	if _, err := io.ReadFull(f, lenBytes[:]); err != nil {
		return nil, fmt.Errorf("reading header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBytes[:])
	if headerLen > maxHeaderLen || int64(headerLen) > fileSize-8 {
		return nil, fmt.Errorf("header length %d doesn't fit in a %d-byte file", headerLen, fileSize)
	}
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("parsing header: %w", err)
	}

	dataStart := 8 + int64(headerLen)
	dataLen := fileSize - dataStart
	tensors := make(map[string]tensor, len(header))
	var scratch []byte
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info tensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("tensor %q: parsing header entry: %w", name, err)
		}
		t, err := readTensor(f, dataStart, dataLen, info, &scratch)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		tensors[name] = t
	}
	return tensors, nil
}

// readTensor reads one tensor's numbers from the data section of the file.
func readTensor(r io.ReaderAt, dataStart, dataLen int64, info tensorInfo, scratch *[]byte) (tensor, error) {
	size, ok := dtypeSizes[info.DType]
	if !ok {
		return tensor{}, fmt.Errorf("unsupported dtype %q", info.DType)
	}
	shape := info.Shape
	if shape == nil {
		shape = []int{} // a scalar
	}
	count := 1
	for _, dim := range shape {
		if dim < 0 {
			return tensor{}, fmt.Errorf("negative dimension in shape %v", shape)
		}
		count *= dim
	}
	begin, end := info.DataOffsets[0], info.DataOffsets[1]
	if begin < 0 || end < begin || end > dataLen {
		return tensor{}, fmt.Errorf("data offsets [%d, %d] are outside the %d-byte data section", begin, end, dataLen)
	}
	if end-begin != int64(count)*int64(size) {
		return tensor{}, fmt.Errorf("%d bytes of data, but shape %v of %s needs %d", end-begin, shape, info.DType, count*size)
	}

	t := tensor{DType: info.DType, Shape: shape, Values: make([]float32, count)}
	if isIntegerDType(info.DType) {
		t.Ints = make([]int64, count)
	}
	chunk := max(1, readChunkBytes/size)
	for i := 0; i < count; i += chunk {
		n := min(chunk, count-i)
		if cap(*scratch) < n*size {
			*scratch = make([]byte, n*size)
		}
		buf := (*scratch)[:n*size]
		if _, err := r.ReadAt(buf, dataStart+begin+int64(i)*int64(size)); err != nil {
			return tensor{}, fmt.Errorf("reading data: %w", err)
		}
		var ints []int64
		if t.Ints != nil {
			ints = t.Ints[i : i+n]
		}
		decode(info.DType, buf, t.Values[i:i+n], ints)
	}
	return t, nil
}

// decode converts little-endian raw values of the given dtype into values,
// and for integer dtypes also into ints.
func decode(dtype string, raw []byte, values []float32, ints []int64) {
	le := binary.LittleEndian
	switch dtype {
	case "F32":
		for k := range values {
			values[k] = math.Float32frombits(le.Uint32(raw[4*k:]))
		}
	case "F64":
		for k := range values {
			values[k] = float32(math.Float64frombits(le.Uint64(raw[8*k:])))
		}
	case "F16":
		for k := range values {
			values[k] = float16ToFloat32(le.Uint16(raw[2*k:]))
		}
	case "BF16":
		for k := range values {
			values[k] = math.Float32frombits(uint32(le.Uint16(raw[2*k:])) << 16)
		}
	default:
		for k := range ints {
			var v int64
			switch dtype {
			case "I64":
				v = int64(le.Uint64(raw[8*k:]))
			case "I32":
				v = int64(int32(le.Uint32(raw[4*k:])))
			case "I16":
				v = int64(int16(le.Uint16(raw[2*k:])))
			case "I8":
				v = int64(int8(raw[k]))
			case "U64":
				v = int64(le.Uint64(raw[8*k:]))
			case "U32":
				v = int64(le.Uint32(raw[4*k:]))
			case "U16":
				v = int64(le.Uint16(raw[2*k:]))
			case "U8", "BOOL":
				v = int64(raw[k])
			}
			ints[k] = v
			values[k] = float32(v)
		}
	}
}

// float16ToFloat32 widens an IEEE 754 half-precision float.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0x1f: // infinity or NaN
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	case exp != 0: // normal
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	case mant == 0: // zero
		return math.Float32frombits(sign)
	}
	// Subnormal: shift the mantissa up until it is normal.
	e := uint32(127 - 15 + 1)
	for mant&0x400 == 0 {
		mant <<= 1
		e--
	}
	return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
}
