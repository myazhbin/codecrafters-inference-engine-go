package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// stEntry is one tensor to write into a test safetensors file.
type stEntry struct {
	Name  string
	DType string
	Shape []int
	Data  []byte
}

func f32Entry(name string, shape []int, values ...float32) stEntry {
	data := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	return stEntry{name, "F32", shape, data}
}

func i32Entry(name string, shape []int, values ...int32) stEntry {
	data := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[4*i:], uint32(v))
	}
	return stEntry{name, "I32", shape, data}
}

// writeSafetensors writes the entries to path, keeping their order in both
// the header and the data, so tests control whether names come sorted. The
// header is padded with spaces to 8 bytes, as the real format does.
func writeSafetensors(t testing.TB, path string, entries []stEntry, metadata map[string]string) {
	t.Helper()
	var header strings.Builder
	header.WriteString("{")
	if metadata != nil {
		meta, err := json.Marshal(metadata)
		if err != nil {
			t.Fatal(err)
		}
		header.WriteString(`"__metadata__":` + string(meta))
	}
	var data bytes.Buffer
	for i, e := range entries {
		if i > 0 || metadata != nil {
			header.WriteString(",")
		}
		info, err := json.Marshal(map[string]any{
			"dtype":        e.DType,
			"shape":        e.Shape,
			"data_offsets": []int{data.Len(), data.Len() + len(e.Data)},
		})
		if err != nil {
			t.Fatal(err)
		}
		header.WriteString(strconv.Quote(e.Name) + ":" + string(info))
		data.Write(e.Data)
	}
	header.WriteString("}")
	headerBytes := []byte(header.String())
	for len(headerBytes)%8 != 0 {
		headerBytes = append(headerBytes, ' ')
	}

	var file bytes.Buffer
	binary.Write(&file, binary.LittleEndian, uint64(len(headerBytes)))
	file.Write(headerBytes)
	file.Write(data.Bytes())
	if err := os.WriteFile(path, file.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeFile writes contents to name inside dir and returns its path.
func writeFile(t testing.TB, dir, name, contents string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// runCommand runs the program with args and stdin, failing the test on error,
// and returns what it printed.
func runCommand(t testing.TB, stdin string, args ...string) string {
	t.Helper()
	var out bytes.Buffer
	if err := run(args, strings.NewReader(stdin), &out); err != nil {
		t.Fatalf("run %v: %v", args, err)
	}
	return out.String()
}

// decodeJSON parses the program's output, failing the test if it isn't JSON.
func decodeJSON[T any](t testing.TB, output string) T {
	t.Helper()
	var v T
	if err := json.Unmarshal([]byte(output), &v); err != nil {
		t.Fatalf("output is not the expected JSON: %v\n%s", err, output)
	}
	return v
}

// assertClose fails unless got and want have the same shape and every number
// is within tol.
func assertClose(t testing.TB, got, want [][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d\ngot  %v\nwant %v", len(got), len(want), got, want)
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("row %d has %d numbers, want %d\ngot  %v\nwant %v", i, len(got[i]), len(want[i]), got, want)
		}
		for j := range want[i] {
			if math.Abs(got[i][j]-want[i][j]) > tol {
				t.Fatalf("row %d, number %d: got %v, want %v (tolerance %v)\ngot  %v\nwant %v",
					i, j, got[i][j], want[i][j], tol, got, want)
			}
		}
	}
}
