package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
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

// testModel is a GPT-2-shaped model with random weights, kept both as the
// entries to write and as plain slices for reference implementations.
type testModel struct {
	cfg     config
	entries []stEntry
	values  map[string][]float32
}

func (m *testModel) add(name string, shape []int, values []float32) {
	m.entries = append(m.entries, f32Entry(name, shape, values...))
	m.values[name] = values
}

// set replaces a tensor's numbers in both the entries and the values.
func (m *testModel) set(name string, values ...float32) {
	for i, e := range m.entries {
		if e.Name == name {
			m.entries[i] = f32Entry(name, e.Shape, values...)
			m.values[name] = values
			return
		}
	}
	panic("no tensor " + name)
}

// newRandomModel builds every tensor GPT-2 has, sized by cfg, with random
// numbers between -1 and 1.
func newRandomModel(rng *rand.Rand, cfg config) *testModel {
	m := &testModel{cfg: cfg, values: map[string][]float32{}}
	n := cfg.NEmbd
	random := func(name string, shape ...int) {
		count := 1
		for _, d := range shape {
			count *= d
		}
		m.add(name, shape, randomValues(rng, count))
	}
	random("wte.weight", cfg.VocabSize, n)
	random("wpe.weight", cfg.NPositions, n)
	for l := 0; l < cfg.NLayer; l++ {
		p := fmt.Sprintf("h.%d.", l)
		random(p+"ln_1.weight", n)
		random(p+"ln_1.bias", n)
		random(p+"attn.c_attn.weight", n, 3*n)
		random(p+"attn.c_attn.bias", 3*n)
		random(p+"attn.c_proj.weight", n, n)
		random(p+"attn.c_proj.bias", n)
		random(p+"ln_2.weight", n)
		random(p+"ln_2.bias", n)
		random(p+"mlp.c_fc.weight", n, 4*n)
		random(p+"mlp.c_fc.bias", 4*n)
		random(p+"mlp.c_proj.weight", 4*n, n)
		random(p+"mlp.c_proj.bias", n)
	}
	random("ln_f.weight", n)
	random("ln_f.bias", n)
	return m
}

// write saves the model and its config into dir, returning their paths.
func (m *testModel) write(t testing.TB, dir string) (modelPath, configPath string) {
	t.Helper()
	modelPath = filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, modelPath, m.entries, map[string]string{"format": "pt"})
	cfg, err := json.Marshal(m.cfg)
	if err != nil {
		t.Fatal(err)
	}
	return modelPath, writeFile(t, dir, "config.json", string(cfg))
}

// rowsJSON encodes rows as JSON, for input.json.
func rowsJSON(t testing.TB, rows [][]float64) string {
	t.Helper()
	data, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// randomRows returns count rows of width random numbers between -1 and 1.
func randomRows(rng *rand.Rand, count, width int) [][]float64 {
	rows := make([][]float64, count)
	for i := range rows {
		rows[i] = make([]float64, width)
		for j := range rows[i] {
			rows[i][j] = rng.Float64()*2 - 1
		}
	}
	return rows
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
