package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type inspectEntry struct {
	Name  string    `json:"name"`
	DType string    `json:"dtype"`
	Shape []int     `json:"shape"`
	First []float64 `json:"first"`
}

func inspectFile(t *testing.T, entries []stEntry, metadata map[string]string) []inspectEntry {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensors(t, path, entries, metadata)
	return decodeJSON[[]inspectEntry](t, runCommand(t, "", "inspect", path))
}

func TestInspectStageExample(t *testing.T) {
	got := inspectFile(t, []stEntry{f32Entry("scale", []int{4}, 1, 0.5, 0.25, 0.125)}, nil)
	want := []inspectEntry{{"scale", "F32", []int{4}, []float64{1, 0.5, 0.25, 0.125}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestInspectPrintsOnlyFirstFourNumbers(t *testing.T) {
	got := inspectFile(t, []stEntry{f32Entry("w", []int{2, 3}, 1, 2, 3, 4, 5, 6)}, nil)
	want := []inspectEntry{{"w", "F32", []int{2, 3}, []float64{1, 2, 3, 4}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestInspectSortsByNameAndPrintsShortTensorsWhole(t *testing.T) {
	got := inspectFile(t, []stEntry{
		f32Entry("zeta", []int{2}, 3, 4),
		f32Entry("h.10.w", []int{1}, 9),
		f32Entry("alpha", []int{3}, -1, 0, 1),
		f32Entry("h.2.w", []int{1}, 8),
	}, map[string]string{"format": "pt"})
	want := []inspectEntry{
		{"alpha", "F32", []int{3}, []float64{-1, 0, 1}},
		{"h.10.w", "F32", []int{1}, []float64{9}},
		{"h.2.w", "F32", []int{1}, []float64{8}},
		{"zeta", "F32", []int{2}, []float64{3, 4}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestInspectI32KeepsExactIntegers(t *testing.T) {
	// 16777217 = 2^24 + 1 is the first integer float32 can't hold.
	got := inspectFile(t, []stEntry{
		i32Entry("ids", []int{5}, 7, -3, 16777217, math.MaxInt32, 0),
		f32Entry("bias", []int{2}, 0.1, -2.5),
	}, nil)
	want := []inspectEntry{
		{"bias", "F32", []int{2}, []float64{float64(float32(0.1)), -2.5}},
		{"ids", "I32", []int{5}, []float64{7, -3, 16777217, math.MaxInt32}},
	}
	if len(got) != 2 || got[1].DType != "I32" || !reflect.DeepEqual(got[1], want[1]) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	// float32 values print in their shortest form, which reads back as the same float32.
	if got[0].Name != "bias" || float32(got[0].First[0]) != float32(0.1) || got[0].First[1] != -2.5 {
		t.Fatalf("got %+v, want %+v", got[0], want[0])
	}
}

func TestInspectOutputIsOnlyJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensors(t, path, []stEntry{f32Entry("scale", []int{4}, 1, 0.5, 0.25, 0.125)}, nil)
	out := runCommand(t, "", "inspect", path)
	if want := `[{"name":"scale","dtype":"F32","shape":[4],"first":[1,0.5,0.25,0.125]}]` + "\n"; out != want {
		t.Fatalf("got %q, want %q", out, want)
	}
}

func TestInspectScalarAndEmptyTensors(t *testing.T) {
	got := inspectFile(t, []stEntry{
		f32Entry("scalar", []int{}, 42),
		f32Entry("empty", []int{0, 3}),
	}, nil)
	want := []inspectEntry{
		{"empty", "F32", []int{0, 3}, []float64{}},
		{"scalar", "F32", []int{}, []float64{42}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestLoadTensorsOtherDTypes(t *testing.T) {
	f16 := make([]byte, 8)
	for i, bits := range []uint16{0x3c00, 0xc000, 0x0001, 0x7c00} { // 1, -2, smallest subnormal, +Inf
		binary.LittleEndian.PutUint16(f16[2*i:], bits)
	}
	bf16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(bf16, 0x3fc0)     // 1.5
	binary.LittleEndian.PutUint16(bf16[2:], 0xc120) // -10
	f64 := make([]byte, 8)
	binary.LittleEndian.PutUint64(f64, math.Float64bits(0.75))
	i64 := make([]byte, 8)
	binary.LittleEndian.PutUint64(i64, uint64(1<<40))

	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensors(t, path, []stEntry{
		{"half", "F16", []int{4}, f16},
		{"brain", "BF16", []int{2}, bf16},
		{"double", "F64", []int{1}, f64},
		{"long", "I64", []int{1}, i64},
		{"bytes", "U8", []int{2}, []byte{255, 1}},
		{"flags", "BOOL", []int{2}, []byte{1, 0}},
	}, nil)
	tensors, err := loadTensors(path)
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, want []float32) {
		t.Helper()
		if got := tensors[name].Values; !reflect.DeepEqual(got, want) {
			t.Errorf("%s: got %v, want %v", name, got, want)
		}
	}
	check("half", []float32{1, -2, float32(math.Ldexp(1, -24)), float32(math.Inf(1))})
	check("brain", []float32{1.5, -10})
	check("double", []float32{0.75})
	check("long", []float32{1 << 40})
	check("bytes", []float32{255, 1})
	check("flags", []float32{1, 0})
	if got := tensors["long"].Ints; !reflect.DeepEqual(got, []int64{1 << 40}) {
		t.Errorf("long ints: got %v", got)
	}
	if tensors["half"].Ints != nil {
		t.Errorf("float tensor should have no Ints")
	}
}

func TestLoadTensorsReadsLargeTensorsInChunks(t *testing.T) {
	// More than one read chunk, so the chunk boundaries get exercised.
	n := readChunkBytes/4*2 + 3
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i) * 0.5
	}
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensors(t, path, []stEntry{
		f32Entry("small", []int{1}, -1),
		f32Entry("big", []int{n}, values...),
	}, nil)
	tensors, err := loadTensors(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tensors["big"].Values, values) {
		t.Fatal("big tensor values differ")
	}
	if !reflect.DeepEqual(tensors["small"].Values, []float32{-1}) {
		t.Fatalf("small: got %v", tensors["small"].Values)
	}
}

func TestLoadTensorsRejectsBrokenFiles(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.safetensors")
	writeSafetensors(t, good, []stEntry{f32Entry("w", []int{4}, 1, 2, 3, 4)}, nil)
	contents, err := os.ReadFile(good)
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string][]byte{
		"empty":            {},
		"truncated data":   contents[:len(contents)-1],
		"huge header":      append(binary.LittleEndian.AppendUint64(nil, 1<<40), contents[8:]...),
		"not json":         append(binary.LittleEndian.AppendUint64(nil, 8), []byte("nonsense")...),
		"shape mismatch":   []byte(strings.Replace(string(contents), `"shape":[4]`, `"shape":[5]`, 1)),
		"unsupported type": []byte(strings.Replace(string(contents), `"F32"`, `"Q42"`, 1)),
	}
	for name, data := range cases {
		path := filepath.Join(dir, strings.ReplaceAll(name, " ", "_"))
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadTensors(path); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := loadTensors(filepath.Join(dir, "missing")); err == nil {
		t.Error("missing file: expected an error")
	}
	var out bytes.Buffer
	if err := run([]string{"inspect", filepath.Join(dir, "missing")}, strings.NewReader(""), &out); err == nil || out.Len() != 0 {
		t.Errorf("inspect of a missing file: err %v, output %q", err, out.String())
	}
}

func TestRunRejectsUnknownCommands(t *testing.T) {
	var out bytes.Buffer
	if err := run(nil, strings.NewReader(""), &out); err == nil {
		t.Error("expected a usage error with no command")
	}
	if err := run([]string{"bogus"}, strings.NewReader(""), &out); err == nil {
		t.Error("expected an error for an unknown command")
	}
	if err := run([]string{"inspect"}, strings.NewReader(""), &out); err == nil {
		t.Error("expected a usage error for inspect without a file")
	}
}
