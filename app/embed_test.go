package main

import (
	"bytes"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbedStageExample(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, model, []stEntry{
		f32Entry("wte.weight", []int{2, 2}, 1, 2, 3, 4),
		f32Entry("wpe.weight", []int{2, 2}, 10, 20, 30, 40),
	}, nil)
	ids := writeFile(t, dir, "ids.json", "[1, 0]")

	got := decodeJSON[[][]float64](t, runCommand(t, "", "embed", model, ids))
	assertClose(t, got, [][]float64{{13, 24}, {31, 42}}, 0)
}

// randomValues returns n random float32s between -1 and 1.
func randomValues(rng *rand.Rand, n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = rng.Float32()*2 - 1
	}
	return values
}

func TestEmbedRandomTablesWithUnsortedRepeatedIDs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const vocab, positions, width = 6, 8, 4
	wte := randomValues(rng, vocab*width)
	wpe := randomValues(rng, positions*width)

	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, model, []stEntry{
		f32Entry("wpe.weight", []int{positions, width}, wpe...),
		f32Entry("wte.weight", []int{vocab, width}, wte...),
	}, nil)
	idList := []int{5, 2, 2, 0, 5, 3}
	ids := writeFile(t, dir, "ids.json", "[5, 2, 2, 0, 5, 3]")

	want := make([][]float64, len(idList))
	for pos, id := range idList {
		want[pos] = make([]float64, width)
		for j := 0; j < width; j++ {
			want[pos][j] = float64(wte[id*width+j]) + float64(wpe[pos*width+j])
		}
	}
	got := decodeJSON[[][]float64](t, runCommand(t, "", "embed", model, ids))
	assertClose(t, got, want, 1e-12)

	// A repeated ID at a different position gets a different embedding.
	if got[1][0] == got[2][0] && got[1][1] == got[2][1] {
		t.Errorf("token 2 at positions 1 and 2 should differ: %v vs %v", got[1], got[2])
	}
}

func TestEmbedStartsAtAGivenPosition(t *testing.T) {
	w := weights{
		"wte.weight": {DType: "F32", Shape: []int{2, 2}, Values: []float32{1, 2, 3, 4}},
		"wpe.weight": {DType: "F32", Shape: []int{3, 2}, Values: []float32{10, 20, 30, 40, 50, 60}},
	}
	got, err := embed(w, []int{0, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, [][]float64{{31, 42}, {53, 64}}, 0)
}

func TestEmbedRejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, model, []stEntry{
		f32Entry("wte.weight", []int{2, 2}, 1, 2, 3, 4),
		f32Entry("wpe.weight", []int{2, 2}, 10, 20, 30, 40),
	}, nil)
	cases := map[string]string{
		"id past the vocabulary": "[2]",
		"negative id":            "[-1]",
		"fractional id":          "[0.5]",
		"too many positions":     "[0, 0, 0]",
		"not an array":           `{"ids": [0]}`,
	}
	for name, ids := range cases {
		path := writeFile(t, dir, "ids.json", ids)
		var out bytes.Buffer
		if err := run([]string{"embed", model, path}, strings.NewReader(""), &out); err == nil {
			t.Errorf("%s: expected an error, got output %q", name, out.String())
		}
	}

	// Whole numbers written as floats are still IDs.
	path := writeFile(t, dir, "ids.json", "[1.0, 0]")
	got := decodeJSON[[][]float64](t, runCommand(t, "", "embed", model, path))
	assertClose(t, got, [][]float64{{13, 24}, {31, 42}}, 0)
}
