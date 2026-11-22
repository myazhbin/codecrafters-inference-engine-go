package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"
)

// refLinear computes x × w + b the textbook way: number j is the dot product
// of x with column j of w, plus b[j].
func refLinear(x []float64, w, b []float32) []float64 {
	in, out := len(x), len(b)
	res := make([]float64, out)
	for j := 0; j < out; j++ {
		sum := float64(b[j])
		for i := 0; i < in; i++ {
			sum += x[i] * float64(w[i*out+j])
		}
		res[j] = sum
	}
	return res
}

// refAttention is a direct transcription of the stages' attention steps,
// one head at a time.
func refAttention(m *testModel, layer int, x [][]float64) [][]float64 {
	p := fmt.Sprintf("h.%d.attn.", layer)
	n, heads := m.cfg.NEmbd, m.cfg.NHead
	hs := n / heads
	var q, k, v [][]float64
	for _, row := range x {
		qkv := refLinear(row, m.values[p+"c_attn.weight"], m.values[p+"c_attn.bias"])
		q, k, v = append(q, qkv[:n]), append(k, qkv[n:2*n]), append(v, qkv[2*n:])
	}
	var out [][]float64
	for i := range x {
		var sideBySide []float64
		for h := 0; h < heads; h++ {
			var exps []float64
			total := 0.0
			for j := 0; j <= i; j++ {
				score := 0.0
				for d := h * hs; d < (h+1)*hs; d++ {
					score += q[i][d] * k[j][d]
				}
				e := math.Exp(score / math.Sqrt(float64(hs)))
				exps = append(exps, e)
				total += e
			}
			sum := make([]float64, hs)
			for j, e := range exps {
				for d := 0; d < hs; d++ {
					sum[d] += e / total * v[j][h*hs+d]
				}
			}
			sideBySide = append(sideBySide, sum...)
		}
		out = append(out, refLinear(sideBySide, m.values[p+"c_proj.weight"], m.values[p+"c_proj.bias"]))
	}
	return out
}

func TestAttnStageExample(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, model, []stEntry{
		f32Entry("h.0.attn.c_attn.weight", []int{1, 3}, 1, 1, 1),
		f32Entry("h.0.attn.c_attn.bias", []int{3}, 0, 0, 0),
		f32Entry("h.0.attn.c_proj.weight", []int{1, 1}, 1),
		f32Entry("h.0.attn.c_proj.bias", []int{1}, 0),
	}, nil)
	config := writeFile(t, dir, "config.json", `{ "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }`)
	input := writeFile(t, dir, "input.json", "[[1], [2]]")

	got := decodeJSON[[][]float64](t, runCommand(t, "", "attn", model, config, "0", input))
	assertClose(t, got, [][]float64{{1.0}, {1.8807970779778822}}, 1e-12)
}

func TestAttnRandomSingleHeadModels(t *testing.T) {
	for _, tc := range []struct{ layers, layer int }{{1, 0}, {2, 1}, {2, 0}} {
		t.Run(fmt.Sprintf("layer %d of %d", tc.layer, tc.layers), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(10*tc.layers + tc.layer)))
			m := newRandomModel(rng, config{NEmbd: 4, NHead: 1, NLayer: tc.layers, NPositions: 8, VocabSize: 6})
			dir := t.TempDir()
			model, config := m.write(t, dir)
			rows := randomRows(rng, 3, 4)
			input := writeFile(t, dir, "input.json", rowsJSON(t, rows))

			got := decodeJSON[[][]float64](t, runCommand(t, "", "attn", model, config, fmt.Sprint(tc.layer), input))
			assertClose(t, got, refAttention(m, tc.layer, rows), 1e-9)
		})
	}
}

func TestAttnMultiHeadStageExample(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	// The weights copy each row into its query, key and value.
	writeSafetensors(t, model, []stEntry{
		f32Entry("h.0.attn.c_attn.weight", []int{2, 6}, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1),
		f32Entry("h.0.attn.c_attn.bias", []int{6}, 0, 0, 0, 0, 0, 0),
		f32Entry("h.0.attn.c_proj.weight", []int{2, 2}, 1, 0, 0, 1),
		f32Entry("h.0.attn.c_proj.bias", []int{2}, 0, 0),
	}, nil)
	config := writeFile(t, dir, "config.json", `{ "n_embd": 2, "n_head": 2, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }`)
	input := writeFile(t, dir, "input.json", "[[1, 0], [0, 1]]")

	got := decodeJSON[[][]float64](t, runCommand(t, "", "attn", model, config, "0", input))
	assertClose(t, got, [][]float64{{1.0, 0.0}, {0.5, 0.7310585786300049}}, 1e-12)
}

func TestAttnRandomMultiHeadModels(t *testing.T) {
	for _, tc := range []struct{ embd, heads, layers, layer int }{
		{4, 2, 1, 0}, {4, 2, 2, 1}, {8, 4, 2, 1}, {6, 3, 1, 0}, {8, 8, 1, 0},
	} {
		t.Run(fmt.Sprintf("%d heads of %d, layer %d of %d", tc.heads, tc.embd, tc.layer, tc.layers), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.embd*100 + tc.heads*10 + tc.layer)))
			m := newRandomModel(rng, config{NEmbd: tc.embd, NHead: tc.heads, NLayer: tc.layers, NPositions: 8, VocabSize: 6})
			dir := t.TempDir()
			model, config := m.write(t, dir)
			rows := randomRows(rng, 4, tc.embd)
			input := writeFile(t, dir, "input.json", rowsJSON(t, rows))

			got := decodeJSON[[][]float64](t, runCommand(t, "", "attn", model, config, fmt.Sprint(tc.layer), input))
			assertClose(t, got, refAttention(m, tc.layer, rows), 1e-9)
		})
	}
}

func TestAttnHeadsChangeTheOutput(t *testing.T) {
	// The same weights read as one head or as two heads give different rows,
	// except at position 0, which only ever attends to itself.
	rng := rand.New(rand.NewSource(11))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 6})
	rows := randomRows(rng, 3, 4)
	one, err := loadAttention(tensorsOf(m), m.cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	two := one
	two.nHead = 2
	a, b := one.forward(rows), two.forward(rows)
	assertClose(t, b[:1], a[:1], 1e-12)
	if math.Abs(a[2][0]-b[2][0]) < 1e-9 {
		t.Errorf("one head and two heads gave the same row: %v", a[2])
	}
}

func TestConfigRejectsUnevenHeads(t *testing.T) {
	dir := t.TempDir()
	for _, cfg := range []string{
		`{"n_embd": 4, "n_head": 3, "n_layer": 1}`,
		`{"n_embd": 4, "n_head": 0, "n_layer": 1}`,
		`{"n_embd": 0, "n_head": 1, "n_layer": 1}`,
	} {
		if _, err := readConfig(writeFile(t, dir, "config.json", cfg)); err == nil {
			t.Errorf("%s: expected an error", cfg)
		}
	}
}

func TestAttnIsCausal(t *testing.T) {
	// Changing a later row must not change any earlier row's output.
	rng := rand.New(rand.NewSource(7))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	attn, err := loadAttention(tensorsOf(m), m.cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	rows := randomRows(rng, 4, 4)
	before := attn.forward(rows)
	rows[3] = []float64{9, -9, 9, -9}
	after := attn.forward(rows)
	assertClose(t, after[:3], before[:3], 0)
	if after[3][0] == before[3][0] {
		t.Error("the changed row's own output should change")
	}
}

func TestAttnRejectsBadInput(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 6})
	dir := t.TempDir()
	model, config := m.write(t, dir)
	goodInput := writeFile(t, dir, "input.json", "[[1, 2, 3, 4]]")
	badInput := writeFile(t, dir, "bad.json", "[[1, 2, 3]]")

	for name, args := range map[string][]string{
		"missing layer":   {"attn", model, config, "1", goodInput},
		"negative layer":  {"attn", model, config, "-1", goodInput},
		"layer not a num": {"attn", model, config, "x", goodInput},
		"short row":       {"attn", model, config, "0", badInput},
		"too few args":    {"attn", model, config, "0"},
	} {
		var out bytes.Buffer
		if err := run(args, strings.NewReader(""), &out); err == nil {
			t.Errorf("%s: expected an error, got output %q", name, out.String())
		}
	}
}

// tensorsOf turns a test model into the loader's map, without a file.
func tensorsOf(m *testModel) map[string]tensor {
	tensors := map[string]tensor{}
	for _, e := range m.entries {
		tensors[e.Name] = tensor{DType: "F32", Shape: e.Shape, Values: m.values[e.Name]}
	}
	return tensors
}
