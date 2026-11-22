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

// refAttention is a direct transcription of the stage's single-head steps.
func refAttention(m *testModel, layer int, x [][]float64) [][]float64 {
	p := fmt.Sprintf("h.%d.attn.", layer)
	n := m.cfg.NEmbd
	var q, k, v [][]float64
	for _, row := range x {
		qkv := refLinear(row, m.values[p+"c_attn.weight"], m.values[p+"c_attn.bias"])
		q, k, v = append(q, qkv[:n]), append(k, qkv[n:2*n]), append(v, qkv[2*n:])
	}
	var out [][]float64
	for i := range x {
		var exps []float64
		total := 0.0
		for j := 0; j <= i; j++ {
			score := 0.0
			for d := 0; d < n; d++ {
				score += q[i][d] * k[j][d]
			}
			e := math.Exp(score / math.Sqrt(float64(n)))
			exps = append(exps, e)
			total += e
		}
		sum := make([]float64, n)
		for j, e := range exps {
			for d := 0; d < n; d++ {
				sum[d] += e / total * v[j][d]
			}
		}
		out = append(out, refLinear(sum, m.values[p+"c_proj.weight"], m.values[p+"c_proj.bias"]))
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

func TestAttnIsCausal(t *testing.T) {
	// Changing a later row must not change any earlier row's output.
	rng := rand.New(rand.NewSource(7))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 1, NLayer: 1, NPositions: 8, VocabSize: 6})
	attn, err := loadAttention(weights(tensorsOf(m)), m.cfg, 0)
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
