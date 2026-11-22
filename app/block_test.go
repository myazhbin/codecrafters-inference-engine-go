package main

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

// refLayerNorm follows the stage's formula, with the variance dividing by
// the count of numbers.
func refLayerNorm(x []float64, weight, bias []float32) []float64 {
	n := float64(len(x))
	mean := 0.0
	for _, v := range x {
		mean += v / n
	}
	variance := 0.0
	for _, v := range x {
		variance += (v - mean) * (v - mean) / n
	}
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = (v-mean)/math.Sqrt(variance+0.00001)*float64(weight[i]) + float64(bias[i])
	}
	return out
}

// refResidualBlock is the block of the "Residual connections" stage: the
// input rows plus attention over their ln_1-normed copies.
func refResidualBlock(m *testModel, layer int, x [][]float64) [][]float64 {
	p := fmt.Sprintf("h.%d.", layer)
	var normed [][]float64
	for _, row := range x {
		normed = append(normed, refLayerNorm(row, m.values[p+"ln_1.weight"], m.values[p+"ln_1.bias"]))
	}
	attended := refAttention(m, layer, normed)
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			out[i][j] = x[i][j] + attended[i][j]
		}
	}
	return out
}

// zeroMLP makes a layer's MLP add nothing, as in the "Residual connections"
// stage's files.
func zeroMLP(m *testModel, layer int) {
	n := m.cfg.NEmbd
	p := fmt.Sprintf("h.%d.mlp.", layer)
	m.set(p+"c_fc.weight", make([]float32, n*4*n)...)
	m.set(p+"c_fc.bias", make([]float32, 4*n)...)
	m.set(p+"c_proj.weight", make([]float32, 4*n*n)...)
	m.set(p+"c_proj.bias", make([]float32, n)...)
}

// copyingBlockFile writes the stages' 2-number block whose attention copies
// each row into its query, key and value, with layer norms of weight 1 and
// bias 0, and an MLP that is either all zeros or keeps the wide row's first
// two numbers and puts them back.
func copyingBlockFile(t *testing.T, dir string, keepingMLP bool) string {
	model := filepath.Join(dir, "model.safetensors")
	fc := make([]float32, 2*8)
	proj := make([]float32, 8*2)
	if keepingMLP {
		fc[0*8+0], fc[1*8+1] = 1, 1     // wide[0] = x[0], wide[1] = x[1]
		proj[0*2+0], proj[1*2+1] = 1, 1 // out[0] = wide[0], out[1] = wide[1]
	}
	writeSafetensors(t, model, []stEntry{
		f32Entry("h.0.ln_1.weight", []int{2}, 1, 1),
		f32Entry("h.0.ln_1.bias", []int{2}, 0, 0),
		f32Entry("h.0.attn.c_attn.weight", []int{2, 6}, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1),
		f32Entry("h.0.attn.c_attn.bias", []int{6}, 0, 0, 0, 0, 0, 0),
		f32Entry("h.0.attn.c_proj.weight", []int{2, 2}, 1, 0, 0, 1),
		f32Entry("h.0.attn.c_proj.bias", []int{2}, 0, 0),
		f32Entry("h.0.ln_2.weight", []int{2}, 1, 1),
		f32Entry("h.0.ln_2.bias", []int{2}, 0, 0),
		f32Entry("h.0.mlp.c_fc.weight", []int{2, 8}, fc...),
		f32Entry("h.0.mlp.c_fc.bias", []int{8}, make([]float32, 8)...),
		f32Entry("h.0.mlp.c_proj.weight", []int{8, 2}, proj...),
		f32Entry("h.0.mlp.c_proj.bias", []int{2}, 0, 0),
	}, nil)
	return model
}

func TestBlockResidualStageExample(t *testing.T) {
	dir := t.TempDir()
	model := copyingBlockFile(t, dir, false)
	config := writeFile(t, dir, "config.json", `{ "n_embd": 2, "n_head": 1, "n_layer": 2, "n_positions": 4, "vocab_size": 2 }`)
	input := writeFile(t, dir, "input.json", "[[1, 2]]")

	got := decodeJSON[[][]float64](t, runCommand(t, "", "block", model, config, "0", input))
	assertClose(t, got, [][]float64{{0, 3}}, 1e-4)
	// Exactly: one row attends only to itself, so attention returns the
	// normed row, [-0.5, 0.5] / sqrt(0.25 + 0.00001).
	normed := 0.5 / math.Sqrt(0.25+0.00001)
	assertClose(t, got, [][]float64{{1 - normed, 2 + normed}}, 1e-12)
}

func TestLayerNorm(t *testing.T) {
	ones := []float32{1, 1, 1, 1}
	zeros := []float32{0, 0, 0, 0}
	ln := layerNorm{ones, zeros}
	small := ln.apply([]float64{2, 4, 6, 8})
	large := ln.apply([]float64{20, 40, 60, 80})
	want := [][]float64{{-1.34, -0.45, 0.45, 1.34}}
	assertClose(t, [][]float64{small}, want, 0.005)
	assertClose(t, [][]float64{large}, want, 0.005)

	// The variance divides by the count: [1, 2] has variance 0.25, not 0.5.
	got := layerNorm{[]float32{1, 1}, []float32{0, 0}}.apply([]float64{1, 2})
	assertClose(t, [][]float64{got}, [][]float64{{-0.5 / math.Sqrt(0.25+1e-5), 0.5 / math.Sqrt(0.25+1e-5)}}, 1e-15)

	// The weight scales and the bias shifts each number.
	got = layerNorm{[]float32{2, 3}, []float32{10, -10}}.apply([]float64{1, 2})
	n := 0.5 / math.Sqrt(0.25+1e-5)
	assertClose(t, [][]float64{got}, [][]float64{{-2*n + 10, 3*n - 10}}, 1e-12)

	// A row of equal numbers has variance 0, and the epsilon keeps it finite.
	got = ln.apply([]float64{5, 5, 5, 5})
	assertClose(t, [][]float64{got}, [][]float64{{0, 0, 0, 0}}, 0)
}

func TestBlockRandomResidualModels(t *testing.T) {
	for _, tc := range []struct{ layers, layer int }{{1, 0}, {2, 0}, {2, 1}} {
		t.Run(fmt.Sprintf("layer %d of %d", tc.layer, tc.layers), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(500 + 10*tc.layers + tc.layer)))
			m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: tc.layers, NPositions: 8, VocabSize: 6})
			for l := 0; l < tc.layers; l++ {
				zeroMLP(m, l)
			}
			dir := t.TempDir()
			model, config := m.write(t, dir)
			rows := randomRows(rng, 3, 4)
			input := writeFile(t, dir, "input.json", rowsJSON(t, rows))

			got := decodeJSON[[][]float64](t, runCommand(t, "", "block", model, config, fmt.Sprint(tc.layer), input))
			assertClose(t, got, refResidualBlock(m, tc.layer, rows), 1e-9)
		})
	}
}

func TestBlockNeedsLayerNormTensors(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	tensors := tensorsOf(m)
	delete(tensors, "h.0.ln_1.bias")
	if _, err := loadBlock(tensors, m.cfg, 0); err == nil {
		t.Error("expected an error for a missing ln_1.bias")
	}
}
