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
			input := writeFile(t, dir, "input.json", toJSON(t, rows))

			got := decodeJSON[[][]float64](t, runCommand(t, "", "block", model, config, fmt.Sprint(tc.layer), input))
			assertClose(t, got, refResidualBlock(m, tc.layer, rows), 1e-9)
		})
	}
}

// refGelu is the stage's GELU formula.
func refGelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// refBlock is the full transformer block: the residual attention block,
// then the input plus the MLP of its ln_2-normed copy.
func refBlock(m *testModel, layer int, x [][]float64) [][]float64 {
	p := fmt.Sprintf("h.%d.", layer)
	rows := refResidualBlock(m, layer, x)
	out := make([][]float64, len(rows))
	for i, row := range rows {
		normed := refLayerNorm(row, m.values[p+"ln_2.weight"], m.values[p+"ln_2.bias"])
		wide := refLinear(normed, m.values[p+"mlp.c_fc.weight"], m.values[p+"mlp.c_fc.bias"])
		for j := range wide {
			wide[j] = refGelu(wide[j])
		}
		narrow := refLinear(wide, m.values[p+"mlp.c_proj.weight"], m.values[p+"mlp.c_proj.bias"])
		out[i] = make([]float64, len(row))
		for j := range row {
			out[i][j] = row[j] + narrow[j]
		}
	}
	return out
}

func TestGelu(t *testing.T) {
	for _, tc := range []struct{ x, want float64 }{
		{0, 0},
		{1, 0.8411919906082768},   // the exact erf version gives 0.8413447...
		{-1, -0.1588080093917232}, // the exact erf version gives -0.1586552...
		{3, 2.996362607918227},    // the exact erf version gives 2.9959503...
		{-10, -0},
	} {
		if got := gelu(tc.x); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("gelu(%v) = %v, want %v", tc.x, got, tc.want)
		}
		if got := gelu(tc.x); math.Abs(got-refGelu(tc.x)) > 1e-15 {
			t.Errorf("gelu(%v) = %v, reference says %v", tc.x, got, refGelu(tc.x))
		}
	}
}

func TestBlockTransformerStageExample(t *testing.T) {
	dir := t.TempDir()
	model := copyingBlockFile(t, dir, true)
	config := writeFile(t, dir, "config.json", `{ "n_embd": 2, "n_head": 1, "n_layer": 2, "n_positions": 4, "vocab_size": 2 }`)
	input := writeFile(t, dir, "input.json", "[[1, 2]]")

	got := decodeJSON[[][]float64](t, runCommand(t, "", "block", model, config, "0", input))
	assertClose(t, got, [][]float64{{-0.15878819, 3.84116958}}, 1e-6)
}

func TestBlockRandomFullModels(t *testing.T) {
	for _, tc := range []struct{ embd, heads, layers, layer int }{
		{4, 2, 1, 0}, {4, 2, 2, 0}, {4, 2, 2, 1}, {8, 4, 3, 2},
	} {
		t.Run(fmt.Sprintf("%d heads, layer %d of %d", tc.heads, tc.layer, tc.layers), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(600 + tc.embd*10 + tc.layer)))
			m := newRandomModel(rng, config{NEmbd: tc.embd, NHead: tc.heads, NLayer: tc.layers, NPositions: 8, VocabSize: 6})
			dir := t.TempDir()
			model, config := m.write(t, dir)
			rows := randomRows(rng, 3, tc.embd)
			input := writeFile(t, dir, "input.json", toJSON(t, rows))

			got := decodeJSON[[][]float64](t, runCommand(t, "", "block", model, config, fmt.Sprint(tc.layer), input))
			assertClose(t, got, refBlock(m, tc.layer, rows), 1e-9)
		})
	}
}

func TestBlockReadsMLPWidthFromShape(t *testing.T) {
	// An MLP 3 times as wide as the row instead of GPT-2's 4.
	rng := rand.New(rand.NewSource(77))
	cfg := config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6}
	m := newRandomModel(rng, cfg)
	for i, e := range m.entries {
		switch e.Name {
		case "h.0.mlp.c_fc.weight":
			m.entries[i] = f32Entry(e.Name, []int{4, 12}, randomValues(rng, 48)...)
		case "h.0.mlp.c_fc.bias":
			m.entries[i] = f32Entry(e.Name, []int{12}, randomValues(rng, 12)...)
		case "h.0.mlp.c_proj.weight":
			m.entries[i] = f32Entry(e.Name, []int{12, 4}, randomValues(rng, 48)...)
		default:
			continue
		}
		m.values[e.Name] = tensorsOfEntry(m.entries[i])
	}
	b, err := loadBlock(tensorsOf(m), cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	rows := randomRows(rng, 2, 4)
	assertClose(t, b.forward(rows), refBlock(m, 0, rows), 1e-9)
}

// tensorsOfEntry decodes an F32 entry's numbers.
func tensorsOfEntry(e stEntry) []float32 {
	values := make([]float32, len(e.Data)/4)
	for i := range values {
		values[i] = math.Float32frombits(uint32(e.Data[4*i]) | uint32(e.Data[4*i+1])<<8 | uint32(e.Data[4*i+2])<<16 | uint32(e.Data[4*i+3])<<24)
	}
	return values
}

func TestBlockNeedsLayerNormTensors(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	for _, name := range []string{"h.0.ln_1.bias", "h.0.ln_2.weight", "h.0.mlp.c_fc.bias", "h.0.mlp.c_proj.weight"} {
		tensors := tensorsOf(m)
		delete(tensors, name)
		if _, err := loadBlock(tensors, m.cfg, 0); err == nil {
			t.Errorf("expected an error for a missing %s", name)
		}
	}
	// An MLP projection that doesn't come back to n_embd is an error too.
	tensors := tensorsOf(m)
	proj := tensors["h.0.mlp.c_proj.weight"]
	proj.Shape = []int{8, 8}
	tensors["h.0.mlp.c_proj.weight"] = proj
	if _, err := loadBlock(tensors, m.cfg, 0); err == nil {
		t.Error("expected an error for a c_proj of the wrong shape")
	}
}
