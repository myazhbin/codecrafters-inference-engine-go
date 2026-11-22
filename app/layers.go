package main

import (
	"fmt"
	"math"
)

// linear is a weight of shape [in, out] and a bias of out numbers. GPT-2
// stores its weights this way round, so a row times the weight is a plain
// matrix multiplication with no transpose.
type linear struct {
	w matrix
	b []float32
}

// loadLinear reads prefix.weight and prefix.bias, checking that the weight
// takes in numbers and, when out isn't -1, gives out numbers.
func loadLinear(w weights, prefix string, in, out int) (linear, error) {
	weight, err := w.matrix(prefix + ".weight")
	if err != nil {
		return linear{}, err
	}
	bias, err := w.get(prefix + ".bias")
	if err != nil {
		return linear{}, err
	}
	if weight.rows != in || (out != -1 && weight.cols != out) {
		return linear{}, fmt.Errorf("%s.weight has shape [%d, %d], want [%d, %d]", prefix, weight.rows, weight.cols, in, out)
	}
	if len(bias.Values) != weight.cols {
		return linear{}, fmt.Errorf("%s.bias has %d numbers, want %d", prefix, len(bias.Values), weight.cols)
	}
	return linear{weight, bias.Values}, nil
}

// apply returns x × w + b. Number j is the dot product of x with column j of
// w, plus b[j]. The loop runs along w's rows, which sit next to each other in
// memory, and adds up in float64.
func (l linear) apply(x []float64) []float64 {
	out := make([]float64, l.w.cols)
	for j, b := range l.b {
		out[j] = float64(b)
	}
	for i, xi := range x {
		row := l.w.row(i)
		for j, wij := range row {
			out[j] += xi * float64(wij)
		}
	}
	return out
}

// softmaxInPlace turns scores into weights that add up to 1: e raised to
// each score, divided by their total. Subtracting the largest score first
// keeps exp from overflowing and gives the same weights.
func softmaxInPlace(scores []float64) {
	if len(scores) == 0 {
		return
	}
	largest := scores[0]
	for _, s := range scores[1:] {
		largest = max(largest, s)
	}
	total := 0.0
	for i, s := range scores {
		scores[i] = math.Exp(s - largest)
		total += scores[i]
	}
	for i := range scores {
		scores[i] /= total
	}
}

// attentionLayer holds one layer's attention weights.
type attentionLayer struct {
	nEmbd int
	cAttn linear // row → query, key and value, side by side
	cProj linear // weighted values → output row
}

func loadAttention(w weights, cfg config, layer int) (attentionLayer, error) {
	prefix := fmt.Sprintf("h.%d.attn", layer)
	cAttn, err := loadLinear(w, prefix+".c_attn", cfg.NEmbd, 3*cfg.NEmbd)
	if err != nil {
		return attentionLayer{}, err
	}
	cProj, err := loadLinear(w, prefix+".c_proj", cfg.NEmbd, cfg.NEmbd)
	if err != nil {
		return attentionLayer{}, err
	}
	return attentionLayer{cfg.NEmbd, cAttn, cProj}, nil
}

// forward runs causal self-attention: each output row is a weighted average
// of the values of its own position and every earlier one, projected by
// c_proj.
func (a attentionLayer) forward(rows [][]float64) [][]float64 {
	n := a.nEmbd
	queries := make([][]float64, len(rows))
	keys := make([][]float64, len(rows))
	values := make([][]float64, len(rows))
	for i, row := range rows {
		qkv := a.cAttn.apply(row)
		queries[i], keys[i], values[i] = qkv[:n], qkv[n:2*n], qkv[2*n:]
	}

	scale := 1 / math.Sqrt(float64(n))
	out := make([][]float64, len(rows))
	for i := range rows {
		// Score this position against itself and every earlier one.
		scores := make([]float64, i+1)
		for j := 0; j <= i; j++ {
			scores[j] = dot(queries[i], keys[j]) * scale
		}
		softmaxInPlace(scores)
		mixed := make([]float64, n)
		for j, weight := range scores {
			for d, v := range values[j] {
				mixed[d] += weight * v
			}
		}
		out[i] = a.cProj.apply(mixed)
	}
	return out
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
