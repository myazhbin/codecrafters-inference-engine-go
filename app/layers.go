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

// layerNormEpsilon is the 0.00001 from GPT-2's definition of layer norm.
const layerNormEpsilon = 1e-5

// layerNorm shifts and scales a row so its numbers average 0 with a standard
// deviation of about 1, then applies a weight and a bias of its own.
type layerNorm struct {
	weight, bias []float32
}

func loadLayerNorm(w weights, prefix string, n int) (layerNorm, error) {
	weight, err := w.get(prefix + ".weight")
	if err != nil {
		return layerNorm{}, err
	}
	bias, err := w.get(prefix + ".bias")
	if err != nil {
		return layerNorm{}, err
	}
	if len(weight.Values) != n || len(bias.Values) != n {
		return layerNorm{}, fmt.Errorf("%s has %d weights and %d biases, want %d of each", prefix, len(weight.Values), len(bias.Values), n)
	}
	return layerNorm{weight.Values, bias.Values}, nil
}

// apply returns (x - mean) / sqrt(variance + 0.00001) * weight + bias, where
// the variance is the average of (x - mean)^2, dividing by the count of
// numbers rather than one less.
func (ln layerNorm) apply(x []float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	variance := 0.0
	for _, v := range x {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(x))
	inv := 1 / math.Sqrt(variance+layerNormEpsilon)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = (v-mean)*inv*float64(ln.weight[i]) + float64(ln.bias[i])
	}
	return out
}

// attentionLayer holds one layer's attention weights.
type attentionLayer struct {
	nEmbd int
	nHead int
	cAttn linear // row → query, key and value, side by side
	cProj linear // the heads' results, side by side → output row
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
	return attentionLayer{cfg.NEmbd, cfg.NHead, cAttn, cProj}, nil
}

// forward runs causal multi-head self-attention. Each query, key and value
// is cut into nHead equal slices, and each head turns its own slices into
// weights over this position and every earlier one. The heads' weighted
// averages of the values go side by side, in head order, into c_proj.
func (a attentionLayer) forward(rows [][]float64) [][]float64 {
	n := a.nEmbd
	queries := make([][]float64, len(rows))
	keys := make([][]float64, len(rows))
	values := make([][]float64, len(rows))
	for i, row := range rows {
		qkv := a.cAttn.apply(row)
		queries[i], keys[i], values[i] = qkv[:n], qkv[n:2*n], qkv[2*n:]
	}

	headSize := n / a.nHead
	scale := 1 / math.Sqrt(float64(headSize))
	out := make([][]float64, len(rows))
	for i := range rows {
		mixed := make([]float64, n)
		scores := make([]float64, i+1)
		for h := 0; h < a.nHead; h++ {
			lo, hi := h*headSize, (h+1)*headSize
			// Score this position against itself and every earlier one,
			// using only this head's slices.
			q := queries[i][lo:hi]
			for j := 0; j <= i; j++ {
				scores[j] = dot(q, keys[j][lo:hi]) * scale
			}
			softmaxInPlace(scores)
			headOut := mixed[lo:hi]
			for j, weight := range scores {
				for d, v := range values[j][lo:hi] {
					headOut[d] += weight * v
				}
			}
		}
		out[i] = a.cProj.apply(mixed)
	}
	return out
}

// block is one transformer layer: attention on a layer-normed copy of each
// row, added back onto the row it came from.
//
//	row ─┬─► ln_1 ─► attention ─► + ─► output
//	     └────────────────────────┘
type block struct {
	ln1  layerNorm
	attn attentionLayer
}

func loadBlock(w weights, cfg config, layer int) (block, error) {
	ln1, err := loadLayerNorm(w, fmt.Sprintf("h.%d.ln_1", layer), cfg.NEmbd)
	if err != nil {
		return block{}, err
	}
	attn, err := loadAttention(w, cfg, layer)
	if err != nil {
		return block{}, err
	}
	return block{ln1, attn}, nil
}

func (b block) forward(rows [][]float64) [][]float64 {
	normed := make([][]float64, len(rows))
	for i, row := range rows {
		normed[i] = b.ln1.apply(row)
	}
	attended := b.attn.forward(normed)
	out := make([][]float64, len(rows))
	for i, row := range rows {
		out[i] = add(row, attended[i])
	}
	return out
}

// add returns a + b, number by number.
func add(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
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
