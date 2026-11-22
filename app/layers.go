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

// kvCache keeps one layer's key and value for every position so far, in
// order. Head h's key and value are slice h of each, so these rows hold the
// lists for every head of the layer at once.
type kvCache struct {
	keys, values [][]float64
}

// forward runs causal multi-head self-attention on rows that start at
// position 0.
func (a attentionLayer) forward(rows [][]float64) [][]float64 {
	return a.forwardCached(rows, &kvCache{})
}

// forwardCached runs causal multi-head self-attention on rows that come
// right after the positions already in cache, adding their keys and values
// to it.
//
// Each query, key and value is cut into nHead equal slices, and each head
// turns its own slices into weights over the row's position and every
// earlier one. The heads' weighted averages of the values go side by side,
// in head order, into c_proj. Rows are taken left to right, each adding its
// key and value before scoring, so a row only ever sees itself and earlier
// positions.
func (a attentionLayer) forwardCached(rows [][]float64, cache *kvCache) [][]float64 {
	n := a.nEmbd
	headSize := n / a.nHead
	scale := 1 / math.Sqrt(float64(headSize))
	out := make([][]float64, len(rows))
	for i, row := range rows {
		qkv := a.cAttn.apply(row)
		query := qkv[:n]
		cache.keys = append(cache.keys, qkv[n:2*n])
		cache.values = append(cache.values, qkv[2*n:])

		positions := len(cache.keys)
		scores := make([]float64, positions)
		mixed := make([]float64, n)
		for h := 0; h < a.nHead; h++ {
			lo, hi := h*headSize, (h+1)*headSize
			q := query[lo:hi]
			for j, key := range cache.keys {
				scores[j] = dot(q, key[lo:hi]) * scale
			}
			softmaxInPlace(scores)
			headOut := mixed[lo:hi]
			for j, weight := range scores {
				for d, v := range cache.values[j][lo:hi] {
					headOut[d] += weight * v
				}
			}
		}
		out[i] = a.cProj.apply(mixed)
	}
	return out
}

// gelu keeps positive numbers near themselves and pushes negative ones
// toward 0. This is GPT-2's tanh approximation, not the exact erf version;
// the two differ by up to about 0.0005.
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

// mlp widens a row, applies GELU to every number, and narrows it back.
type mlp struct {
	cFc   linear // n_embd → hidden, 4 * n_embd in GPT-2
	cProj linear // hidden → n_embd
}

func loadMLP(w weights, cfg config, layer int) (mlp, error) {
	prefix := fmt.Sprintf("h.%d.mlp", layer)
	// The hidden width comes from c_fc.weight's shape.
	cFc, err := loadLinear(w, prefix+".c_fc", cfg.NEmbd, -1)
	if err != nil {
		return mlp{}, err
	}
	cProj, err := loadLinear(w, prefix+".c_proj", cFc.w.cols, cfg.NEmbd)
	if err != nil {
		return mlp{}, err
	}
	return mlp{cFc, cProj}, nil
}

func (m mlp) forward(row []float64) []float64 {
	hidden := m.cFc.apply(row)
	for i, v := range hidden {
		hidden[i] = gelu(v)
	}
	return m.cProj.apply(hidden)
}

// block is one transformer layer. Attention mixes rows together and the MLP
// works on each row alone; each runs on a layer-normed copy of the rows, and
// its output is added back onto the rows the copy came from.
//
//	row ─┬─► ln_1 ─► attention ─► + ─┬─► ln_2 ─► MLP ─► + ─► output
//	     └────────────────────────┘  └──────────────────┘
type block struct {
	ln1  layerNorm
	attn attentionLayer
	ln2  layerNorm
	mlp  mlp
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
	ln2, err := loadLayerNorm(w, fmt.Sprintf("h.%d.ln_2", layer), cfg.NEmbd)
	if err != nil {
		return block{}, err
	}
	m, err := loadMLP(w, cfg, layer)
	if err != nil {
		return block{}, err
	}
	return block{ln1, attn, ln2, m}, nil
}

// forward runs the block on rows that start at position 0.
func (b block) forward(rows [][]float64) [][]float64 {
	return b.forwardCached(rows, &kvCache{})
}

// forwardCached runs the block on rows that come right after the positions
// already in this layer's cache. Only attention looks at other positions;
// the layer norms and the MLP work on one row at a time.
func (b block) forwardCached(rows [][]float64, cache *kvCache) [][]float64 {
	normed := make([][]float64, len(rows))
	for i, row := range rows {
		normed[i] = b.ln1.apply(row)
	}
	attended := b.attn.forwardCached(normed, cache)
	out := make([][]float64, len(rows))
	for i, row := range rows {
		row = add(row, attended[i])
		out[i] = add(row, b.mlp.forward(b.ln2.apply(row)))
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
