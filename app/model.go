package main

import (
	"fmt"
)

// matrix is a 2-D tensor viewed as rows of cols numbers, stored row after row.
type matrix struct {
	rows, cols int
	data       []float32
}

// row returns row r's numbers.
func (m matrix) row(r int) []float32 {
	return m.data[r*m.cols : (r+1)*m.cols]
}

// weights gives checked access to a model file's tensors by name.
type weights map[string]tensor

func (w weights) get(name string) (tensor, error) {
	t, ok := w[name]
	if !ok {
		return tensor{}, fmt.Errorf("model file has no tensor %q", name)
	}
	return t, nil
}

// matrix returns the 2-D tensor called name.
func (w weights) matrix(name string) (matrix, error) {
	t, err := w.get(name)
	if err != nil {
		return matrix{}, err
	}
	if len(t.Shape) != 2 {
		return matrix{}, fmt.Errorf("tensor %q has shape %v, want 2 dimensions", name, t.Shape)
	}
	return matrix{t.Shape[0], t.Shape[1], t.Values}, nil
}

// gpt2 is a whole model: embeddings, n_layer blocks, and a final layer norm.
type gpt2 struct {
	cfg    config
	w      weights
	wte    matrix
	blocks []block
	lnF    layerNorm
}

func loadModel(w weights, cfg config) (*gpt2, error) {
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	if wte.cols != cfg.NEmbd {
		return nil, fmt.Errorf("wte.weight rows have %d numbers, want n_embd = %d", wte.cols, cfg.NEmbd)
	}
	m := &gpt2{cfg: cfg, w: w, wte: wte}
	for layer := 0; layer < cfg.NLayer; layer++ {
		b, err := loadBlock(w, cfg, layer)
		if err != nil {
			return nil, err
		}
		m.blocks = append(m.blocks, b)
	}
	if m.lnF, err = loadLayerNorm(w, "ln_f", cfg.NEmbd); err != nil {
		return nil, err
	}
	return m, nil
}

// sequence is a run of tokens fed through the model so far, with every
// layer's keys and values kept so that later tokens can reuse them.
type sequence struct {
	m      *gpt2
	caches []kvCache // one per layer
	length int       // how many positions have been fed
}

func (m *gpt2) newSequence() *sequence {
	return &sequence{m: m, caches: make([]kvCache, len(m.blocks))}
}

// feed embeds the IDs at the positions after those already fed, and runs
// them through every block in layer order, returning the final rows.
func (s *sequence) feed(ids []int) ([][]float64, error) {
	rows, err := embed(s.m.w, ids, s.length)
	if err != nil {
		return nil, err
	}
	for layer, b := range s.m.blocks {
		rows = b.forwardCached(rows, &s.caches[layer])
	}
	s.length += len(ids)
	return rows, nil
}

// forward embeds the IDs and runs every block on the rows, in layer order.
func (m *gpt2) forward(ids []int) ([][]float64, error) {
	return m.newSequence().feed(ids)
}

// logits scores every token as the one after row: the dot product of the
// layer-normed row with that token's row of wte.weight. GPT-2 reuses its
// embedding table here; there is no separate output table.
func (m *gpt2) logits(row []float64) []float64 {
	last := m.lnF.apply(row)
	scores := make([]float64, m.wte.rows)
	for t := range scores {
		sum := 0.0
		for j, w := range m.wte.row(t) {
			sum += last[j] * float64(w)
		}
		scores[t] = sum
	}
	return scores
}

// nextLogits scores every token as the one after the whole sequence of ids.
func (m *gpt2) nextLogits(ids []int) ([]float64, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("there are no token IDs to continue")
	}
	rows, err := m.forward(ids)
	if err != nil {
		return nil, err
	}
	return m.logits(rows[len(rows)-1]), nil
}

// embed turns each token ID into a row of numbers: row id of wte.weight plus
// row position of wpe.weight. The first ID sits at position start.
func embed(w weights, ids []int, start int) ([][]float64, error) {
	wte, err := w.matrix("wte.weight")
	if err != nil {
		return nil, err
	}
	wpe, err := w.matrix("wpe.weight")
	if err != nil {
		return nil, err
	}
	if wte.cols != wpe.cols {
		return nil, fmt.Errorf("wte.weight rows have %d numbers but wpe.weight rows have %d", wte.cols, wpe.cols)
	}
	rows := make([][]float64, len(ids))
	for i, id := range ids {
		pos := start + i
		if id < 0 || id >= wte.rows {
			return nil, fmt.Errorf("token ID %d is outside the %d-token vocabulary", id, wte.rows)
		}
		if pos >= wpe.rows {
			return nil, fmt.Errorf("position %d is past the model's %d positions", pos, wpe.rows)
		}
		tok, posRow := wte.row(id), wpe.row(pos)
		row := make([]float64, wte.cols)
		for j := range row {
			row[j] = float64(tok[j]) + float64(posRow[j])
		}
		rows[i] = row
	}
	return rows, nil
}
