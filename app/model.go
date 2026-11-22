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
