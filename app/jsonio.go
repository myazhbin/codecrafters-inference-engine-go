package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// readJSONFile parses the JSON file at path into v.
func readJSONFile(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}
	return nil
}

// config holds a model's sizes, named as in GPT-2's config.json.
type config struct {
	NEmbd      int `json:"n_embd"`
	NHead      int `json:"n_head"`
	NLayer     int `json:"n_layer"`
	NPositions int `json:"n_positions"`
	VocabSize  int `json:"vocab_size"`
}

func readConfig(path string) (config, error) {
	var cfg config
	if err := readJSONFile(path, &cfg); err != nil {
		return config{}, err
	}
	if cfg.NEmbd <= 0 {
		return config{}, fmt.Errorf("%s: n_embd must be positive, got %d", path, cfg.NEmbd)
	}
	if cfg.NLayer < 0 {
		return config{}, fmt.Errorf("%s: n_layer can't be negative, got %d", path, cfg.NLayer)
	}
	return cfg, nil
}

// readRows reads a JSON array of rows of numbers, such as [[1, 2], [3, 4]].
func readRows(path string, width int) ([][]float64, error) {
	var rows [][]float64
	if err := readJSONFile(path, &rows); err != nil {
		return nil, err
	}
	for i, row := range rows {
		if len(row) != width {
			return nil, fmt.Errorf("%s: row %d has %d numbers, want n_embd = %d", path, i, len(row), width)
		}
	}
	if rows == nil {
		rows = [][]float64{}
	}
	return rows, nil
}

// readIDs reads a JSON array of token IDs, such as [464, 3139, 286].
func readIDs(path string) ([]int, error) {
	var raw []float64
	if err := readJSONFile(path, &raw); err != nil {
		return nil, err
	}
	ids := make([]int, len(raw))
	for i, v := range raw {
		if v != math.Trunc(v) || v < 0 || v > math.MaxInt32 {
			return nil, fmt.Errorf("%s: %v is not a token ID", path, v)
		}
		ids[i] = int(v)
	}
	return ids, nil
}
