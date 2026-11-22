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
