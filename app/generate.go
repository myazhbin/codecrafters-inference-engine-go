package main

import (
	"fmt"
)

// generateOptions is tokens.json: how many new tokens to write.
type generateOptions struct {
	Count *int `json:"count"`
}

func readGenerateOptions(path string) (generateOptions, error) {
	var opts generateOptions
	if err := readJSONFile(path, &opts); err != nil {
		return generateOptions{}, err
	}
	if opts.Count == nil || *opts.Count < 0 {
		return generateOptions{}, fmt.Errorf("%s needs a count of 0 or more", path)
	}
	return opts, nil
}

// argmax returns the index of the highest score, the lower index on a tie.
func argmax(scores []float64) int {
	best := 0
	for i, s := range scores {
		if s > scores[best] {
			best = i
		}
	}
	return best
}

// generate writes count new token IDs after ids. Each step scores every
// token as the next one, takes the highest score, and appends it, so the
// next step scores the longer sequence.
func (m *gpt2) generate(ids []int, count int) ([]int, error) {
	seq := append([]int(nil), ids...)
	newIDs := make([]int, 0, count)
	for len(newIDs) < count {
		scores, err := m.nextLogits(seq)
		if err != nil {
			return nil, err
		}
		next := argmax(scores)
		seq = append(seq, next)
		newIDs = append(newIDs, next)
	}
	return newIDs, nil
}
