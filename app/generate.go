package main

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
)

// generateOptions is tokens.json: how many new tokens to write and, when
// sampling, the temperature and top_p.
type generateOptions struct {
	Count       *int     `json:"count"`
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
}

func readGenerateOptions(path string) (generateOptions, error) {
	var opts generateOptions
	if err := readJSONFile(path, &opts); err != nil {
		return generateOptions{}, err
	}
	if opts.Count == nil || *opts.Count < 0 {
		return generateOptions{}, fmt.Errorf("%s needs a count of 0 or more", path)
	}
	if opts.Temperature != nil && *opts.Temperature < 0 {
		return generateOptions{}, fmt.Errorf("%s: temperature can't be negative", path)
	}
	return opts, nil
}

// tokenPicker chooses the next token from every token's score.
type tokenPicker func(scores []float64) (int, error)

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

// greedy always takes the highest score.
func greedy(scores []float64) (int, error) {
	return argmax(scores), nil
}

// newPicker returns the picker tokens.json asks for. Without a temperature
// it is greedy and never reads stdin. With one, every pick samples using the
// next number read from randoms.
func newPicker(opts generateOptions, randoms io.Reader) tokenPicker {
	if opts.Temperature == nil || *opts.Temperature == 0 {
		return greedy
	}
	temperature, topP := *opts.Temperature, 1.0
	if opts.TopP != nil {
		topP = *opts.TopP
	}
	numbers := bufio.NewScanner(randoms)
	numbers.Split(bufio.ScanWords)
	return func(scores []float64) (int, error) {
		if !numbers.Scan() {
			if err := numbers.Err(); err != nil {
				return 0, fmt.Errorf("reading a random number from stdin: %w", err)
			}
			return 0, fmt.Errorf("stdin ran out of random numbers")
		}
		r, err := strconv.ParseFloat(numbers.Text(), 64)
		if err != nil {
			return 0, fmt.Errorf("reading a random number from stdin: %w", err)
		}
		return sample(scores, temperature, topP, r), nil
	}
}

// sample picks a token at random, weighted by its probability, where r is a
// random number between 0 and 1:
//
//	p = softmax(each logit / temperature)
//	order = the token ids sorted by p, highest first, the lower id first on a tie
//	kept = the shortest start of order whose p adds up to at least top_p
//	divide each kept p by the sum of the kept p, so they add up to 1 again
//	walk kept in order, adding up p, and pick the first id whose total passes r
func sample(scores []float64, temperature, topP, r float64) int {
	p := make([]float64, len(scores))
	for i, s := range scores {
		p[i] = s / temperature
	}
	softmaxInPlace(p) // subtracts the largest logit first, so nothing underflows

	order := make([]int, len(p))
	for i := range order {
		order[i] = i
	}
	// A stable sort keeps the lower id first among equal probabilities.
	sort.SliceStable(order, func(a, b int) bool { return p[order[a]] > p[order[b]] })

	kept := 0
	if topP >= 1 {
		// Every token with a probability above 0. Rounding could make the
		// running total reach 1 early, or never quite reach it.
		for kept < len(order) && p[order[kept]] > 0 {
			kept++
		}
	} else {
		total := 0.0
		for kept < len(order) {
			total += p[order[kept]]
			kept++
			if total >= topP {
				break
			}
		}
	}
	kept = max(kept, 1)

	sum := 0.0
	for _, id := range order[:kept] {
		sum += p[id]
	}
	total := 0.0
	for _, id := range order[:kept] {
		total += p[id] / sum
		if total > r {
			return id
		}
	}
	// Rounding left the total a hair under r, which is below 1.
	return order[kept-1]
}

// generate writes count new token IDs after ids. Each step scores every
// token as the next one, lets pick choose one, and appends it, so the next
// step scores the longer sequence.
//
// Earlier tokens' keys and values never change, so the sequence keeps them:
// the prompt runs through the layers once, and then each new token runs
// alone at its position in the whole sequence. The last new token is never
// fed back in, since nothing is scored after it.
func (m *gpt2) generate(ids []int, count int, pick tokenPicker) ([]int, error) {
	return m.newSequence().generate(ids, count, pick)
}

// generate feeds ids into the sequence and then writes count new IDs after
// them, feeding in every new ID but the last.
func (seq *sequence) generate(ids []int, count int, pick tokenPicker) ([]int, error) {
	newIDs := make([]int, 0, count)
	if count == 0 {
		return newIDs, nil
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("there are no token IDs to continue")
	}
	m := seq.m
	rows, err := seq.feed(ids)
	if err != nil {
		return nil, err
	}
	last := rows[len(rows)-1]
	for {
		next, err := pick(m.logits(last))
		if err != nil {
			return nil, err
		}
		newIDs = append(newIDs, next)
		if len(newIDs) == count {
			return newIDs, nil
		}
		rows, err := seq.feed([]int{next})
		if err != nil {
			return nil, err
		}
		last = rows[0]
	}
}
