package main

import (
	"bytes"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

// refGreedy follows the stage's pseudocode, recomputing the whole sequence
// with the reference pieces at every step.
func refGreedy(m *testModel, ids []int, count int) []int {
	seq := append([]int(nil), ids...)
	newIDs := []int{}
	for i := 0; i < count; i++ {
		logits := refLogits(m, seq)
		best := 0
		for tok := range logits {
			if logits[tok] > logits[best] {
				best = tok
			}
		}
		seq = append(seq, best)
		newIDs = append(newIDs, best)
	}
	return newIDs
}

// generateFiles writes a model and its inputs, returning the four paths.
func generateFiles(t *testing.T, m *testModel, ids []int, tokensJSON string) []string {
	t.Helper()
	dir := t.TempDir()
	model, config := m.write(t, dir)
	return []string{model, config, writeFile(t, dir, "ids.json", toJSON(t, ids)), writeFile(t, dir, "tokens.json", tokensJSON)}
}

func runGenerate(t *testing.T, stdin string, files []string) []int {
	t.Helper()
	return decodeJSON[[]int](t, runCommand(t, stdin, append([]string{"generate"}, files...)...))
}

func TestGenerateGreedyStageExample(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	m := newRandomModel(rng, config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2})
	m.set("wte.weight", 1, 0, 0, 1)
	m.set("ln_f.weight", 0, 0)
	m.set("ln_f.bias", 0, 1)

	got := runGenerate(t, "", generateFiles(t, m, []int{0}, `{ "count": 2 }`))
	if want := []int{1, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestGenerateGreedyTiesPickTheLowerID(t *testing.T) {
	// With ln_f's weight and bias all zeros, every token scores exactly 0.
	rng := rand.New(rand.NewSource(2))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	m.set("ln_f.weight", 0, 0, 0, 0)
	m.set("ln_f.bias", 0, 0, 0, 0)

	got := runGenerate(t, "", generateFiles(t, m, []int{3, 5}, `{"count": 3}`))
	if want := []int{0, 0, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if got := argmax([]float64{1, 3, 3, 2}); got != 1 {
		t.Errorf("argmax of a tie picked %d, want 1", got)
	}
}

func TestGenerateGreedyRandomModels(t *testing.T) {
	varied := false
	for seed := int64(0); seed < 8; seed++ {
		rng := rand.New(rand.NewSource(800 + seed))
		m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 16, VocabSize: 6})
		ids := []int{rng.Intn(6), rng.Intn(6), rng.Intn(6)}

		got := runGenerate(t, "", generateFiles(t, m, ids, `{"count": 4}`))
		want := refGreedy(m, ids, 4)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("seed %d, ids %v: got %v, want %v", seed, ids, got, want)
		}
		for _, id := range got {
			varied = varied || id != got[0]
		}
	}
	if !varied {
		t.Error("every run wrote one ID over and over; the test isn't checking much")
	}
}

func TestGenerateCountZeroPrintsEmptyArray(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	files := generateFiles(t, m, []int{1}, `{"count": 0}`)
	if out := runCommand(t, "", append([]string{"generate"}, files...)...); out != "[]\n" {
		t.Fatalf("got %q, want []", out)
	}
}

func TestGenerateFillsTheLastPosition(t *testing.T) {
	// 3 prompt IDs and 4 positions: the second new token is never fed back
	// in, so it needs no fifth position.
	rng := rand.New(rand.NewSource(5))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 6})
	got := runGenerate(t, "", generateFiles(t, m, []int{1, 2, 3}, `{"count": 2}`))
	if want := refGreedy(m, []int{1, 2, 3}, 2); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestGenerateRejectsBadInput(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 4, VocabSize: 6})
	for name, tc := range map[string]struct {
		ids    []int
		tokens string
	}{
		// The third new token is scored from 5 rows, but there are 4 positions.
		"past n_positions": {[]int{1, 2, 3}, `{"count": 3}`},
		"no count":         {[]int{1}, `{}`},
		"negative count":   {[]int{1}, `{"count": -1}`},
		"no ids":           {[]int{}, `{"count": 1}`},
	} {
		files := generateFiles(t, m, tc.ids, tc.tokens)
		var out bytes.Buffer
		if err := run(append([]string{"generate"}, files...), strings.NewReader(""), &out); err == nil {
			t.Errorf("%s: expected an error, got %q", name, out.String())
		}
	}
}
