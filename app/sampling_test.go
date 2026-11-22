package main

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

// refSample transcribes the stage's pseudocode as plainly as possible.
func refSample(logits []float64, temperature, topP, r float64) int {
	n := len(logits)
	largest := math.Inf(-1)
	for _, l := range logits {
		largest = math.Max(largest, l)
	}
	p := make([]float64, n)
	sum := 0.0
	for i, l := range logits {
		p[i] = math.Exp((l - largest) / temperature)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	// Selection sort: highest p first, lower id first on a tie.
	var order []int
	used := make([]bool, n)
	for len(order) < n {
		best := -1
		for i := 0; i < n; i++ {
			if !used[i] && (best == -1 || p[i] > p[best]) {
				best = i
			}
		}
		used[best] = true
		order = append(order, best)
	}
	var kept []int
	acc := 0.0
	for _, id := range order {
		if topP >= 1 && p[id] == 0 {
			break
		}
		kept = append(kept, id)
		acc += p[id]
		if topP < 1 && acc >= topP {
			break
		}
	}
	keptSum := 0.0
	for _, id := range kept {
		keptSum += p[id]
	}
	total := 0.0
	for _, id := range kept {
		total += p[id] / keptSum
		if total > r {
			return id
		}
	}
	return kept[len(kept)-1]
}

func refGenerateSampled(m *testModel, ids []int, temperature, topP float64, randoms []float64) []int {
	seq := append([]int(nil), ids...)
	var newIDs []int
	for _, r := range randoms {
		next := refSample(refLogits(m, seq), temperature, topP, r)
		seq = append(seq, next)
		newIDs = append(newIDs, next)
	}
	return newIDs
}

func TestSampleStageExample(t *testing.T) {
	// Both tokens score the same, so each has probability 0.5. The running
	// totals are 0.5 and 1, and 0.6 is past 0.5 but not past 1.
	rng := rand.New(rand.NewSource(1))
	m := newRandomModel(rng, config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2})
	m.set("ln_f.weight", 0, 0)
	m.set("ln_f.bias", 0, 0)
	files := generateFiles(t, m, []int{0}, `{ "count": 1, "temperature": 1, "top_p": 1 }`)

	if got := runGenerate(t, "0.6\n", files); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("got %v, want [1]", got)
	}
	if got := runGenerate(t, "0.4\n", files); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("got %v, want [0]", got)
	}
}

func TestSampleRules(t *testing.T) {
	ln := func(ps ...float64) []float64 { // logits whose softmax is ps
		out := make([]float64, len(ps))
		for i, p := range ps {
			out[i] = math.Log(p)
		}
		return out
	}
	for _, tc := range []struct {
		name                 string
		logits               []float64
		temperature, topP, r float64
		want                 int
	}{
		{"equal scores, r in first half", []float64{0, 0}, 1, 1, 0.4, 0},
		{"equal scores, r in second half", []float64{0, 0}, 1, 1, 0.6, 1},
		{"a total must pass r, not reach it", []float64{0, 0}, 1, 1, 0.5, 1},
		{"highest probability comes first", []float64{0, 2, 1}, 1, 1, 0, 1},
		{"then the next highest", ln(0.2, 0.5, 0.3), 1, 1, 0.75, 2},
		{"then the lowest", ln(0.2, 0.5, 0.3), 1, 1, 0.85, 0},
		{"ties go to the lower id first", []float64{1, 3, 3, 2}, 1, 1, 0.01, 1},
		{"ties, second of the pair", []float64{1, 3, 3, 2}, 1, 1, 0.5, 2},
		// top_p 0.7 keeps 0.5 + 0.3, renormalized to 0.625 and 0.375.
		{"top_p cut, first kept", ln(0.5, 0.3, 0.2), 1, 0.7, 0.62, 0},
		{"top_p cut, second kept", ln(0.5, 0.3, 0.2), 1, 0.7, 0.63, 1},
		{"top_p cut never reaches the dropped token", ln(0.5, 0.3, 0.2), 1, 0.7, 0.999, 1},
		{"top_p reached exactly keeps the shortest start", ln(0.5, 0.25, 0.25), 1, 0.75, 0.9, 1},
		{"top_p 1 keeps every token", ln(0.5, 0.3, 0.2), 1, 1, 0.9, 2},
		{"tiny top_p keeps the top token", ln(0.5, 0.3, 0.2), 1, 0.01, 0.99, 0},
		// Temperature 0.5 squares the odds: 0.25 and 0.0625 out of 0.3125.
		{"low temperature sharpens", ln(0.5, 0.25), 0.5, 1, 0.79, 0},
		{"low temperature, second token", ln(0.5, 0.25), 0.5, 1, 0.81, 1},
		{"high temperature flattens", ln(0.8, 0.2), 1000, 1, 0.51, 1},
		{"huge negative logits don't underflow", []float64{-1000, -1001}, 1, 1, 0.73, 0},
		{"huge negative logits, second", []float64{-1000, -1001}, 1, 1, 0.74, 1},
		{"r just under 1 picks the last kept token", ln(0.1, 0.2, 0.3, 0.4), 1, 1, math.Nextafter(1, 0), 0},
	} {
		got := sample(tc.logits, tc.temperature, tc.topP, tc.r)
		if got != tc.want {
			t.Errorf("%s: got %d, want %d", tc.name, got, tc.want)
		}
		if ref := refSample(tc.logits, tc.temperature, tc.topP, tc.r); ref != tc.want {
			t.Errorf("%s: the reference says %d, want %d", tc.name, ref, tc.want)
		}
	}
}

func TestSampleMatchesReferenceOnRandomScores(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 2000; i++ {
		logits := make([]float64, 1+rng.Intn(12))
		for j := range logits {
			logits[j] = rng.NormFloat64() * 3
			if rng.Intn(4) == 0 && j > 0 {
				logits[j] = logits[j-1] // some exact ties
			}
		}
		temperature := []float64{0.3, 0.7, 1, 1.5}[rng.Intn(4)]
		topP := []float64{0.1, 0.5, 0.9, 1}[rng.Intn(4)]
		r := rng.Float64()
		if got, want := sample(logits, temperature, topP, r), refSample(logits, temperature, topP, r); got != want {
			t.Fatalf("sample(%v, T=%v, top_p=%v, r=%v) = %d, reference says %d", logits, temperature, topP, r, got, want)
		}
	}
}

func TestSampleRandomModels(t *testing.T) {
	for _, tc := range []struct {
		temperature, topP float64
		tokens            string
	}{
		{1, 0.6, `{"count": 4, "temperature": 1, "top_p": 0.6}`},
		{0.7, 1, `{"count": 4, "temperature": 0.7, "top_p": 1}`},
		{1.3, 0.9, `{"count": 4, "temperature": 1.3, "top_p": 0.9}`},
		{0.8, 1, `{"count": 4, "temperature": 0.8}`}, // top_p defaults to 1
	} {
		t.Run(tc.tokens, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.temperature*1000 + tc.topP*10)))
			m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 16, VocabSize: 6})
			ids := []int{rng.Intn(6), rng.Intn(6)}
			randoms := []float64{rng.Float64(), rng.Float64(), rng.Float64(), rng.Float64()}
			var stdin strings.Builder
			for _, r := range randoms {
				fmt.Fprintln(&stdin, r)
			}

			got := runGenerate(t, stdin.String(), generateFiles(t, m, ids, tc.tokens))
			if want := refGenerateSampled(m, ids, tc.temperature, tc.topP, randoms); !reflect.DeepEqual(got, want) {
				t.Fatalf("got %v, want %v", got, want)
			}
		})
	}
}

// failingReader fails the test if anything reads from it.
type failingReader struct{ t *testing.T }

func (f failingReader) Read([]byte) (int, error) {
	f.t.Error("stdin was read, but greedy decoding needs no random numbers")
	return 0, errors.New("stdin should not be read")
}

func TestGreedyNeverReadsStdin(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	for _, tokens := range []string{`{"count": 3}`, `{"count": 3, "temperature": 0}`} {
		files := generateFiles(t, m, []int{1, 2}, tokens)
		var out bytes.Buffer
		if err := run(append([]string{"generate"}, files...), failingReader{t}, &out); err != nil {
			t.Fatal(err)
		}
		if got, want := decodeJSON[[]int](t, out.String()), refGreedy(m, []int{1, 2}, 3); !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: got %v, want %v", tokens, got, want)
		}
	}
}

func TestSampleReadsOneNumberPerToken(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 1, NPositions: 8, VocabSize: 6})
	files := generateFiles(t, m, []int{1}, `{"count": 3, "temperature": 1, "top_p": 1}`)

	// Numbers on one line, or with blank lines and CRLFs, read the same.
	want := refGenerateSampled(m, []int{1}, 1, 1, []float64{0.25, 0.5, 0.75})
	for _, stdin := range []string{"0.25\n0.5\n0.75\n", "0.25\r\n\n0.5\r\n0.75", "0.25 0.5 0.75"} {
		if got := runGenerate(t, stdin, files); !reflect.DeepEqual(got, want) {
			t.Errorf("stdin %q: got %v, want %v", stdin, got, want)
		}
	}

	for name, stdin := range map[string]string{"too few numbers": "0.25\n0.5\n", "not a number": "0.25\nhalf\n0.75\n"} {
		var out bytes.Buffer
		if err := run(append([]string{"generate"}, files...), strings.NewReader(stdin), &out); err == nil {
			t.Errorf("%s: expected an error, got %q", name, out.String())
		}
	}
}
