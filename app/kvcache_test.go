package main

import (
	"math/rand"
	"reflect"
	"testing"
)

// generateUncached is generate without the KV cache, as the "Greedy decode"
// stage wrote it: every step runs the whole sequence through every layer.
func (m *gpt2) generateUncached(ids []int, count int, pick tokenPicker) ([]int, error) {
	seq := append([]int(nil), ids...)
	newIDs := make([]int, 0, count)
	for len(newIDs) < count {
		scores, err := m.nextLogits(seq)
		if err != nil {
			return nil, err
		}
		next, err := pick(scores)
		if err != nil {
			return nil, err
		}
		seq = append(seq, next)
		newIDs = append(newIDs, next)
	}
	return newIDs, nil
}

// refGreedyAtPositionZero is a deliberately broken reference that embeds
// every new token at position 0, the mistake the stage warns about.
func refGreedyAtPositionZero(m *testModel, ids []int, count int) []int {
	n := m.cfg.NEmbd
	wte, wpe := m.values["wte.weight"], m.values["wpe.weight"]
	var rows [][]float64
	embedAt := func(id, pos int) {
		row := make([]float64, n)
		for j := range row {
			row[j] = float64(wte[id*n+j]) + float64(wpe[pos*n+j])
		}
		rows = append(rows, row)
	}
	for pos, id := range ids {
		embedAt(id, pos)
	}
	var newIDs []int
	for len(newIDs) < count {
		hidden := rows
		for layer := 0; layer < m.cfg.NLayer; layer++ {
			hidden = refBlock(m, layer, hidden)
		}
		last := refLayerNorm(hidden[len(hidden)-1], m.values["ln_f.weight"], m.values["ln_f.bias"])
		best, bestScore := 0, 0.0
		for tok := 0; tok < m.cfg.VocabSize; tok++ {
			score := 0.0
			for j := 0; j < n; j++ {
				score += last[j] * float64(wte[tok*n+j])
			}
			if tok == 0 || score > bestScore {
				best, bestScore = tok, score
			}
		}
		newIDs = append(newIDs, best)
		embedAt(best, 0)
	}
	return newIDs
}

func loadTestModel(t *testing.T, m *testModel) *gpt2 {
	t.Helper()
	model, err := loadModel(tensorsOf(m), m.cfg)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestKVCacheMatchesRecomputingEverything(t *testing.T) {
	for _, tc := range []struct{ layers, heads, embd, promptLen, count int }{
		{1, 1, 4, 3, 3}, {2, 2, 4, 3, 4}, {2, 2, 4, 1, 6}, {3, 4, 8, 5, 5}, {2, 2, 4, 2, 1},
	} {
		for seed := int64(0); seed < 5; seed++ {
			rng := rand.New(rand.NewSource(1000*int64(tc.layers) + 100*int64(tc.promptLen) + seed))
			m := newRandomModel(rng, config{NEmbd: tc.embd, NHead: tc.heads, NLayer: tc.layers, NPositions: 16, VocabSize: 6})
			ids := make([]int, tc.promptLen)
			for i := range ids {
				ids[i] = rng.Intn(6)
			}
			model := loadTestModel(t, m)
			cached, err := model.generate(ids, tc.count, greedy)
			if err != nil {
				t.Fatal(err)
			}
			uncached, err := model.generateUncached(ids, tc.count, greedy)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cached, uncached) {
				t.Fatalf("%+v seed %d: cached %v, recomputed %v", tc, seed, cached, uncached)
			}
			if want := refGreedy(m, ids, tc.count); !reflect.DeepEqual(cached, want) {
				t.Fatalf("%+v seed %d: cached %v, reference %v", tc, seed, cached, want)
			}
		}
	}
}

func TestKVCacheEmbedsNewTokensAtTheirPosition(t *testing.T) {
	// Find two-ID prompts where embedding new tokens at position 0 would
	// change the IDs, so these checks could catch that mistake.
	found := 0
	for seed := int64(0); seed < 200 && found < 3; seed++ {
		rng := rand.New(rand.NewSource(2000 + seed))
		m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 8, VocabSize: 6})
		ids := []int{rng.Intn(6), rng.Intn(6)}
		want := refGreedy(m, ids, 4)
		if reflect.DeepEqual(want, refGreedyAtPositionZero(m, ids, 4)) {
			continue
		}
		found++
		got := runGenerate(t, "", generateFiles(t, m, ids, `{"count": 4}`))
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("seed %d, ids %v: got %v, want %v", seed, ids, got, want)
		}
	}
	if found == 0 {
		t.Fatal("no prompt where the position mattered; the test checks nothing")
	}
}

func TestKVCacheKeepsEachLayersKeysAndValues(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 3, NPositions: 16, VocabSize: 6})
	model := loadTestModel(t, m)

	seq := model.newSequence()
	if _, err := seq.feed([]int{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := seq.feed([]int{4}); err != nil {
		t.Fatal(err)
	}
	if seq.length != 4 {
		t.Errorf("length %d, want 4", seq.length)
	}
	for layer, cache := range seq.caches {
		if len(cache.keys) != 4 || len(cache.values) != 4 {
			t.Errorf("layer %d keeps %d keys and %d values, want 4 of each", layer, len(cache.keys), len(cache.values))
		}
		if layer > 0 && reflect.DeepEqual(cache.keys, seq.caches[layer-1].keys) {
			t.Errorf("layers %d and %d share keys", layer-1, layer)
		}
	}
}

func TestKVCacheFeedingInPiecesMatchesAllAtOnce(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	m := newRandomModel(rng, config{NEmbd: 8, NHead: 4, NLayer: 2, NPositions: 16, VocabSize: 6})
	model := loadTestModel(t, m)
	ids := []int{5, 0, 3, 3, 1, 4}

	whole, err := model.forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	seq := model.newSequence()
	var pieces [][]float64
	for _, piece := range [][]int{ids[:2], ids[2:3], ids[3:]} {
		rows, err := seq.feed(piece)
		if err != nil {
			t.Fatal(err)
		}
		pieces = append(pieces, rows...)
	}
	assertClose(t, pieces, whole, 0) // the same arithmetic, so exactly equal
}

func TestKVCacheWithSampling(t *testing.T) {
	rng := rand.New(rand.NewSource(51))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 16, VocabSize: 6})
	model := loadTestModel(t, m)
	randoms := []float64{0.9, 0.1, 0.5, 0.7, 0.3}
	picker := func() tokenPicker {
		i := 0
		return func(scores []float64) (int, error) {
			i++
			return sample(scores, 0.8, 0.9, randoms[i-1]), nil
		}
	}
	cached, err := model.generate([]int{2, 4}, 5, picker())
	if err != nil {
		t.Fatal(err)
	}
	uncached, err := model.generateUncached([]int{2, 4}, 5, picker())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cached, uncached) {
		t.Fatalf("cached %v, recomputed %v", cached, uncached)
	}
	if want := refGenerateSampled(m, []int{2, 4}, 0.8, 0.9, randoms); !reflect.DeepEqual(cached, want) {
		t.Fatalf("cached %v, reference %v", cached, want)
	}
}

func TestKVCacheRunsEachRowOnce(t *testing.T) {
	// The stage's arithmetic: 5 prompt rows and 6 new tokens run 10 rows
	// through each layer with the cache, against 5 + 6 + ... + 10 = 45
	// without. Each row run through a layer adds one key to its cache.
	rng := rand.New(rand.NewSource(61))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 16, VocabSize: 6})
	seq := loadTestModel(t, m).newSequence()
	newIDs, err := seq.generate([]int{1, 2, 3, 4, 5}, 6, greedy)
	if err != nil {
		t.Fatal(err)
	}
	if len(newIDs) != 6 {
		t.Fatalf("got %d new IDs, want 6", len(newIDs))
	}
	for layer, cache := range seq.caches {
		if len(cache.keys) != 10 {
			t.Errorf("layer %d ran %d rows, want 10", layer, len(cache.keys))
		}
	}
	if want := refGreedy(m, []int{1, 2, 3, 4, 5}, 6); !reflect.DeepEqual(newIDs, want) {
		t.Errorf("got %v, want %v", newIDs, want)
	}
}
