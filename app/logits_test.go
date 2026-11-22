package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// refLogits follows the stage's pseudocode with the reference pieces.
func refLogits(m *testModel, ids []int) []float64 {
	n := m.cfg.NEmbd
	wte, wpe := m.values["wte.weight"], m.values["wpe.weight"]
	var rows [][]float64
	for pos, id := range ids {
		row := make([]float64, n)
		for j := range row {
			row[j] = float64(wte[id*n+j]) + float64(wpe[pos*n+j])
		}
		rows = append(rows, row)
	}
	for layer := 0; layer < m.cfg.NLayer; layer++ {
		rows = refBlock(m, layer, rows)
	}
	last := refLayerNorm(rows[len(rows)-1], m.values["ln_f.weight"], m.values["ln_f.bias"])
	logits := make([]float64, m.cfg.VocabSize)
	for tok := range logits {
		for j := 0; j < n; j++ {
			logits[tok] += last[j] * float64(wte[tok*n+j])
		}
	}
	return logits
}

// assertLogitsClose uses the tester's rule: within 0.001, or 0.001 times the
// expected number, whichever is larger.
func assertLogitsClose(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d logits, want %d\ngot  %v\nwant %v", len(got), len(want), got, want)
	}
	for i := range want {
		if tol := max(0.001, 0.001*math.Abs(want[i])); math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("logit %d: got %v, want %v\ngot  %v\nwant %v", i, got[i], want[i], got, want)
		}
	}
}

func TestLogitsStageExample(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	m := newRandomModel(rng, config{NEmbd: 2, NHead: 1, NLayer: 1, NPositions: 4, VocabSize: 2})
	m.set("wte.weight", 1, 1, 1, -1)
	m.set("ln_f.weight", 0, 0)
	m.set("ln_f.bias", 1, -1)
	dir := t.TempDir()
	model, config := m.write(t, dir)
	ids := writeFile(t, dir, "ids.json", "[0]")

	got := decodeJSON[[]float64](t, runCommand(t, "", "logits", model, config, ids))
	assertLogitsClose(t, got, []float64{0, 2})
	if got[0] != 0 || got[1] != 2 {
		t.Errorf("got %v, want exactly [0, 2]", got)
	}
}

func TestLogitsRandomModels(t *testing.T) {
	for _, tc := range []struct{ layers, heads int }{{1, 2}, {2, 2}, {3, 4}} {
		t.Run(fmt.Sprintf("%d layers", tc.layers), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(700 + tc.layers)))
			m := newRandomModel(rng, config{NEmbd: 4 * tc.heads / 2, NHead: tc.heads, NLayer: tc.layers, NPositions: 8, VocabSize: 6})
			dir := t.TempDir()
			model, config := m.write(t, dir)
			idList := []int{rng.Intn(6), rng.Intn(6), rng.Intn(6)}
			ids := writeFile(t, dir, "ids.json", toJSON(t, idList))

			got := decodeJSON[[]float64](t, runCommand(t, "", "logits", model, config, ids))
			want := refLogits(m, idList)
			assertLogitsClose(t, got, want)
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Fatalf("logit %d: got %v, want %v", i, got[i], want[i])
				}
			}
		})
	}
}

func TestLogitsUseEveryLayer(t *testing.T) {
	// Dropping the last layer must change the scores.
	rng := rand.New(rand.NewSource(9))
	cfg := config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 8, VocabSize: 6}
	m := newRandomModel(rng, cfg)
	full, err := loadModel(tensorsOf(m), cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.NLayer = 1
	short, err := loadModel(tensorsOf(m), cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, err := full.nextLogits([]int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	b, err := short.nextLogits([]int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(a[0]-b[0]) < 1e-9 {
		t.Errorf("one and two layers gave the same logits: %v", a)
	}
}

func TestLogitsRejectsBadInput(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	m := newRandomModel(rng, config{NEmbd: 4, NHead: 2, NLayer: 2, NPositions: 4, VocabSize: 6})
	dir := t.TempDir()
	model, config := m.write(t, dir)
	for name, idsJSON := range map[string]string{
		"no ids":             "[]",
		"id past vocabulary": "[6]",
		"too many positions": "[0, 1, 2, 3, 4]",
	} {
		ids := writeFile(t, dir, "ids.json", idsJSON)
		var out bytes.Buffer
		if err := run([]string{"logits", model, config, ids}, strings.NewReader(""), &out); err == nil {
			t.Errorf("%s: expected an error, got %q", name, out.String())
		}
	}
	// A config with more layers than the file holds is an error too.
	bigger := writeFile(t, dir, "bigger.json", `{"n_embd": 4, "n_head": 2, "n_layer": 3, "n_positions": 4, "vocab_size": 6}`)
	ids := writeFile(t, dir, "ids.json", "[0]")
	var out bytes.Buffer
	if err := run([]string{"logits", model, bigger, ids}, strings.NewReader(""), &out); err == nil {
		t.Errorf("expected an error for a missing layer, got %q", out.String())
	}
}
