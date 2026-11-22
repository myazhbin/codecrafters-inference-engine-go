package main

// These tests run GPT-2's own weights. They need a folder outside the
// repository holding model.safetensors (and, optionally, vocab.json) from
// https://huggingface.co/openai-community/gpt2:
//
//	GPT2_DIR=/path/to/folder go test ./app/ -run GPT2 -v
//
// Without GPT2_DIR they are skipped.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

const gpt2Config = `{ "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "vocab_size": 50257 }`

// "The capital of France is"
var gpt2Prompt = []int{464, 3139, 286, 4881, 318}

func gpt2Dir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GPT2_DIR")
	if dir == "" {
		t.Skip("set GPT2_DIR to a folder holding GPT-2's model.safetensors to run this test")
	}
	return dir
}

var gpt2Shared struct {
	once  sync.Once
	model *gpt2
	err   error
}

// loadGPT2 loads the weights once for all the tests that only read them.
func loadGPT2(t *testing.T) *gpt2 {
	t.Helper()
	dir := gpt2Dir(t)
	gpt2Shared.once.Do(func() {
		var cfg config
		if gpt2Shared.err = json.Unmarshal([]byte(gpt2Config), &cfg); gpt2Shared.err != nil {
			return
		}
		tensors, err := loadTensors(filepath.Join(dir, "model.safetensors"))
		if err != nil {
			gpt2Shared.err = err
			return
		}
		gpt2Shared.model, gpt2Shared.err = loadModel(tensors, cfg)
	})
	if gpt2Shared.err != nil {
		t.Fatal(gpt2Shared.err)
	}
	return gpt2Shared.model
}

// gpt2Vocab maps token IDs to text, with Ġ turned back into a space. It
// returns nil when vocab.json isn't there.
func gpt2Vocab(t *testing.T) map[int]string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(gpt2Dir(t), "vocab.json"))
	if err != nil {
		return nil
	}
	var byText map[string]int
	if err := json.Unmarshal(data, &byText); err != nil {
		t.Fatal(err)
	}
	byID := make(map[int]string, len(byText))
	for text, id := range byText {
		byID[id] = strings.ReplaceAll(text, "Ġ", " ")
	}
	return byID
}

// topTwo returns the IDs with the highest and second-highest scores.
func topTwo(scores []float64) (int, int) {
	first := argmax(scores)
	second := -1
	for i, s := range scores {
		if i != first && (second == -1 || s > scores[second]) {
			second = i
		}
	}
	return first, second
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }

func TestGPT2GenerateLikeTheTester(t *testing.T) {
	dir := gpt2Dir(t)
	work := t.TempDir()
	// The tester links GPT-2's weights to model.safetensors.
	model := filepath.Join(work, "model.safetensors")
	if err := os.Symlink(filepath.Join(dir, "model.safetensors"), model); err != nil {
		t.Fatal(err)
	}
	files := []string{
		model,
		writeFile(t, work, "config.json", gpt2Config),
		writeFile(t, work, "ids.json", "[464, 3139, 286, 4881, 318]"),
		writeFile(t, work, "tokens.json", `{"count": 6}`),
	}

	start := time.Now()
	got := runGenerate(t, "", files)
	elapsed := time.Since(start)
	t.Logf("generated %v in %v", got, elapsed)
	if want := []int{262, 3139, 286, 262, 4141, 2066}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if elapsed > 300*time.Second {
		t.Errorf("took %v; the tester stops programs after 300s", elapsed)
	}

	if vocab := gpt2Vocab(t); vocab != nil {
		var text strings.Builder
		for _, id := range got {
			text.WriteString(vocab[id])
		}
		if text.String() != " the capital of the French Republic" {
			t.Errorf("wrote %q", text.String())
		}
	}
}

func TestGPT2LogitsMatchTheLogitsStage(t *testing.T) {
	m := loadGPT2(t)
	scores, err := m.nextLogits(gpt2Prompt)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 50257 {
		t.Fatalf("got %d logits, want 50257", len(scores))
	}
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })

	// " the" -100.25, " now" -100.82, " a" -100.86, then " France" and
	// " Paris", both -101.21.
	want := map[int]float64{262: -100.25, 783: -100.82, 257: -100.86, 4881: -101.21, 6342: -101.21}
	if !reflect.DeepEqual(order[:3], []int{262, 783, 257}) {
		t.Errorf("top three are %v, want [262 783 257]", order[:3])
	}
	for _, id := range order[:5] {
		if w, ok := want[id]; !ok || round2(scores[id]) != w {
			t.Errorf("token %d scored %.4f, want %v", id, scores[id], w)
		}
	}
}

func TestGPT2GreedyLeadsMatchTheGreedyStage(t *testing.T) {
	m := loadGPT2(t)
	steps := []struct {
		picked, runnerUp int
		lead             float64
	}{
		{262, 783, 0.57},   // " the" over " now"
		{3139, 1748, 1.62}, // " capital" over " city"
		{286, 11, 3.87},    // " of" over ","
		{262, 4881, 0.33},  // " the" over " France"
		{4141, 2066, 0.96}, // " French" over " Republic"
		{2066, 8065, 1.32}, // " Republic" over " Empire"
	}
	seq := m.newSequence()
	rows, err := seq.feed(gpt2Prompt)
	if err != nil {
		t.Fatal(err)
	}
	last := rows[len(rows)-1]
	for i, step := range steps {
		scores := m.logits(last)
		picked, runnerUp := topTwo(scores)
		if picked != step.picked || runnerUp != step.runnerUp || round2(scores[picked]-scores[runnerUp]) != step.lead {
			t.Errorf("step %d: picked %d over %d by %.4f, want %d over %d by %v",
				i+1, picked, runnerUp, scores[picked]-scores[runnerUp], step.picked, step.runnerUp, step.lead)
		}
		rows, err := seq.feed([]int{picked})
		if err != nil {
			t.Fatal(err)
		}
		last = rows[0]
	}
	// The KV cache stage notes the seventh step's top two differ by 0.007.
	first, second := topTwo(m.logits(last))
	t.Logf("step 7: %d over %d by %.4f", first, second, m.logits(last)[first]-m.logits(last)[second])
}

func TestGPT2SamplingMatchesTheSampleStage(t *testing.T) {
	m := loadGPT2(t)
	scores, err := m.nextLogits(gpt2Prompt)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		temperature float64
		probs       map[int]float64 // rounded to 3 places
		keptAt09    int
	}{
		{1, map[int]float64{262: 0.085, 783: 0.048, 257: 0.046, 4881: 0.032, 6342: 0.032}, 1503},
		{0.7, map[int]float64{262: 0.227, 783: 0.101, 257: 0.096, 4881: 0.058, 6342: 0.057}, 62},
	} {
		ids, probs := nucleus(scores, tc.temperature, 1)
		if len(ids) != 50257 {
			t.Errorf("T=%v: top_p 1 kept %d tokens, want all 50257", tc.temperature, len(ids))
		}
		for i, id := range ids[:5] {
			if want := tc.probs[id]; math.Round(probs[i]*1000)/1000 != want {
				t.Errorf("T=%v: token %d has probability %.4f, want %v", tc.temperature, id, probs[i], want)
			}
		}
		if kept, _ := nucleus(scores, tc.temperature, 0.9); len(kept) != tc.keptAt09 {
			t.Errorf("T=%v: top_p 0.9 kept %d tokens, want %d", tc.temperature, len(kept), tc.keptAt09)
		}
	}
}

func TestGPT2InspectListsEveryTensor(t *testing.T) {
	dir := gpt2Dir(t)
	got := decodeJSON[[]inspectEntry](t, runCommand(t, "", "inspect", filepath.Join(dir, "model.safetensors")))
	if len(got) != 160 {
		t.Fatalf("got %d tensors, want 160", len(got))
	}
	shapes := map[string][]int{}
	for _, e := range got {
		shapes[e.Name] = e.Shape
		if e.DType != "F32" || len(e.First) != 4 {
			t.Errorf("%s: dtype %s, %d first numbers", e.Name, e.DType, len(e.First))
		}
	}
	for name, want := range map[string][]int{
		"wte.weight":              {50257, 768},
		"wpe.weight":              {1024, 768},
		"h.0.attn.bias":           {1, 1, 1024, 1024}, // the saved causal mask
		"h.11.attn.c_attn.weight": {768, 2304},
		"h.11.mlp.c_fc.weight":    {768, 3072},
		"ln_f.bias":               {768},
	} {
		if !reflect.DeepEqual(shapes[name], want) {
			t.Errorf("%s has shape %v, want %v", name, shapes[name], want)
		}
	}
}

func TestGPT2LoadingNeedsAboutOneCopyOfTheFile(t *testing.T) {
	// The test server gives the program 2 GB. Reading the file into memory
	// and then converting it would allocate it twice.
	path := filepath.Join(gpt2Dir(t), "model.safetensors")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	tensors, err := loadTensors(path)
	if err != nil {
		t.Fatal(err)
	}
	runtime.ReadMemStats(&after)
	allocated := after.TotalAlloc - before.TotalAlloc
	t.Logf("loading a %d MB file allocated %d MB", info.Size()>>20, allocated>>20)
	if float64(allocated) > 1.1*float64(info.Size()) {
		t.Errorf("loading allocated %d bytes for a %d-byte file", allocated, info.Size())
	}
	runtime.KeepAlive(tensors)
}
