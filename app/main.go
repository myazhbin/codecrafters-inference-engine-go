package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
)

// Usage: your_program.sh <command> <arg1> <arg2> ...
func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
		os.Exit(1)
	}
}

// run executes one command. Only the command's JSON result goes to stdout.
func run(args []string, stdin io.Reader, stdout io.Writer) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: your_program.sh <command> [<args>...]")
	}
	command, args := args[0], args[1:]
	switch command {
	case "inspect":
		if len(args) != 1 {
			return fmt.Errorf("usage: inspect <model.safetensors>")
		}
		return inspect(args[0], stdout)
	case "embed":
		if len(args) != 2 {
			return fmt.Errorf("usage: embed <model.safetensors> <ids.json>")
		}
		return embedCommand(args[0], args[1], stdout)
	case "attn":
		if len(args) != 4 {
			return fmt.Errorf("usage: attn <model.safetensors> <config.json> <layer> <input.json>")
		}
		return attnCommand(args[0], args[1], args[2], args[3], stdout)
	default:
		return fmt.Errorf("unknown command %s", command)
	}
}

// inspect prints every tensor's name, dtype, shape and first 4 numbers,
// sorted by name.
func inspect(modelPath string, stdout io.Writer) error {
	tensors, err := loadTensors(modelPath)
	if err != nil {
		return fmt.Errorf("reading tensors: %w", err)
	}

	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	type summary struct {
		Name  string `json:"name"`
		DType string `json:"dtype"`
		Shape []int  `json:"shape"`
		First []any  `json:"first"`
	}
	summaries := make([]summary, 0, len(names))
	for _, name := range names {
		t := tensors[name]
		n := min(4, len(t.Values))
		first := make([]any, n)
		for i := range first {
			if t.Ints != nil {
				first[i] = t.Ints[i] // exact, even past float32's 2^24
			} else {
				first[i] = t.Values[i] // float32, printed in its shortest form
			}
		}
		summaries = append(summaries, summary{name, t.DType, t.Shape, first})
	}
	return writeJSON(stdout, summaries)
}

// embedCommand prints the embedding of each token ID in ids.json.
func embedCommand(modelPath, idsPath string, stdout io.Writer) error {
	tensors, err := loadTensors(modelPath)
	if err != nil {
		return fmt.Errorf("reading tensors: %w", err)
	}
	ids, err := readIDs(idsPath)
	if err != nil {
		return err
	}
	rows, err := embed(tensors, ids, 0)
	if err != nil {
		return err
	}
	return writeJSON(stdout, rows)
}

// layerInputs reads what attn and block share: the weights, the config,
// the layer number, and the input rows.
func layerInputs(modelPath, configPath, layerArg, inputPath string) (weights, config, int, [][]float64, error) {
	cfg, err := readConfig(configPath)
	if err != nil {
		return nil, config{}, 0, nil, err
	}
	layer, err := strconv.Atoi(layerArg)
	if err != nil || layer < 0 {
		return nil, config{}, 0, nil, fmt.Errorf("layer must be a number from 0 up, got %q", layerArg)
	}
	rows, err := readRows(inputPath, cfg.NEmbd)
	if err != nil {
		return nil, config{}, 0, nil, err
	}
	tensors, err := loadTensors(modelPath)
	if err != nil {
		return nil, config{}, 0, nil, fmt.Errorf("reading tensors: %w", err)
	}
	return tensors, cfg, layer, rows, nil
}

// attnCommand prints one layer's attention output for the input rows.
func attnCommand(modelPath, configPath, layerArg, inputPath string, stdout io.Writer) error {
	w, cfg, layer, rows, err := layerInputs(modelPath, configPath, layerArg, inputPath)
	if err != nil {
		return err
	}
	attn, err := loadAttention(w, cfg, layer)
	if err != nil {
		return err
	}
	return writeJSON(stdout, attn.forward(rows))
}

// writeJSON prints v as one line of JSON.
func writeJSON(w io.Writer, v any) error {
	encoded, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("encoding output: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", encoded)
	return err
}
