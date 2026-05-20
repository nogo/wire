package vram

import (
	"fmt"
	"os"
	"strings"
)

const vramMultiplier = 1.3

// EstimateVRAM returns the estimated VRAM bytes needed to load a GGUF model.
// It stats the file at modelPath and applies the 1.3× heuristic
// (file size + 30% KV cache overhead).
// For on-demand HF models that aren't in the local cache yet,
// it returns a conservative placeholder based on model name pattern.
func EstimateVRAM(modelPath string) (uint64, error) {
	info, err := os.Stat(modelPath)
	if err != nil {
		return 0, fmt.Errorf("stat model %q: %w", modelPath, err)
	}
	if info.IsDir() {
		return 0, fmt.Errorf("model path %q is a directory", modelPath)
	}
	return uint64(float64(info.Size()) * vramMultiplier), nil
}

// PlaceholderVRAM returns a conservative VRAM estimate for a model that
// hasn't been downloaded yet. It parses the model name for known size hints
// (e.g., "7b", "13b", "70b") and uses approximate GGUF sizes.
// Returns 0 if the model name doesn't match any known pattern.
func PlaceholderVRAM(modelName string) uint64 {
	name := strings.ToLower(modelName)
	for pattern, size := range knownModelSizes {
		if strings.Contains(name, pattern) {
			return uint64(float64(size) * vramMultiplier)
		}
	}
	return 0
}

// knownModelSizes maps common parameter-count hints to approximate GGUF file bytes.
var knownModelSizes = map[string]uint64{
	"0.5b": 500 * 1024 * 1024, // ~500 MiB
	"1b":   1 << 30,           // ~1 GiB
	"1.5b": 1.5 * 1024 * 1024 * 1024,
	"2b":   2 << 30,  // ~2 GiB
	"3b":   3 << 30,  // ~3 GiB
	"7b":   4 << 30,  // ~4 GiB (Q4_K_M)
	"8b":   5 << 30,  // ~5 GiB
	"13b":  8 << 30,  // ~8 GiB
	"14b":  9 << 30,  // ~9 GiB
	"27b":  16 << 30, // ~16 GiB
	"30b":  20 << 30, // ~20 GiB
	"32b":  20 << 30, // ~20 GiB
	"70b":  40 << 30, // ~40 GiB
	"72b":  42 << 30, // ~42 GiB
}
