package vram

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// HFCache tracks VRAM estimates for models found in the HuggingFace blob cache.
type HFCache struct {
	cacheDir  string
	estimates sync.Map // modelRef → uint64 (VRAM estimate)
}

// NewHFCache creates a cache scanner for the given HuggingFace cache directory.
func NewHFCache(cacheDir string) *HFCache {
	return &HFCache{cacheDir: cacheDir}
}

// Scan walks the HF cache and populates VRAM estimates for GGUF models.
// It resolves model references to local blob paths and applies the 1.3× heuristic.
// Returns the map of model-ref → VRAM estimate (bytes).
func (c *HFCache) Scan() map[string]uint64 {
	results := make(map[string]uint64)
	if c.cacheDir == "" {
		return results
	}

	modelsDir := filepath.Join(c.cacheDir, "models")
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		slog.Debug("HF cache models dir not found", "dir", modelsDir, "error", err)
		return results
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		modelRef := entry.Name()
		modelDir := filepath.Join(modelsDir, modelRef)

		ggufPath := findGGUF(modelDir)
		if ggufPath == "" {
			continue
		}

		info, err := os.Stat(ggufPath)
		if err != nil {
			slog.Warn("stat GGUF file", "path", ggufPath, "error", err)
			continue
		}

		estimate := uint64(float64(info.Size()) * vramMultiplier)
		results[modelRef] = estimate
		c.estimates.Store(modelRef, estimate)

		slog.Info("HF cache: VRAM estimate", "model", modelRef, "bytes", estimate)
	}
	return results
}

// Estimate returns a previously scanned VRAM estimate for modelRef.
// Returns 0 if the model was not found in the cache scan.
func (c *HFCache) Estimate(modelRef string) uint64 {
	if v, ok := c.estimates.Load(modelRef); ok {
		return v.(uint64)
	}
	return 0
}

// findGGUF locates the first .gguf file under dir.
func findGGUF(dir string) string {
	// Use a closure to walk and stop early.
	var found string
	filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(path), ".gguf") {
			found = path
			return filepath.SkipAll
		}
		return nil
	})
	return found
}
