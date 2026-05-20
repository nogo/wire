package modelindex

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type Source int

const (
	SourceLocal Source = iota
	SourceHFCache
)

// Entry is one discovered model.
type Entry struct {
	Name   string // derived name (HF: "org/repo:QUANT", local: "filename-without-ext")
	path   string // actual GGUF file path (set for both local and HF entries)
	Source Source
}

const defaultTTL = 30 * time.Second

// Index is a TTL-refreshed map from model name → Entry.
// Local dir models use -m <path>; HF cache models use -hf <name>.
type Index struct {
	mu         sync.RWMutex
	entries    map[string]Entry
	until      time.Time
	modelDirs  []string
	hfCacheDir string
	ttl        time.Duration
}

// New creates an Index. modelDirs are walked for *.gguf files; hfCacheDir is
// scanned for the HuggingFace hub layout. Either may be empty.
func New(modelDirs []string, hfCacheDir string) *Index {
	return &Index{
		modelDirs:  modelDirs,
		hfCacheDir: hfCacheDir,
		ttl:        defaultTTL,
	}
}

// Scan performs an immediate scan and logs results. Call once at startup.
func (idx *Index) Scan() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.entries = idx.build()
	idx.until = time.Now().Add(idx.ttl)
}

// Lookup returns the load info for a model name.
// path is the GGUF file path when Source==SourceLocal; empty when Source==SourceHFCache
// (caller should use `-hf name`).
func (idx *Index) Lookup(name string) (path string, source Source, ok bool) {
	idx.refreshIfNeeded()
	idx.mu.RLock()
	e, ok := idx.entries[name]
	idx.mu.RUnlock()
	if !ok {
		return "", 0, false
	}
	if e.Source == SourceLocal {
		return e.path, e.Source, true
	}
	return "", e.Source, true
}

// FilePath returns the actual GGUF file path for any entry regardless of source.
// Used for VRAM file-size estimation.
func (idx *Index) FilePath(name string) string {
	idx.refreshIfNeeded()
	idx.mu.RLock()
	e, ok := idx.entries[name]
	idx.mu.RUnlock()
	if !ok {
		return ""
	}
	return e.path
}

// List returns all discovered model entries.
func (idx *Index) List() []Entry {
	idx.refreshIfNeeded()
	idx.mu.RLock()
	out := make([]Entry, 0, len(idx.entries))
	for _, e := range idx.entries {
		out = append(out, e)
	}
	idx.mu.RUnlock()
	return out
}

func (idx *Index) refreshIfNeeded() {
	idx.mu.RLock()
	fresh := time.Now().Before(idx.until)
	idx.mu.RUnlock()
	if fresh {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if time.Now().Before(idx.until) {
		return
	}
	idx.entries = idx.build()
	idx.until = time.Now().Add(idx.ttl)
}

// build scans all sources and returns the merged entries. Call with write lock held.
func (idx *Index) build() map[string]Entry {
	result := make(map[string]Entry)

	// HF cache first (lower priority than local dirs).
	if idx.hfCacheDir != "" {
		for _, e := range scanHFCache(idx.hfCacheDir) {
			result[e.Name] = e
		}
	}

	// Local dirs (higher priority; first match wins within local dirs).
	for _, dir := range idx.modelDirs {
		entries, err := scanLocalDir(dir)
		if err != nil {
			slog.Warn("model_dirs: cannot scan directory", "dir", dir, "error", err)
			continue
		}
		for _, e := range entries {
			if existing, dup := result[e.Name]; dup && existing.Source == SourceLocal {
				slog.Warn("model_dirs: duplicate model name; using first match", "name", e.Name, "kept", existing.path, "ignored", e.path)
				continue
			}
			result[e.Name] = e
			slog.Info("model_dirs: discovered model", "name", e.Name, "path", e.path)
		}
	}

	return result
}

// scanLocalDir returns all *.gguf entries in dir (non-recursive).
func scanLocalDir(dir string) ([]Entry, error) {
	infos, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []Entry
	for _, info := range infos {
		if info.IsDir() {
			continue
		}
		name := info.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}
		absPath, err := filepath.Abs(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		// Strip .gguf case-insensitively.
		modelName := name[:len(name)-5]
		out = append(out, Entry{Name: modelName, path: absPath, Source: SourceLocal})
	}
	return out, nil
}

// scanHFCache returns model entries from the HuggingFace hub cache directory.
// Names are in "org/repo:QUANT" format.
func scanHFCache(hfCacheDir string) []Entry {
	entries, err := os.ReadDir(hfCacheDir)
	if err != nil {
		slog.Debug("HF cache dir not readable", "dir", hfCacheDir, "error", err)
		return nil
	}

	var out []Entry
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || !strings.HasPrefix(name, "models--") {
			continue
		}
		// models--org--repo → org/repo
		parts := strings.SplitN(name, "--", 3)
		if len(parts) != 3 {
			continue
		}
		repo := parts[1] + "/" + parts[2]

		snapshotsDir := filepath.Join(hfCacheDir, name, "snapshots")
		snapshots, err := os.ReadDir(snapshotsDir)
		if err != nil {
			continue
		}
		seen := make(map[string]bool)
		for _, snap := range snapshots {
			if !snap.IsDir() {
				continue
			}
			ggufs, err := filepath.Glob(filepath.Join(snapshotsDir, snap.Name(), "*.gguf"))
			if err != nil {
				continue
			}
			for _, ggufPath := range ggufs {
				base := strings.TrimSuffix(filepath.Base(ggufPath), ".gguf")
				if strings.HasPrefix(base, "mmproj") {
					continue
				}
				quant := extractQuant(base)
				if quant == "" {
					continue
				}
				id := repo + ":" + quant
				if !seen[id] {
					seen[id] = true
					out = append(out, Entry{Name: id, path: ggufPath, Source: SourceHFCache})
				}
			}
		}
	}
	return out
}

// extractQuant extracts the quantization suffix from a GGUF filename base.
func extractQuant(base string) string {
	for i := len(base) - 1; i >= 0; i-- {
		if i > 0 && base[i] == 'B' && base[i-1] >= '0' && base[i-1] <= '9' {
			break
		}
		if i+1 < len(base) && base[i] == 'Q' && base[i+1] >= '0' && base[i+1] <= '9' {
			return base[i:]
		}
	}
	return ""
}
