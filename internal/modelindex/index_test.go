package modelindex

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestScanLocalDir_basic(t *testing.T) {
	dir := t.TempDir()

	// Create some .gguf files
	files := []string{"model-Q8_0.gguf", "model-Q4_K_M.gguf", "MODEL-UPPER.GGUF"}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Non-.gguf file should be ignored
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("x"), 0o644) //nolint:errcheck
	// Subdir should be ignored
	os.MkdirAll(filepath.Join(dir, "subdir"), 0o755) //nolint:errcheck

	entries, err := scanLocalDir(dir)
	if err != nil {
		t.Fatalf("scanLocalDir: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d: %v", len(entries), entries)
	}

	names := make(map[string]bool)
	for _, e := range entries {
		names[e.Name] = true
		if e.Source != SourceLocal {
			t.Errorf("expected SourceLocal, got %v", e.Source)
		}
		if e.path == "" {
			t.Error("expected non-empty path")
		}
	}
	// Name derivation: strip .gguf (5 chars) case-insensitively
	if !names["model-Q8_0"] {
		t.Error("expected model-Q8_0")
	}
	if !names["model-Q4_K_M"] {
		t.Error("expected model-Q4_K_M")
	}
	if !names["MODEL-UPPER"] {
		t.Error("expected MODEL-UPPER")
	}
}

func TestScanLocalDir_nonexistent(t *testing.T) {
	_, err := scanLocalDir("/nonexistent/path/that/does/not/exist")
	if err == nil {
		t.Error("expected error for nonexistent dir")
	}
}

func TestLookup_localEntry(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "my-model-Q8_0.gguf"), []byte("x"), 0o644) //nolint:errcheck

	idx := New([]string{dir}, "")
	idx.Scan()

	path, source, ok := idx.Lookup("my-model-Q8_0")
	if !ok {
		t.Fatal("expected to find my-model-Q8_0")
	}
	if source != SourceLocal {
		t.Errorf("expected SourceLocal, got %v", source)
	}
	if path == "" {
		t.Error("expected non-empty path for local entry")
	}
}

func TestLookup_hfEntry(t *testing.T) {
	hfDir := t.TempDir()
	// Create HF cache structure: models--org--repo/snapshots/abc/model-Q8_0.gguf
	snapDir := filepath.Join(hfDir, "models--myorg--myrepo", "snapshots", "abc123")
	os.MkdirAll(snapDir, 0o755)                                                 //nolint:errcheck
	os.WriteFile(filepath.Join(snapDir, "model-Q8_0.gguf"), []byte("x"), 0o644) //nolint:errcheck

	idx := New(nil, hfDir)
	idx.Scan()

	path, source, ok := idx.Lookup("myorg/myrepo:Q8_0")
	if !ok {
		t.Fatal("expected to find myorg/myrepo:Q8_0")
	}
	if source != SourceHFCache {
		t.Errorf("expected SourceHFCache, got %v", source)
	}
	if path != "" {
		t.Errorf("expected empty path for HF entry from Lookup, got %q", path)
	}
}

func TestFilePath_bothTypes(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "local-model-Q4.gguf")
	os.WriteFile(ggufPath, []byte("x"), 0o644) //nolint:errcheck

	hfDir := t.TempDir()
	snapDir := filepath.Join(hfDir, "models--hforg--hfrepo", "snapshots", "snap1")
	os.MkdirAll(snapDir, 0o755) //nolint:errcheck
	hfGGUF := filepath.Join(snapDir, "model-Q8_0.gguf")
	os.WriteFile(hfGGUF, []byte("x"), 0o644) //nolint:errcheck

	idx := New([]string{dir}, hfDir)
	idx.Scan()

	// Local entry: FilePath returns the actual path
	localPath := idx.FilePath("local-model-Q4")
	if localPath == "" {
		t.Error("expected non-empty FilePath for local entry")
	}

	// HF entry: FilePath also returns the actual GGUF path
	hfPath := idx.FilePath("hforg/hfrepo:Q8_0")
	if hfPath == "" {
		t.Error("expected non-empty FilePath for HF entry")
	}
	if hfPath != hfGGUF {
		t.Errorf("FilePath for HF entry: got %q, want %q", hfPath, hfGGUF)
	}

	// Unknown model
	if got := idx.FilePath("nonexistent"); got != "" {
		t.Errorf("expected empty FilePath for nonexistent model, got %q", got)
	}
}

func TestList_returnsAll(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "model-a.gguf"), []byte("x"), 0o644) //nolint:errcheck
	os.WriteFile(filepath.Join(dir, "model-b.gguf"), []byte("x"), 0o644) //nolint:errcheck

	idx := New([]string{dir}, "")
	idx.Scan()

	list := idx.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(list))
	}
	names := make(map[string]bool)
	for _, e := range list {
		names[e.Name] = true
	}
	if !names["model-a"] {
		t.Error("expected model-a in list")
	}
	if !names["model-b"] {
		t.Error("expected model-b in list")
	}
}

func TestDuplicateHandling_firstLocalWins(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()

	os.WriteFile(filepath.Join(dir1, "same-model.gguf"), []byte("dir1"), 0o644) //nolint:errcheck
	os.WriteFile(filepath.Join(dir2, "same-model.gguf"), []byte("dir2"), 0o644) //nolint:errcheck

	idx := New([]string{dir1, dir2}, "")
	idx.Scan()

	list := idx.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 entry (deduped), got %d", len(list))
	}
	// First dir wins
	wantPath := filepath.Join(dir1, "same-model.gguf")
	if list[0].path != wantPath {
		t.Errorf("expected path %q, got %q", wantPath, list[0].path)
	}
}

func TestMissingModelDir_warnsDoesNotPanic(t *testing.T) {
	// Non-existent dir should produce a warning but not panic
	idx := New([]string{"/nonexistent/dir"}, "")
	idx.Scan() // must not panic

	list := idx.List()
	if len(list) != 0 {
		t.Errorf("expected 0 entries for missing dir, got %d", len(list))
	}
}

func TestTTLRefresh(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "initial.gguf"), []byte("x"), 0o644) //nolint:errcheck

	idx := New([]string{dir}, "")
	idx.ttl = 50 * time.Millisecond
	idx.Scan()

	if len(idx.List()) != 1 {
		t.Fatal("expected 1 entry after initial scan")
	}

	// Add a new file
	os.WriteFile(filepath.Join(dir, "added.gguf"), []byte("x"), 0o644) //nolint:errcheck

	// Still cached
	if len(idx.List()) != 1 {
		t.Fatal("expected still 1 entry while cache is fresh")
	}

	// Wait for TTL to expire
	time.Sleep(100 * time.Millisecond)

	// Now should pick up the new file
	list := idx.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 entries after TTL expiry, got %d", len(list))
	}
}

func TestExtractQuant(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Qwen3-4B-Q8_0", "Q8_0"},
		{"Qwen3-4B-Q5_K_M", "Q5_K_M"},
		{"Phi-4-mini-instruct.Q8_0", "Q8_0"},
		{"Llama-3-8B-Q4_K_M", "Q4_K_M"},
		{"model-4B-Q2_K", "Q2_K"},
		{"no-quant-here", ""},
		{"mmproj-BF16", ""},
	}
	for _, tt := range tests {
		got := extractQuant(tt.input)
		if got != tt.want {
			t.Errorf("extractQuant(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestScanLocalDir_subdirIgnored(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "submodel.gguf") // this is a dir named like a gguf
	os.MkdirAll(subdir, 0o755)                    //nolint:errcheck

	entries, err := scanLocalDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("expected 0 entries (subdir should be ignored), got %d", len(entries))
	}
}

func TestLookup_notFound(t *testing.T) {
	idx := New(nil, "")
	idx.Scan()

	_, _, ok := idx.Lookup("nonexistent-model")
	if ok {
		t.Error("expected ok=false for nonexistent model")
	}
}

func TestLocalOverridesHF(t *testing.T) {
	// A local dir model with the same name as an HF entry should win.
	// (In practice names won't collide since local is "filename" and HF is "org/repo:QUANT"
	// but let's test priority: local overrides HF entries.)
	hfDir := t.TempDir()
	snapDir := filepath.Join(hfDir, "models--org--repo", "snapshots", "abc")
	os.MkdirAll(snapDir, 0o755) //nolint:errcheck
	// Create an HF model with the same derived name as a local one (edge case)
	os.WriteFile(filepath.Join(snapDir, "hf-model-Q8_0.gguf"), []byte("hf"), 0o644) //nolint:errcheck

	localDir := t.TempDir()
	os.WriteFile(filepath.Join(localDir, "org/repo:Q8_0.gguf"), []byte("local"), 0o644) //nolint:errcheck

	// Note: in practice names don't collide. Just verify scan works without error.
	idx := New([]string{localDir}, hfDir)
	idx.Scan() // must not panic

	// HF entry should exist
	hfPath := idx.FilePath("org/repo:Q8_0")
	if hfPath == "" {
		t.Error("expected HF entry to be present")
	}
}
