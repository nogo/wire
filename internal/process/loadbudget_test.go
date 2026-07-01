package process

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadBudget_OverrideHonored(t *testing.T) {
	// A non-zero override wins regardless of size.
	if got := LoadBudget("/does/not/exist", 42*time.Second); got != 42*time.Second {
		t.Fatalf("override not honored: got %v, want 42s", got)
	}
}

func TestLoadBudget_EmptyOrMissingPathIsFloor(t *testing.T) {
	if got := LoadBudget("", 0); got != 60*time.Second {
		t.Fatalf("empty path: got %v, want base 60s", got)
	}
	if got := LoadBudget("/does/not/exist", 0); got != 60*time.Second {
		t.Fatalf("missing file: got %v, want base 60s", got)
	}
}

func TestLoadBudget_ScalesWithSize(t *testing.T) {
	dir := t.TempDir()
	big := filepath.Join(dir, "big.gguf")
	f, err := os.Create(big)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(3 << 30); err != nil { // 3 GiB sparse
		t.Fatal(err)
	}
	f.Close()

	// base 60s + 3 GiB * 6s/GB = 78s
	if got := LoadBudget(big, 0); got != 78*time.Second {
		t.Fatalf("3 GiB budget: got %v, want 78s", got)
	}
}

func TestLoadBudget_Capped(t *testing.T) {
	dir := t.TempDir()
	huge := filepath.Join(dir, "huge.gguf")
	f, err := os.Create(huge)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(500 << 30); err != nil { // 500 GiB sparse → far past the cap
		t.Fatal(err)
	}
	f.Close()

	if got := LoadBudget(huge, 0); got != 15*time.Minute {
		t.Fatalf("cap: got %v, want 15m", got)
	}
}
