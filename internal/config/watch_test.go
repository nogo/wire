package config

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const (
	validWatchYAML = `
models:
  "org/model":
    context: 4096
warmup:
  - "org/model"
`
	updatedWatchYAML = `
models:
  "org/model":
    context: 8192
warmup:
  - "org/model"
`
	// Valid YAML that fails our custom validation (warmup entry has no matching pattern).
	invalidWatchYAML = `
models:
  "org/model-A":
    context: 4096
warmup:
  - "org/model-B"
`
)

func setupWatchDir(t *testing.T, content string) (dir, path string) {
	t.Helper()
	dir = t.TempDir()
	path = filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("WIRE_CONFIG", path)
	return dir, path
}

func TestWatch_SuccessfulReload(t *testing.T) {
	_, path := setupWatchDir(t, validWatchYAML)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, err := Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	if err := os.WriteFile(path, []byte(updatedWatchYAML), 0o600); err != nil {
		t.Fatal(err)
	}

	select {
	case cfg := <-ch:
		if cfg == nil {
			t.Fatal("received nil config")
		}
		if cfg.Models["org/model"].Context != 8192 {
			t.Errorf("context: got %d, want 8192", cfg.Models["org/model"].Context)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for config reload")
	}
}

func TestWatch_InvalidConfigRejected(t *testing.T) {
	_, path := setupWatchDir(t, validWatchYAML)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, err := Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	if err := os.WriteFile(path, []byte(invalidWatchYAML), 0o600); err != nil {
		t.Fatal(err)
	}

	// Wait longer than the debounce; nothing should arrive.
	select {
	case cfg, ok := <-ch:
		if ok {
			t.Errorf("expected nothing on channel after invalid config, got: %+v", cfg)
		} else {
			t.Error("channel closed unexpectedly")
		}
	case <-time.After(1200 * time.Millisecond):
		// correct: invalid config was discarded
	}
}

func TestWatch_Debounce(t *testing.T) {
	_, path := setupWatchDir(t, validWatchYAML)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, err := Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	// Write three times in quick succession (50ms apart); total elapsed ~150ms,
	// well within the 500ms debounce window. Only the final write should produce
	// a reload.
	for i := range 3 {
		content := validWatchYAML
		if i == 2 {
			content = updatedWatchYAML
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Expect exactly one config after the debounce fires.
	select {
	case cfg := <-ch:
		if cfg == nil {
			t.Fatal("received nil config")
		}
		if cfg.Models["org/model"].Context != 8192 {
			t.Errorf("context: got %d, want 8192", cfg.Models["org/model"].Context)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for debounced reload")
	}

	// No second reload should arrive within a short window.
	select {
	case cfg, ok := <-ch:
		if ok {
			t.Errorf("expected only one reload, got second: %+v", cfg)
		}
	case <-time.After(300 * time.Millisecond):
		// correct: only one reload emitted
	}
}

func TestWatch_ContextCancellation(t *testing.T) {
	setupWatchDir(t, validWatchYAML)

	ctx, cancel := context.WithCancel(context.Background())

	ch, err := Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	cancel()

	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected channel to be closed after context cancellation")
		}
	case <-time.After(time.Second):
		t.Fatal("timeout: channel not closed after context cancellation")
	}
}

func TestWatch_WriteRenamePattern(t *testing.T) {
	// Simulate the editor write-rename pattern:
	// write new content to a temp file, then atomically rename it over the
	// config file (as vim, sed -i, etc. do). The old inode is deleted; a
	// directory-level watch must still detect the change.
	dir, path := setupWatchDir(t, validWatchYAML)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, err := Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	tmp := filepath.Join(dir, "config.yaml.tmp")
	if err := os.WriteFile(tmp, []byte(updatedWatchYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatal(err)
	}

	select {
	case cfg := <-ch:
		if cfg == nil {
			t.Fatal("received nil config")
		}
		if cfg.Models["org/model"].Context != 8192 {
			t.Errorf("context: got %d, want 8192", cfg.Models["org/model"].Context)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timeout: write-rename pattern not detected")
	}
}
