package config

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func setEnv(t *testing.T, key, value string) {
	t.Helper()
	t.Setenv(key, value)
}

func TestLoad_MissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nonexistent.yaml")
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected ErrNotExist in chain, got: %v", err)
	}
	if !contains(err.Error(), path) {
		t.Errorf("error %q does not mention path %q", err.Error(), path)
	}
}

func TestLoad_Defaults(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    context: 4096
warmup:
  - "org/model"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Listen != ":9480" {
		t.Errorf("listen default: got %q, want %q", cfg.Listen, ":9480")
	}
	if cfg.MaxModels != 1 {
		t.Errorf("max_models default: got %d, want 1", cfg.MaxModels)
	}
	if cfg.Models["org/model"].Slots != 1 {
		t.Errorf("slots default: got %d, want 1", cfg.Models["org/model"].Slots)
	}
}

func TestLoad_ExplicitValues(t *testing.T) {
	path := writeConfig(t, `
listen: ":9090"
monitor: true
max_models: 3
models:
  "org/model-A":
    context: 8192
    gpu_layers: 35
    slots: 4
    pooling: "mean"
  "org/model-B":
    context: 2048
warmup:
  - "org/model-A"
  - "org/model-B"
  - "org/model-C"
`)
	// max_models=3 and 3 warmup entries — valid, but model-C has no pattern match.
	// Use a valid config instead.
	path = writeConfig(t, `
listen: ":9090"
monitor: true
max_models: 2
models:
  "org/model-A":
    context: 8192
    gpu_layers: 35
    slots: 4
    pooling: "mean"
  "org/model-B":
    context: 2048
warmup:
  - "org/model-A"
  - "org/model-B"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Listen != ":9090" {
		t.Errorf("listen: got %q, want %q", cfg.Listen, ":9090")
	}
	if !cfg.Monitor {
		t.Error("monitor: got false, want true")
	}
	if cfg.MaxModels != 2 {
		t.Errorf("max_models: got %d, want 2", cfg.MaxModels)
	}

	a := cfg.Models["org/model-A"]
	if a.Context != 8192 {
		t.Errorf("model-A context: got %d, want 8192", a.Context)
	}
	if a.GPULayers != 35 {
		t.Errorf("model-A gpu_layers: got %d, want 35", a.GPULayers)
	}
	if a.Slots != 4 {
		t.Errorf("model-A slots: got %d, want 4", a.Slots)
	}
	if a.Pooling != "mean" {
		t.Errorf("model-A pooling: got %q, want %q", a.Pooling, "mean")
	}

	if len(cfg.Warmup) != 2 {
		t.Errorf("warmup len: got %d, want 2", len(cfg.Warmup))
	}
}

func TestLoad_GlobMatch(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/Qwen3-*":
    context: 4096
    gpu_layers: 20
warmup:
  - "org/Qwen3-8B-Instruct"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	params, ok := cfg.MatchModel("org/Qwen3-8B-Instruct")
	if !ok {
		t.Fatal("MatchModel: expected match for org/Qwen3-8B-Instruct")
	}
	if params.Context != 4096 {
		t.Errorf("context: got %d, want 4096", params.Context)
	}

	_, ok = cfg.MatchModel("org/Llama-3-8B")
	if ok {
		t.Error("MatchModel: expected no match for org/Llama-3-8B")
	}
}

func TestLoad_GlobMatch_NoMatch(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    context: 4096
warmup:
  - "org/model"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, ok := cfg.MatchModel("other/model")
	if ok {
		t.Error("MatchModel: expected no match for other/model")
	}
}

func TestLoad_WarmupNoPatternMatch(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model-A":
    context: 4096
warmup:
  - "org/model-B"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	if !contains(err.Error(), "org/model-B") {
		t.Errorf("error should mention org/model-B, got: %v", err)
	}
}

func TestLoad_WarmupMultiplePatternMatch(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model-*":
    context: 4096
  "org/*":
    context: 2048
warmup:
  - "org/model-A"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for multiple pattern matches, got nil")
	}
	if !contains(err.Error(), "org/model-A") {
		t.Errorf("error should mention org/model-A, got: %v", err)
	}
}

func TestLoad_MaxModelsExceeded(t *testing.T) {
	path := writeConfig(t, `
max_models: 1
models:
  "org/model-A":
    context: 4096
  "org/model-B":
    context: 2048
warmup:
  - "org/model-A"
  - "org/model-B"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for max_models exceeded, got nil")
	}
	if !contains(err.Error(), "max_models") {
		t.Errorf("error should mention max_models, got: %v", err)
	}
}

func TestLoad_SlotsDefault(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/multi":
    slots: 4
  "org/single":
    context: 2048
warmup:
  - "org/multi"
  - "org/single"
max_models: 2
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Models["org/multi"].Slots != 4 {
		t.Errorf("multi.slots: got %d, want 4", cfg.Models["org/multi"].Slots)
	}
	if cfg.Models["org/single"].Slots != 1 {
		t.Errorf("single.slots default: got %d, want 1", cfg.Models["org/single"].Slots)
	}
}

func TestLoad_MonitorDefault(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    context: 2048
warmup:
  - "org/model"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Monitor {
		t.Error("monitor default: got true, want false")
	}
}

func TestLoad_XDGConfigHome(t *testing.T) {
	xdg := t.TempDir()
	confDir := filepath.Join(xdg, "wire")
	if err := os.MkdirAll(confDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(confDir, "config.yaml"), []byte(`
models:
  "org/model":
    context: 2048
warmup:
  - "org/model"
`), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("XDG_CONFIG_HOME", xdg)
	t.Setenv("WIRE_CONFIG", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Listen != ":9480" {
		t.Errorf("listen: got %q, want %q", cfg.Listen, ":9480")
	}
}

func TestLoad_EmptyWarmup(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    context: 2048
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.Warmup) != 0 {
		t.Errorf("warmup: got %v, want empty", cfg.Warmup)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		func() bool {
			for i := range len(s) - len(sub) + 1 {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
			return false
		}())
}
