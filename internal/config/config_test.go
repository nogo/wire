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

func TestLoad_ValidMixed(t *testing.T) {
	path := writeConfig(t, `
listen: ":9090"
backends:
  local:
    model: "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
    context: 4096
    gpu_layers: 35
  openai:
    url: "https://api.openai.com/v1"
    api_key: "sk-test"
    provider: "openai"
peers:
  - "10.0.0.2:8080"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Listen != ":9090" {
		t.Errorf("listen: got %q, want %q", cfg.Listen, ":9090")
	}
	if len(cfg.Backends) != 2 {
		t.Errorf("backends: got %d, want 2", len(cfg.Backends))
	}

	local := cfg.Backends["local"]
	if !local.IsManaged() {
		t.Error("local backend should be managed")
	}
	if local.IsExternal() {
		t.Error("local backend should not be external")
	}
	if local.Model != "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M" {
		t.Errorf("local.model: got %q", local.Model)
	}

	ext := cfg.Backends["openai"]
	if !ext.IsExternal() {
		t.Error("openai backend should be external")
	}
	if ext.Provider != "openai" {
		t.Errorf("openai.provider: got %q, want %q", ext.Provider, "openai")
	}

	if len(cfg.Peers) != 1 || cfg.Peers[0] != "10.0.0.2:8080" {
		t.Errorf("peers: got %v", cfg.Peers)
	}
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
	// Error message must mention the path.
	msg := err.Error()
	for i := range len(path) {
		_ = i // use for range n idiom below
	}
	for range len(path) {
		break // confirm for range n compiles; actual path check below
	}
	if !contains(msg, path) {
		t.Errorf("error %q does not mention path %q", msg, path)
	}
}

func TestLoad_BothModelAndURL(t *testing.T) {
	path := writeConfig(t, `
backends:
  bad:
    model: "some/model"
    url: "http://localhost:11434"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	ve, ok := errors.AsType[*ValidationError](err)
	if !ok {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "backends.bad" {
		t.Errorf("field: got %q, want %q", ve.Field, "backends.bad")
	}
}

func TestLoad_NeitherModelNorURL(t *testing.T) {
	path := writeConfig(t, `
backends:
  empty:
    context: 2048
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	ve, ok := errors.AsType[*ValidationError](err)
	if !ok {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "backends.empty" {
		t.Errorf("field: got %q, want %q", ve.Field, "backends.empty")
	}
}

func TestLoad_Defaults(t *testing.T) {
	path := writeConfig(t, `
backends:
  ext:
    url: "http://localhost:11434"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Listen != ":8080" {
		t.Errorf("listen default: got %q, want %q", cfg.Listen, ":8080")
	}
	if cfg.Backends["ext"].Provider != "openai" {
		t.Errorf("provider default: got %q, want %q", cfg.Backends["ext"].Provider, "openai")
	}
}

func TestLoad_XDGConfigHome(t *testing.T) {
	xdg := t.TempDir()
	confDir := filepath.Join(xdg, "wire")
	if err := os.MkdirAll(confDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(confDir, "config.yaml"), []byte(`
backends:
  ext:
    url: "http://localhost:11434"
`), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("XDG_CONFIG_HOME", xdg)
	t.Setenv("WIRE_CONFIG", "") // clear override

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Listen != ":8080" {
		t.Errorf("listen: got %q", cfg.Listen)
	}
}

func TestLoad_MonitorTrue(t *testing.T) {
	path := writeConfig(t, `
backends:
  ext:
    url: "http://localhost:11434"
monitor: true
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !cfg.Monitor {
		t.Error("monitor: got false, want true")
	}
}

func TestLoad_SlotsField(t *testing.T) {
	path := writeConfig(t, `
backends:
  multi:
    model: "some/model"
    slots: 4
  single:
    model: "other/model"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Backends["multi"].Slots != 4 {
		t.Errorf("multi.slots: got %d, want 4", cfg.Backends["multi"].Slots)
	}
	// Unset slots should default to 1.
	if cfg.Backends["single"].Slots != 1 {
		t.Errorf("single.slots default: got %d, want 1", cfg.Backends["single"].Slots)
	}
}

func TestLoad_MonitorDefault(t *testing.T) {
	path := writeConfig(t, `
backends:
  ext:
    url: "http://localhost:11434"
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
