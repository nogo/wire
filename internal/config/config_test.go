package config

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
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
	if cfg.DrainTimeout != Duration(30*time.Second) {
		t.Errorf("drain_timeout default: got %v, want 30s", cfg.DrainTimeout)
	}
	if cfg.QueueDepth != 200 {
		t.Errorf("queue_depth default: got %d, want 200", cfg.QueueDepth)
	}
}

func TestLoad_ExplicitValues(t *testing.T) {
	_ = writeConfig(t, `
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
	path := writeConfig(t, `
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

func TestLoad_DrainTimeoutAndQueueDepth(t *testing.T) {
	path := writeConfig(t, `
drain_timeout: "1m"
queue_depth: 500
models:
  "org/model":
    context: 2048
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.DrainTimeout != Duration(time.Minute) {
		t.Errorf("drain_timeout: got %v, want 1m", cfg.DrainTimeout)
	}
	if cfg.QueueDepth != 500 {
		t.Errorf("queue_depth: got %d, want 500", cfg.QueueDepth)
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
  - "org/model-A"
  - "org/model-B"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.Warmup) != 1 || cfg.Warmup[0] != "org/model-A" {
		t.Errorf("unmatched warmup entry should be dropped, got: %v", cfg.Warmup)
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

func TestLoad_KVType(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    kv_type: "q8_0"
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	p := cfg.Models["org/model"]
	if p.KVType != "q8_0" {
		t.Errorf("kv_type: got %q, want %q", p.KVType, "q8_0")
	}
}

func TestLoad_KVTypeInvalid(t *testing.T) {
	path := writeConfig(t, `
models:
  "org/model":
    kv_type: "bf16"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	if !contains(err.Error(), "bf16") {
		t.Errorf("error should mention invalid value, got: %v", err)
	}
}

func TestLoad_NoKVType(t *testing.T) {
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
	p := cfg.Models["org/model"]
	if p.KVType != "" {
		t.Errorf("expected empty kv_type, got: %q", p.KVType)
	}
}

func TestLoad_ModelDirs(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, `
model_dirs:
  - `+dir+`
models:
  "org/model":
    context: 2048
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.ModelDirs) != 1 {
		t.Fatalf("model_dirs: got %d entries, want 1", len(cfg.ModelDirs))
	}
	if cfg.ModelDirs[0] != dir {
		t.Errorf("model_dirs[0]: got %q, want %q", cfg.ModelDirs[0], dir)
	}
}

func TestLoad_Providers(t *testing.T) {
	t.Setenv("TEST_API_KEY", "sk-test-123")
	path := writeConfig(t, `
providers:
  openrouter:
    api_key: ${TEST_API_KEY}
  custom:
    base_url: https://custom.api.com
    api_key: literal-key
models:
  "claude-sonnet":
    provider: openrouter
    upstream_model: anthropic/claude-3.5-sonnet
    capabilities: [chat, code, tools]
  "custom-model":
    provider: custom
    upstream_model: my-model
  "local-model":
    context: 4096
    gpu_layers: 20
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Provider env var expansion.
	if cfg.Providers["openrouter"].APIKey != "sk-test-123" {
		t.Errorf("openrouter api_key: got %q, want %q", cfg.Providers["openrouter"].APIKey, "sk-test-123")
	}

	// Default base URL for known provider.
	if cfg.Providers["openrouter"].BaseURL != "https://openrouter.ai/api" {
		t.Errorf("openrouter base_url: got %q, want default", cfg.Providers["openrouter"].BaseURL)
	}

	// Explicit base URL preserved.
	if cfg.Providers["custom"].BaseURL != "https://custom.api.com" {
		t.Errorf("custom base_url: got %q, want %q", cfg.Providers["custom"].BaseURL, "https://custom.api.com")
	}

	// Remote model fields.
	cs := cfg.Models["claude-sonnet"]
	if !cs.IsRemote() {
		t.Error("claude-sonnet should be remote")
	}
	if cs.Provider != "openrouter" {
		t.Errorf("provider: got %q, want %q", cs.Provider, "openrouter")
	}
	if cs.UpstreamModel != "anthropic/claude-3.5-sonnet" {
		t.Errorf("upstream_model: got %q", cs.UpstreamModel)
	}
	if len(cs.Capabilities) != 3 || cs.Capabilities[0] != "chat" {
		t.Errorf("capabilities: got %v", cs.Capabilities)
	}

	// Slots default should NOT be applied to remote models.
	if cs.Slots != 0 {
		t.Errorf("remote model slots: got %d, want 0", cs.Slots)
	}

	// Local model is not remote.
	lm := cfg.Models["local-model"]
	if lm.IsRemote() {
		t.Error("local-model should not be remote")
	}
	if lm.Slots != 1 {
		t.Errorf("local model slots default: got %d, want 1", lm.Slots)
	}
}

func TestLoad_ProviderMissing(t *testing.T) {
	path := writeConfig(t, `
models:
  "remote-model":
    provider: nonexistent
    upstream_model: some-model
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for missing provider")
	}
	if !contains(err.Error(), "nonexistent") {
		t.Errorf("error should mention provider name, got: %v", err)
	}
}

func TestLoad_ProviderEmptyAPIKey(t *testing.T) {
	t.Setenv("EMPTY_VAR", "")
	path := writeConfig(t, `
providers:
  bad:
    base_url: https://example.com
    api_key: ${EMPTY_VAR}
models:
  "model":
    provider: bad
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for empty api_key")
	}
	if !contains(err.Error(), "api_key") {
		t.Errorf("error should mention api_key, got: %v", err)
	}
}

func TestLoad_ProviderNoBaseURL(t *testing.T) {
	path := writeConfig(t, `
providers:
  unknown_provider:
    api_key: some-key
models:
  "model":
    provider: unknown_provider
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for missing base_url")
	}
	if !contains(err.Error(), "base_url") {
		t.Errorf("error should mention base_url, got: %v", err)
	}
}

func TestLoad_RemoteWithLocalFields(t *testing.T) {
	path := writeConfig(t, `
providers:
  test:
    base_url: https://example.com
    api_key: key
models:
  "model":
    provider: test
    gpu_layers: 20
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for remote model with local fields")
	}
	if !contains(err.Error(), "local-only") {
		t.Errorf("error should mention local-only fields, got: %v", err)
	}
}

func TestLoad_RemoteWithKVType(t *testing.T) {
	path := writeConfig(t, `
providers:
  test:
    base_url: https://example.com
    api_key: key
models:
  "model":
    provider: test
    kv_type: "q8_0"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for remote model with kv_type")
	}
	if !contains(err.Error(), "local-only") {
		t.Errorf("error should mention local-only fields, got: %v", err)
	}
}

func TestLoad_RemoteInWarmup(t *testing.T) {
	path := writeConfig(t, `
providers:
  test:
    base_url: https://example.com
    api_key: key
models:
  "remote-model":
    provider: test
warmup:
  - "remote-model"
`)
	setEnv(t, "WIRE_CONFIG", path)

	_, err := Load()
	if err == nil {
		t.Fatal("expected validation error for remote model in warmup")
	}
	if !contains(err.Error(), "remote") {
		t.Errorf("error should mention remote, got: %v", err)
	}
}

func TestProvider_ShouldListModels(t *testing.T) {
	t.Run("default_true", func(t *testing.T) {
		p := Provider{BaseURL: "https://example.com", APIKey: "key"}
		if !p.ShouldListModels() {
			t.Error("ShouldListModels should default to true")
		}
	})
	t.Run("explicit_true", func(t *testing.T) {
		v := true
		p := Provider{BaseURL: "https://example.com", APIKey: "key", ListModels: &v}
		if !p.ShouldListModels() {
			t.Error("ShouldListModels should be true")
		}
	})
	t.Run("explicit_false", func(t *testing.T) {
		v := false
		p := Provider{BaseURL: "https://example.com", APIKey: "key", ListModels: &v}
		if p.ShouldListModels() {
			t.Error("ShouldListModels should be false")
		}
	})
}

func TestLoad_ProviderListModels(t *testing.T) {
	t.Setenv("TEST_KEY", "key")
	path := writeConfig(t, `
providers:
  listed:
    base_url: https://example.com
    api_key: ${TEST_KEY}
    list_models: true
  unlisted:
    base_url: https://example.com
    api_key: ${TEST_KEY}
    list_models: false
  default_provider:
    base_url: https://example.com
    api_key: ${TEST_KEY}
`)
	setEnv(t, "WIRE_CONFIG", path)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !cfg.Providers["listed"].ShouldListModels() {
		t.Error("listed provider should list models")
	}
	if cfg.Providers["unlisted"].ShouldListModels() {
		t.Error("unlisted provider should not list models")
	}
	if !cfg.Providers["default_provider"].ShouldListModels() {
		t.Error("default provider should list models (nil defaults to true)")
	}
}

func TestModelParams_Equal_Remote(t *testing.T) {
	a := ModelParams{Provider: "openai", UpstreamModel: "gpt-4o", Capabilities: []string{"chat", "tools"}}
	b := ModelParams{Provider: "openai", UpstreamModel: "gpt-4o", Capabilities: []string{"chat", "tools"}}
	if !a.Equal(b) {
		t.Error("identical remote params should be equal")
	}

	c := ModelParams{Provider: "openai", UpstreamModel: "gpt-4o-mini", Capabilities: []string{"chat", "tools"}}
	if a.Equal(c) {
		t.Error("different upstream_model should not be equal")
	}

	d := ModelParams{Provider: "openai", UpstreamModel: "gpt-4o", Capabilities: []string{"chat"}}
	if a.Equal(d) {
		t.Error("different capabilities length should not be equal")
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
