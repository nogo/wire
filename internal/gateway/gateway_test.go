package gateway

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
)

// mockManager is a test double for backendManager.
type mockManager struct {
	addrs   map[string]string
	healthy map[string]bool
	models  []monitor.ModelInfo

	// Optional override for LoadOrGet behaviour.
	loadOrGetFn func(name string) (string, func(), error)
}

func (m *mockManager) Healthy(name string) bool {
	return m.healthy[name]
}

func (m *mockManager) LoadOrGet(_ context.Context, name string, _ config.ModelParams) (string, func(), error) {
	if m.loadOrGetFn != nil {
		return m.loadOrGetFn(name)
	}
	// Default: return addr if healthy, error otherwise.
	if !m.healthy[name] {
		return "", func() {}, context.DeadlineExceeded // triggers 504 — use a different sentinel
	}
	addr, ok := m.addrs[name]
	if !ok {
		return "", func() {}, &unavailableError{name}
	}
	return addr, func() {}, nil
}

func (m *mockManager) ListModels() []monitor.ModelInfo {
	return m.models
}

// unavailableError signals that a model is known but not loaded.
type unavailableError struct{ model string }

func (e *unavailableError) Error() string { return "model not loaded: " + e.model }

// testConfig builds a minimal Config for tests.
// modelName is used as both an exact-match pattern in Models and in Warmup.
func testConfig(modelName string, slots int, _ string) *config.Config {
	return &config.Config{
		Listen: ":0",
		Models: map[string]config.ModelParams{
			modelName: {Slots: slots},
		},
		Warmup:    []string{modelName},
		MaxModels: 1,
	}
}

func TestGateway_Health(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	cfg.Warmup = []string{"org/model"}

	t.Run("healthy", func(t *testing.T) {
		mgr := &mockManager{
			addrs:   map[string]string{"org/model": "http://localhost:9999"},
			healthy: map[string]bool{"org/model": true},
		}
		g := newGateway(cfg, mgr, nil, slog.Default())
		w := httptest.NewRecorder()
		g.handleHealth(w, httptest.NewRequest(http.MethodGet, "/health", nil))
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	t.Run("degraded", func(t *testing.T) {
		mgr := &mockManager{
			addrs:   map[string]string{},
			healthy: map[string]bool{"org/model": false},
		}
		g := newGateway(cfg, mgr, nil, slog.Default())
		w := httptest.NewRecorder()
		g.handleHealth(w, httptest.NewRequest(http.MethodGet, "/health", nil))
		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503", w.Code)
		}
	})
}

func TestGateway_HandleRoute_UnknownModel(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"no-such-model","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestGateway_HandleRoute_MissingModel(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestGateway_HandleRoute_ModelNotLoaded(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	// LoadOrGet returns a non-deadline error → 503.
	mgr := &mockManager{
		loadOrGetFn: func(name string) (string, func(), error) {
			return "", func() {}, &unavailableError{name}
		},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/model","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
}

func TestGateway_HandleRoute_Proxy(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"model":   "org/model",
			"choices": []any{},
		})
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:   map[string]string{"org/model": backend.URL},
		healthy: map[string]bool{"org/model": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/model","messages":[{"role":"user","content":"hi"}]}`))
	r.Header.Set("Content-Type", "application/json")
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got := resp["model"]; got != "org/model" {
		t.Errorf("model in response = %q, want %q", got, "org/model")
	}
}

func TestGateway_HandleRoute_BodyPreserved(t *testing.T) {
	var receivedBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/model","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:   map[string]string{"org/model": backend.URL},
		healthy: map[string]bool{"org/model": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	// Use empty messages so prefixHash returns 0 and no id_slot is injected,
	// allowing us to verify the body passes through unchanged.
	original := `{"model":"org/model","messages":[]}`
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(original))
	g.handleRoute(w, r)

	if string(receivedBody) != original {
		t.Errorf("body not preserved: got len=%d, want len=%d", len(receivedBody), len(original))
	}
}

func TestGateway_HandleRoute_GlobMatch(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/Qwen3-8B","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := &config.Config{
		Listen: ":0",
		Models: map[string]config.ModelParams{
			"org/Qwen3-*": {Slots: 1, Context: 4096},
		},
		Warmup:    []string{"org/Qwen3-8B"},
		MaxModels: 1,
	}
	mgr := &mockManager{
		addrs:   map[string]string{"org/Qwen3-8B": backend.URL},
		healthy: map[string]bool{"org/Qwen3-8B": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/Qwen3-8B","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	// A model not matching the glob should return 404.
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/Llama-8B","messages":[]}`))
	g.handleRoute(w2, r2)

	if w2.Code != http.StatusNotFound {
		t.Errorf("non-matching model: status = %d, want 404", w2.Code)
	}
}

func TestGateway_Models(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	mgr := &mockManager{
		models: []monitor.ModelInfo{
			{ID: "org/model", Status: "healthy"},
		},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	g.handleModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
			Status  string `json:"status"`
		} `json:"data"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Object != "list" {
		t.Errorf("object = %q, want %q", resp.Object, "list")
	}
	if len(resp.Data) != 1 {
		t.Fatalf("data length = %d, want 1", len(resp.Data))
	}
	if resp.Data[0].ID != "org/model" {
		t.Errorf("model id = %q, want %q", resp.Data[0].ID, "org/model")
	}
	if resp.Data[0].OwnedBy != "wire" {
		t.Errorf("owned_by = %q, want %q", resp.Data[0].OwnedBy, "wire")
	}
	if resp.Data[0].Status != "healthy" {
		t.Errorf("status = %q, want %q", resp.Data[0].Status, "healthy")
	}
}

func TestGateway_Models_Empty(t *testing.T) {
	cfg := &config.Config{Listen: ":0", Models: map[string]config.ModelParams{}}
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	g.handleModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Data []any `json:"data"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Data) != 0 {
		t.Errorf("expected empty data, got %v", resp.Data)
	}
}
