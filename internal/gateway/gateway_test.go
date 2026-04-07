package gateway

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nogo/wire/internal/config"
)

// mockManager is a test double for backendManager.
type mockManager struct {
	addrs   map[string]string
	healthy map[string]bool
}

func (m *mockManager) Addr(name string) (string, bool) {
	v, ok := m.addrs[name]
	return v, ok
}

func (m *mockManager) Healthy(name string) bool {
	return m.healthy[name]
}

func TestModelAliases(t *testing.T) {
	aliases := modelAliases("Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M")
	want := map[string]bool{
		"Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M": true,
		"Qwen3-Coder-30B-A3B-Instruct-GGUF":             true,
		"Qwen3-Coder-30B-A3B-Instruct-Q4_K_M":           true,
		"Qwen3-Coder-30B-A3B-Instruct":                  true,
	}
	got := make(map[string]bool)
	for _, a := range aliases {
		got[a] = true
	}
	for k := range want {
		if !got[k] {
			t.Errorf("missing alias %q", k)
		}
	}
}

func TestRouteTable_Resolve(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"chat": {Model: "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"},
		},
	}
	rt := buildRouteTable(cfg)

	for _, model := range []string{
		"chat",
		"Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M",
		"Qwen3-Coder-30B-A3B-Instruct-GGUF",
		"Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
		"Qwen3-Coder-30B-A3B-Instruct",
	} {
		name, err := rt.resolve(model)
		if err != nil {
			t.Errorf("resolve(%q): unexpected error: %v", model, err)
			continue
		}
		if name != "chat" {
			t.Errorf("resolve(%q) = %q, want %q", model, name, "chat")
		}
	}

	if _, err := rt.resolve("unknown-model"); err == nil {
		t.Error("resolve unknown model: expected error, got nil")
	}
}

func TestGateway_Health(t *testing.T) {
	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"chat": {Model: "owner/model-GGUF:Q4"},
		},
	}

	t.Run("healthy", func(t *testing.T) {
		mgr := &mockManager{
			addrs:   map[string]string{"chat": "http://localhost:9999"},
			healthy: map[string]bool{"chat": true},
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
			healthy: map[string]bool{"chat": false},
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
	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"chat": {Model: "owner/model-GGUF:Q4"},
		},
	}
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"no-such-model","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestGateway_HandleRoute_MissingModel(t *testing.T) {
	cfg := &config.Config{
		Listen:   ":0",
		Backends: map[string]config.Backend{"chat": {Model: "owner/model-GGUF:Q4"}},
	}
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestGateway_HandleRoute_UnhealthyBackend(t *testing.T) {
	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"chat": {Model: "owner/model-GGUF:Q4"},
		},
	}
	mgr := &mockManager{
		addrs:   map[string]string{"chat": "http://localhost:9999"},
		healthy: map[string]bool{"chat": false},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chat","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
}

func TestGateway_HandleRoute_ProxyAndRewrite(t *testing.T) {
	// Mock backend returns a response with the internal llama-server model name.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"model":   "testmodel-Q4",
			"choices": []any{},
		})
	}))
	defer backend.Close()

	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"chat": {Model: "owner/testmodel-GGUF:Q4"},
		},
	}
	mgr := &mockManager{
		addrs:   map[string]string{"chat": backend.URL},
		healthy: map[string]bool{"chat": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chat","messages":[{"role":"user","content":"hi"}]}`))
	r.Header.Set("Content-Type", "application/json")
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got := resp["model"]; got != "chat" {
		t.Errorf("model in response = %q, want %q", got, "chat")
	}
}

func TestGateway_HandleRoute_BodyPreserved(t *testing.T) {
	// Verify the full request body reaches the backend unchanged.
	var receivedBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"m","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"m": {URL: backend.URL},
		},
	}
	mgr := &mockManager{
		addrs:   map[string]string{"m": backend.URL},
		healthy: map[string]bool{"m": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	original := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 1000) + `"}]}`
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(original))
	g.handleRoute(w, r)

	if string(receivedBody) != original {
		t.Errorf("body not preserved: got len=%d, want len=%d", len(receivedBody), len(original))
	}
}

func TestGateway_Models(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"object": "list",
			"data": []map[string]any{
				{"id": "internal-llama-name", "object": "model", "created": int64(0), "owned_by": "llama"},
			},
		})
	}))
	defer backend.Close()

	cfg := &config.Config{
		Listen: ":0",
		Backends: map[string]config.Backend{
			"chat": {Model: "owner/model-GGUF:Q4"},
		},
	}
	mgr := &mockManager{
		addrs:   map[string]string{"chat": backend.URL},
		healthy: map[string]bool{"chat": true},
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
			ID string `json:"id"`
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
	if resp.Data[0].ID != "chat" {
		t.Errorf("model id = %q, want %q", resp.Data[0].ID, "chat")
	}
}

func TestGateway_NoBackends(t *testing.T) {
	cfg := &config.Config{Listen: ":0", Backends: map[string]config.Backend{}}
	g := newGateway(cfg, &mockManager{}, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"x"}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
}
