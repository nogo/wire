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
	addrs  map[string]string
	loaded map[string]bool
	models []monitor.ModelInfo

	// Optional override for LoadOrGet behaviour.
	loadOrGetFn func(name string) (string, func(), error)
}

func (m *mockManager) Loaded(name string) bool {
	return m.loaded[name]
}

func (m *mockManager) LoadOrGet(_ context.Context, name string, _ config.ModelParams) (string, func(), error) {
	if m.loadOrGetFn != nil {
		return m.loadOrGetFn(name)
	}
	// Default: return addr if healthy, error otherwise.
	if !m.loaded[name] {
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

	t.Run("loaded", func(t *testing.T) {
		mgr := &mockManager{
			addrs:  map[string]string{"org/model": "http://localhost:9999"},
			loaded: map[string]bool{"org/model": true},
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
			addrs:  map[string]string{},
			loaded: map[string]bool{"org/model": false},
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
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
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
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
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
		addrs:  map[string]string{"org/Qwen3-8B": backend.URL},
		loaded: map[string]bool{"org/Qwen3-8B": true},
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
			{ID: "org/model", Status: "loaded"},
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
	if resp.Data[0].Status != "loaded" {
		t.Errorf("status = %q, want %q", resp.Data[0].Status, "loaded")
	}
}

// mockHub implements nctxProvider for tests.
type mockHub struct {
	nctx map[string]int
}

func (m *mockHub) NCtx(model string) int {
	return m.nctx[model]
}

func TestHandleRoute_contextGuard_reject(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/model","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())
	g.hub = &mockHub{nctx: map[string]int{"org/model": 100}}

	// 400 chars of content → 100 estimated tokens > 90% of 100 (threshold = 90).
	longContent := strings.Repeat("x", 400)
	body := `{"model":"org/model","messages":[{"role":"user","content":"` + longContent + `"}]}`
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	g.handleRoute(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	errObj, _ := resp["error"].(map[string]any)
	if errObj["code"] != "context_length_exceeded" {
		t.Errorf("error code = %v, want context_length_exceeded", errObj["code"])
	}
}

func TestHandleRoute_contextGuard_pass(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/model","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())
	g.hub = &mockHub{nctx: map[string]int{"org/model": 100}}

	// "hello" = 5 chars → 1 estimated token, well below threshold of 90.
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/model","messages":[{"role":"user","content":"hello"}]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

func TestHandleRoute_contextGuard_noHub(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/model","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
	}
	// hub is nil — large requests must pass through unchanged.
	g := newGateway(cfg, mgr, nil, slog.Default())

	longContent := strings.Repeat("x", 10000)
	body := `{"model":"org/model","messages":[{"role":"user","content":"` + longContent + `"}]}`
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (guard must not fire when hub is nil)", w.Code)
	}
}

func TestHandleRoute_upstreamContextLengthExceeded(t *testing.T) {
	// Simulate llama-server returning 500 with context_length_exceeded error.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"context too long","type":"invalid_request_error","code":"context_length_exceeded"}}`)) //nolint:errcheck
	}))
	defer backend.Close()

	cfg := testConfig("org/model", 1, backend.URL)
	mgr := &mockManager{
		addrs:  map[string]string{"org/model": backend.URL},
		loaded: map[string]bool{"org/model": true},
	}
	g := newGateway(cfg, mgr, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/model","messages":[{"role":"user","content":"hello"}]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	errObj, _ := resp["error"].(map[string]any)
	if errObj["code"] != "context_length_exceeded" {
		t.Errorf("error code = %v, want context_length_exceeded", errObj["code"])
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name string
		body string
		want int
	}{
		{
			name: "chat single string content",
			body: `{"messages":[{"role":"user","content":"` + strings.Repeat("a", 400) + `"}]}`,
			want: 100, // 400 chars / 4
		},
		{
			name: "chat array content",
			body: `{"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("b", 200) + `"}]}]}`,
			want: 50, // 200 chars / 4
		},
		{
			name: "embeddings string input",
			body: `{"model":"embed","input":"` + strings.Repeat("c", 80) + `"}`,
			want: 20, // 80 chars / 4
		},
		{
			name: "embeddings array input",
			body: `{"model":"embed","input":["` + strings.Repeat("d", 40) + `","` + strings.Repeat("e", 40) + `"]}`,
			want: 20, // 80 chars / 4
		},
		{
			name: "invalid json",
			body: `not json`,
			want: 0,
		},
		{
			name: "empty messages",
			body: `{"messages":[]}`,
			want: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := estimateTokens([]byte(tc.body)); got != tc.want {
				t.Errorf("estimateTokens = %d, want %d", got, tc.want)
			}
		})
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
