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
		g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())
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
		g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())
		w := httptest.NewRecorder()
		g.handleHealth(w, httptest.NewRequest(http.MethodGet, "/health", nil))
		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503", w.Code)
		}
	})
}

func TestGateway_HandleRoute_UnknownModel(t *testing.T) {
	cfg := testConfig("org/model", 1, "")
	g := newGateway(cfg, &mockManager{}, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, &mockManager{}, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

	w := httptest.NewRecorder()
	g.handleModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Type   string `json:"type"`
			Status string `json:"status"`
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
	if resp.Data[0].Type != "local" {
		t.Errorf("type = %q, want %q", resp.Data[0].Type, "local")
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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())
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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())
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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

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
	g := newGateway(cfg, &mockManager{}, nil, nil, nil, slog.Default())

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

func TestGateway_HandleRoute_Remote(t *testing.T) {
	var receivedAuth string
	var receivedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"model":   "openai/gpt-4o",
			"choices": []any{},
		})
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Listen: ":0",
		Providers: map[string]config.Provider{
			"test": {BaseURL: upstream.URL, APIKey: "sk-test-key"},
		},
		Models: map[string]config.ModelParams{
			"smart": {Provider: "test", UpstreamModel: "openai/gpt-4o", Capabilities: []string{"chat", "tools"}},
		},
		MaxModels: 1,
	}
	// Mock manager should NOT be called for remote models.
	mgr := &mockManager{
		loadOrGetFn: func(name string) (string, func(), error) {
			t.Errorf("LoadOrGet called for remote model %q", name)
			return "", func() {}, nil
		},
	}
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"smart","messages":[{"role":"user","content":"hi"}]}`))
	r.Header.Set("Content-Type", "application/json")
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	if receivedAuth != "Bearer sk-test-key" {
		t.Errorf("Authorization = %q, want %q", receivedAuth, "Bearer sk-test-key")
	}

	// Verify model field was rewritten to upstream_model.
	var upstreamReq struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(receivedBody, &upstreamReq); err != nil {
		t.Fatalf("unmarshal upstream body: %v", err)
	}
	if upstreamReq.Model != "openai/gpt-4o" {
		t.Errorf("upstream model = %q, want %q", upstreamReq.Model, "openai/gpt-4o")
	}

	// Verify X-Wire-Model response header.
	if got := w.Header().Get("X-Wire-Model"); got != "openai/gpt-4o" {
		t.Errorf("X-Wire-Model = %q, want %q", got, "openai/gpt-4o")
	}
}

func TestGateway_HandleRoute_Remote_SSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected Flusher")
		}
		for _, chunk := range []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
			"data: [DONE]\n\n",
		} {
			w.Write([]byte(chunk)) //nolint:errcheck
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Listen: ":0",
		Providers: map[string]config.Provider{
			"test": {BaseURL: upstream.URL, APIKey: "key"},
		},
		Models: map[string]config.ModelParams{
			"stream-model": {Provider: "test"},
		},
		MaxModels: 1,
	}
	g := newGateway(cfg, &mockManager{}, nil, nil, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"stream-model","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "data: ") {
		t.Errorf("expected SSE data in response, got: %s", body)
	}
}

func TestGateway_Models_WithRemote(t *testing.T) {
	cfg := &config.Config{
		Listen: ":0",
		Providers: map[string]config.Provider{
			"openrouter": {BaseURL: "https://openrouter.ai/api", APIKey: "key"},
		},
		Models: map[string]config.ModelParams{
			"local-model":  {Slots: 1, Capabilities: []string{"chat"}},
			"remote-model": {Provider: "openrouter", UpstreamModel: "openai/gpt-4o", Capabilities: []string{"chat", "tools", "vision"}},
		},
		MaxModels: 1,
	}
	mgr := &mockManager{
		models: []monitor.ModelInfo{
			{ID: "local-model", Status: "loaded"},
		},
	}
	g := newGateway(cfg, mgr, nil, nil, nil, slog.Default())

	w := httptest.NewRecorder()
	g.handleModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Data []struct {
			ID           string   `json:"id"`
			Type         string   `json:"type"`
			Status       string   `json:"status"`
			Capabilities []string `json:"capabilities"`
			Provider     string   `json:"provider"`
		} `json:"data"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("data length = %d, want 2", len(resp.Data))
	}

	// Find local and remote entries.
	var local, remote bool
	for _, m := range resp.Data {
		switch m.ID {
		case "local-model":
			local = true
			if m.Type != "local" {
				t.Errorf("local-model type = %q, want local", m.Type)
			}
			if m.Provider != "" {
				t.Errorf("local-model provider = %q, want empty", m.Provider)
			}
			if len(m.Capabilities) != 1 || m.Capabilities[0] != "chat" {
				t.Errorf("local capabilities = %v", m.Capabilities)
			}
		case "remote-model":
			remote = true
			if m.Type != "remote" {
				t.Errorf("remote-model type = %q, want remote", m.Type)
			}
			if m.Provider != "openrouter" {
				t.Errorf("remote provider = %q, want openrouter", m.Provider)
			}
			if m.Status != "available" {
				t.Errorf("remote status = %q, want %q", m.Status, "available")
			}
			if len(m.Capabilities) != 3 {
				t.Errorf("remote capabilities = %v", m.Capabilities)
			}
		}
	}
	if !local {
		t.Error("missing local-model in response")
	}
	if !remote {
		t.Error("missing remote-model in response")
	}
}

func TestGateway_HandleRoute_Remote_WithTracker(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"gpt-4o","choices":[]}`)) //nolint:errcheck
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Listen: ":0",
		Providers: map[string]config.Provider{
			"test": {BaseURL: upstream.URL, APIKey: "key"},
		},
		Models: map[string]config.ModelParams{
			"tracked": {Provider: "test", UpstreamModel: "gpt-4o"},
		},
		MaxModels: 1,
	}
	tracker := monitor.NewRemoteTracker()
	g := newGateway(cfg, &mockManager{}, nil, tracker, nil, slog.Default())

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"tracked","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	snap := tracker.Snapshot()
	stats, ok := snap["tracked"]
	if !ok {
		t.Fatal("expected tracker to have 'tracked' entry")
	}
	if stats.Requests != 1 {
		t.Errorf("requests = %d, want 1", stats.Requests)
	}
	if stats.Provider != "test" {
		t.Errorf("provider = %q, want %q", stats.Provider, "test")
	}
	if stats.InFlight != 0 {
		t.Errorf("in_flight = %d, want 0 (request completed)", stats.InFlight)
	}
}

// TestGateway_UpdateConfig verifies that UpdateConfig causes new requests to
// use the updated model patterns: the old model returns 404 and the new model
// routes successfully.
func TestGateway_UpdateConfig(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"org/new-model","choices":[]}`)) //nolint:errcheck
	}))
	defer backend.Close()

	oldCfg := &config.Config{
		Listen:    ":0",
		Models:    map[string]config.ModelParams{"org/old-model": {Slots: 1}},
		Warmup:    []string{"org/old-model"},
		MaxModels: 1,
	}
	newCfg := &config.Config{
		Listen:    ":0",
		Models:    map[string]config.ModelParams{"org/new-model": {Slots: 1}},
		Warmup:    []string{"org/new-model"},
		MaxModels: 1,
	}

	mgr := &mockManager{
		addrs:  map[string]string{"org/new-model": backend.URL},
		loaded: map[string]bool{"org/new-model": true},
	}
	g := newGateway(oldCfg, mgr, nil, nil, nil, slog.Default())

	// Before update: old-model matches, new-model does not.
	w := httptest.NewRecorder()
	g.handleRoute(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/new-model","messages":[]}`)))
	if w.Code != http.StatusNotFound {
		t.Errorf("before UpdateConfig: status = %d, want 404", w.Code)
	}

	g.UpdateConfig(newCfg)

	// After update: new-model matches and proxies successfully.
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/new-model","messages":[]}`))
	g.handleRoute(w2, r2)
	if w2.Code != http.StatusOK {
		t.Errorf("after UpdateConfig: status = %d, want 200", w2.Code)
	}

	// Old model no longer matches.
	w3 := httptest.NewRecorder()
	g.handleRoute(w3, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"org/old-model","messages":[]}`)))
	if w3.Code != http.StatusNotFound {
		t.Errorf("after UpdateConfig: old model status = %d, want 404", w3.Code)
	}
}

func TestGateway_HandleRoute_DiscoveredModel(t *testing.T) {
	var receivedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"openai/gpt-4o","choices":[]}`)) //nolint:errcheck
	}))
	defer upstream.Close()

	// Discovery mock: serves a model list with one model.
	discoverySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"data": []map[string]string{
				{"id": "openai/gpt-4o", "owned_by": "openai"},
				{"id": "anthropic/claude-3.5-sonnet", "owned_by": "anthropic"},
			},
		})
	}))
	defer discoverySrv.Close()

	// Config has no model entries for gpt-4o, but has the provider.
	cfg := &config.Config{
		Listen: ":0",
		Providers: map[string]config.Provider{
			"test-prov": {BaseURL: upstream.URL, APIKey: "key"},
		},
		Models:    map[string]config.ModelParams{},
		MaxModels: 1,
	}

	// Create real discovery pointing at our mock.
	discovery := monitor.NewModelDiscovery([]monitor.ProviderEndpoint{{
		Name:    "test-prov",
		BaseURL: discoverySrv.URL,
		APIKey:  "key",
	}}, slog.Default())
	discovery.Start()
	defer discovery.Stop()

	g := newGateway(cfg, &mockManager{}, nil, nil, discovery, slog.Default())

	// Request for a discovered model should route to the provider.
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"openai/gpt-4o","messages":[]}`))
	g.handleRoute(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	// Model name should NOT be rewritten (no upstream_model set).
	if !strings.Contains(string(receivedBody), `"model":"openai/gpt-4o"`) {
		t.Errorf("body should contain original model name, got: %s", receivedBody)
	}

	// Unknown model not in config or discovery should still 404.
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"totally-unknown","messages":[]}`))
	g.handleRoute(w2, r2)
	if w2.Code != http.StatusNotFound {
		t.Errorf("unknown model status = %d, want 404", w2.Code)
	}
}

func TestGateway_Models_WithDiscovery(t *testing.T) {
	discoverySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"data": []map[string]string{
				{"id": "discovered-model", "owned_by": "some-org"},
			},
		})
	}))
	defer discoverySrv.Close()

	cfg := &config.Config{
		Listen:    ":0",
		Models:    map[string]config.ModelParams{},
		MaxModels: 1,
	}
	discovery := monitor.NewModelDiscovery([]monitor.ProviderEndpoint{{
		Name: "prov", BaseURL: discoverySrv.URL, APIKey: "key",
	}}, slog.Default())
	discovery.Start()
	defer discovery.Stop()

	g := newGateway(cfg, &mockManager{}, nil, nil, discovery, slog.Default())

	w := httptest.NewRecorder()
	g.handleModels(w, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	var resp struct {
		Data []struct {
			ID       string `json:"id"`
			Type     string `json:"type"`
			Status   string `json:"status"`
			Provider string `json:"provider"`
		} `json:"data"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}

	found := false
	for _, m := range resp.Data {
		if m.ID == "discovered-model" {
			found = true
			if m.Type != "remote" {
				t.Errorf("discovered model type = %q, want remote", m.Type)
			}
			if m.Provider != "prov" {
				t.Errorf("provider = %q, want %q", m.Provider, "prov")
			}
			if m.Status != "available" {
				t.Errorf("status = %q, want %q", m.Status, "available")
			}
		}
	}
	if !found {
		t.Error("discovered-model not in /v1/models response")
	}
}
