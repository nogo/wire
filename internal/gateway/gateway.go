package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/gateway/queue"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/process"
)

const (
	// warmupMaxAttempts is the maximum number of times each warmup model is
	// submitted through the admission queue before giving up.
	warmupMaxAttempts = 3
	// warmupRetryInterval is the delay between consecutive warmup attempts.
	// Three attempts span roughly 5 minutes (2 × 150 s = 300 s).
	warmupRetryInterval = 150 * time.Second
)

// nctxProvider wraps the NCtx lookup; *monitor.Hub satisfies this interface.
type nctxProvider interface {
	NCtx(model string) int
}

// processManager is the subset of process.Manager the Gateway depends on.
type processManager interface {
	Loaded(name string) bool
	// LoadOrGet returns a ticket for the named model, loading it on demand if
	// not already running. The caller waits on ticket.Ready() for the backend
	// address and must call ticket.Release() when the request completes.
	LoadOrGet(modelName string, params config.ModelParams) *process.LoadTicket
	// ListModels returns all known models in non-idle states.
	ListModels() []monitor.ModelInfo
}

// hfCache caches HF model IDs with a TTL to avoid repeated filesystem I/O.
type hfCache struct {
	mu    sync.Mutex
	ids   []string
	until time.Time
}

func (c *hfCache) get(ttl time.Duration) ([]string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if time.Now().Before(c.until) {
		return c.ids, true
	}
	return nil, false
}

func (c *hfCache) set(ids []string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ids = ids
	c.until = time.Now().Add(ttl)
}

// Gateway is an HTTP reverse-proxy that routes OpenAI-compatible requests to llama-server processes.
type Gateway struct {
	cfg            atomic.Pointer[config.Config]
	mgr            processManager
	q              *queue.Queue
	log            *slog.Logger
	affinity       *affinityTable
	hub            nctxProvider
	remote         *monitor.RemoteTracker
	discovery      *monitor.ModelDiscovery
	server         *http.Server
	hfCacheDir     string        // HuggingFace hub cache directory; empty disables scanning
	hfCache        hfCache       // cached HF model IDs
	warmupInterval time.Duration // 0 → warmupRetryInterval; overridden in tests
}

// New creates a Gateway backed by the given process manager.
// hub may be nil; when non-nil, /monitor and /monitor/ws routes are registered.
// remote may be nil; when non-nil, remote provider requests are tracked.
func New(cfg *config.Config, mgr *process.Manager, hub *monitor.Hub, remote *monitor.RemoteTracker, discovery *monitor.ModelDiscovery, log *slog.Logger) *Gateway {
	g := newGateway(cfg, mgr, hub, remote, discovery, log)
	if home, err := os.UserHomeDir(); err == nil {
		g.hfCacheDir = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	return g
}

func newGateway(cfg *config.Config, mgr processManager, hub *monitor.Hub, remote *monitor.RemoteTracker, discovery *monitor.ModelDiscovery, log *slog.Logger) *Gateway {
	// Build affinity table from warmup models.
	modelSlots := make(map[string]int, len(cfg.Warmup))
	for _, name := range cfg.Warmup {
		if params, ok := cfg.MatchModel(name); ok {
			modelSlots[name] = params.Slots
		}
	}

	g := &Gateway{mgr: mgr, log: log, affinity: newAffinityTable(modelSlots), remote: remote, discovery: discovery}
	g.cfg.Store(cfg)

	qDepth := cfg.QueueDepth
	if qDepth <= 0 {
		qDepth = 200
	}
	g.q = queue.New(qDepth, func(ctx context.Context, model string) (string, func(), error) {
		params, ok := g.cfg.Load().MatchModel(model)
		if !ok {
			return "", nil, fmt.Errorf("unknown model: %s", model)
		}
		t := g.mgr.LoadOrGet(model, params)
		select {
		case addr, ok := <-t.Ready():
			if !ok {
				return "", nil, t.Err()
			}
			return addr, t.Release, nil
		case <-ctx.Done():
			// Drain in background so the manager's in-flight counter stays consistent.
			go func() {
				if _, ok := <-t.Ready(); ok {
					t.Release()
				}
			}()
			return "", nil, ctx.Err()
		}
	})

	// Store hub as nctxProvider nil-safely: a nil *monitor.Hub must not become
	// a non-nil interface value, which would cause g.hub == nil checks to fail.
	if hub != nil {
		g.hub = hub
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.handleRoute)
	mux.HandleFunc("POST /v1/completions", g.handleRoute)
	mux.HandleFunc("POST /v1/embeddings", g.handleRoute)
	mux.HandleFunc("POST /v1/rerank", g.handleRoute)
	mux.HandleFunc("GET /v1/models", g.handleModels)
	mux.HandleFunc("GET /health", g.handleHealth)

	if hub != nil {
		mux.HandleFunc("GET /monitor", monitor.HandlePage)
		mux.HandleFunc("GET /monitor/ws", monitor.HandleWS(hub))
	}

	g.server = &http.Server{Addr: cfg.Listen, Handler: mux}
	return g
}

// ListenAndServe starts the HTTP server and shuts it down when ctx is cancelled.
func (g *Gateway) ListenAndServe(ctx context.Context) error {
	errCh := make(chan error, 1)
	go func() {
		if err := g.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return g.server.Shutdown(shutCtx)
	}
}

// UpdateConfig atomically replaces the gateway's config. New requests will use
// the updated model patterns and params immediately.
func (g *Gateway) UpdateConfig(cfg *config.Config) {
	g.cfg.Store(cfg)
}

// Warmup submits each model in cfg.Warmup through the gateway's admission
// queue in config order. Each model is attempted up to warmupMaxAttempts times;
// if VRAM or capacity is insufficient, the attempt is retried after a delay.
// A warning is logged for any model that fails all attempts; startup continues.
// Warmup runs until all models are processed or ctx is cancelled.
func (g *Gateway) Warmup(ctx context.Context) {
	interval := g.warmupInterval
	if interval == 0 {
		interval = warmupRetryInterval
	}
	for _, model := range g.cfg.Load().Warmup {
		if ctx.Err() != nil {
			return
		}
		g.warmupOne(ctx, model, interval)
	}
}

func (g *Gateway) warmupOne(ctx context.Context, model string, retryInterval time.Duration) {
	var lastErr error
	for attempt := 1; attempt <= warmupMaxAttempts; attempt++ {
		if attempt > 1 {
			select {
			case <-time.After(retryInterval):
			case <-ctx.Done():
				return
			}
		}

		ticket := g.q.Enqueue(model, queue.PriorityHigh, ctx)
		select {
		case _, ok := <-ticket.Ready():
			if ok {
				ticket.Release()
				g.log.Info("warmup model loaded", "model", model, "attempt", attempt)
				return
			}
			lastErr = ticket.Err()
		case <-ctx.Done():
			return
		}

		if !isWarmupRetryable(lastErr) {
			break
		}
		g.log.Warn("warmup model load attempt failed",
			"model", model, "attempt", attempt, "max_attempts", warmupMaxAttempts, "error", lastErr)
	}
	g.log.Warn("warmup model failed to load after all attempts",
		"model", model, "attempts", warmupMaxAttempts, "error", lastErr)
}

func isWarmupRetryable(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, queue.ErrQueueFull) ||
		errors.Is(err, queue.ErrModelEvicted) ||
		errors.Is(err, process.ErrModelDraining) ||
		errors.Is(err, context.DeadlineExceeded)
}

// handleRoute extracts the model field from the JSON body and proxies to the matching process.
// If the model is known but not loaded, it is started on demand; the HTTP connection
// is held open until the model becomes healthy.
func (g *Gateway) handleRoute(w http.ResponseWriter, r *http.Request) {
	// Read body once for all operations: model extraction, token estimation, slot affinity.
	allBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "failed to read body", "bad_request")
		return
	}

	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(allBody, &payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error(), "bad_request")
		return
	}
	if payload.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing model field", "bad_request")
		return
	}

	currentCfg := g.cfg.Load()
	params, ok := currentCfg.MatchModel(payload.Model)
	var snap monitor.DiscoverySnapshot
	if !ok && g.discovery != nil {
		snap = g.discovery.Snapshot()
		if dm, found := snap.Lookup(payload.Model); found {
			if _, provOk := currentCfg.Providers[dm.Provider]; provOk {
				params = config.ModelParams{Provider: dm.Provider}
				ok = true
			}
		}
	}
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "unknown model: "+payload.Model, "model_not_found")
		return
	}

	if params.IsRemote() {
		g.handleRemote(w, r, payload.Model, params, currentCfg, allBody)
		return // handleRemote handles body rewrite if needed
	}

	prio := pathPriority(r.URL.Path)
	ticket := g.q.Enqueue(payload.Model, prio, r.Context())
	var addr string
	select {
	case a, ok := <-ticket.Ready():
		if !ok {
			err := ticket.Err()
			switch {
			case errors.Is(err, queue.ErrQueueFull):
				w.Header().Set("Retry-After", "5")
				writeError(w, http.StatusServiceUnavailable, "server_error", "queue at capacity", "queue_full")
			case errors.Is(err, queue.ErrModelEvicted), errors.Is(err, process.ErrModelDraining):
				w.Header().Set("Retry-After", "10")
				writeError(w, http.StatusServiceUnavailable, "server_error", "model unavailable, retry later", "model_draining")
			case errors.Is(err, context.DeadlineExceeded):
				writeError(w, http.StatusGatewayTimeout, "server_error", "model load timed out", "load_timeout")
			default:
				writeError(w, http.StatusServiceUnavailable, "server_error", err.Error(), "model_not_loaded")
			}
			return
		}
		addr = a
	case <-r.Context().Done():
		writeError(w, http.StatusServiceUnavailable, "server_error", "request cancelled", "request_cancelled")
		return
	}
	defer ticket.Release()

	if g.hub != nil {
		if nCtx := g.hub.NCtx(payload.Model); nCtx > 0 {
			if est := estimateTokens(allBody); est > 0 && est > nCtx*9/10 {
				writeError(w, http.StatusBadRequest, "invalid_request_error",
					fmt.Sprintf("estimated %d tokens exceeds context window of %d (n_ctx)", est, nCtx),
					"context_length_exceeded")
				return
			}
		}
	}

	g.affinity.Register(payload.Model, params.Slots)

	hash := prefixHash(allBody)
	if slotID, assigned := g.affinity.Assign(payload.Model, hash); assigned {
		allBody = injectSlot(allBody, slotID)
	}
	r.Body = io.NopCloser(bytes.NewReader(allBody))
	r.ContentLength = int64(len(allBody))

	target, parseErr := url.Parse(addr)
	if parseErr != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "invalid backend URL", "internal_error")
		return
	}

	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Host = target.Host
		},
		FlushInterval:  -1,
		ModifyResponse: rewriteContextLengthError,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			g.log.Error("proxy error", "model", payload.Model, "error", err)
			writeError(w, http.StatusBadGateway, "server_error", "upstream error: "+err.Error(), "upstream_error")
		},
	}

	proxy.ServeHTTP(w, r)
}

// handleModels returns all known models with their current status.
func (g *Gateway) handleModels(w http.ResponseWriter, _ *http.Request) {
	type modelObj struct {
		ID           string   `json:"id"`
		Object       string   `json:"object"`
		Type         string   `json:"type"`
		Status       string   `json:"status,omitempty"`
		Capabilities []string `json:"capabilities,omitempty"`
		Provider     string   `json:"provider,omitempty"`
	}
	type modelsResp struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}

	models := g.Models()
	data := make([]modelObj, 0, len(models))
	for _, m := range models {
		data = append(data, modelObj{
			ID:           m.ID,
			Object:       m.Object,
			Type:         m.Type,
			Status:       m.Status,
			Capabilities: m.Capabilities,
			Provider:     m.Provider,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: data}) //nolint:errcheck
}

// Models returns all known models as ModelInfo slice.
func (g *Gateway) Models() []monitor.ModelInfo {
	cfg := g.cfg.Load()
	var models []monitor.ModelInfo

	// Local models from Manager (loaded/running)
	localModels := g.mgr.ListModels()
	for _, mi := range localModels {
		obj := monitor.ModelInfo{
			ID:     mi.ID,
			Object: "model",
			Type:   "local",
			Status: mi.Status,
		}
		if p, ok := cfg.MatchModel(mi.ID); ok {
			obj.Capabilities = p.Capabilities
		}
		models = append(models, obj)
	}

	listed := make(map[string]bool, len(models))
	for _, m := range models {
		listed[m.ID] = true
	}

	// HF cache models (local but not loaded)
	for _, cm := range g.scanHFCache() {
		if listed[cm] {
			continue
		}
		models = append(models, monitor.ModelInfo{
			ID:     cm,
			Object: "model",
			Type:   "local",
			Status: "available",
		})
		listed[cm] = true
	}

	// Configured remote models
	configuredRemote := make(map[string]bool)
	for pattern, p := range cfg.Models {
		if !p.IsRemote() {
			continue
		}
		configuredRemote[pattern] = true
		models = append(models, monitor.ModelInfo{
			ID:           pattern,
			Object:       "model",
			Type:         "remote",
			Status:       "available",
			Capabilities: p.Capabilities,
			Provider:     p.Provider,
		})
	}

	// Discovered remote models
	if g.discovery != nil {
		for _, dm := range g.discovery.Snapshot().AllModels() {
			if configuredRemote[dm.ID] {
				continue
			}
			models = append(models, monitor.ModelInfo{
				ID:       dm.ID,
				Object:   "model",
				Type:     "remote",
				Status:   "available",
				Provider: dm.Provider,
			})
		}
	}

	return models
}

// handleHealth returns 200 if at least one warmup model is healthy, 503 otherwise.
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if slices.ContainsFunc(g.cfg.Load().Warmup, g.mgr.Loaded) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"}) //nolint:errcheck
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(map[string]string{"status": "degraded"}) //nolint:errcheck
}

// handleRemote proxies a request to a remote OpenAI-compatible provider,
// bypassing the process manager entirely. body may be rewound if upstream_model rewrite is needed.
func (g *Gateway) handleRemote(w http.ResponseWriter, r *http.Request, modelName string, params config.ModelParams, cfg *config.Config, body []byte) {
	provider, ok := cfg.Providers[params.Provider]
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "unknown provider: "+params.Provider, "config_error")
		return
	}

	if params.UpstreamModel != "" {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(body, &obj); err == nil {
			if raw, err := json.Marshal(params.UpstreamModel); err == nil {
				obj["model"] = raw
				if rewritten, err := json.Marshal(obj); err == nil {
					body = rewritten
				}
			}
		}
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))

	baseTarget, err := url.Parse(provider.BaseURL)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "invalid provider URL", "config_error")
		return
	}

	apiKey := provider.APIKey
	reqPath := r.URL.Path
	upstreamModel := params.UpstreamModel
	if upstreamModel == "" {
		upstreamModel = modelName
	}

	// Track remote request metrics.
	var done func(int64, int)
	if g.remote != nil {
		done = g.remote.Start(modelName, params.Provider)
	}
	start := time.Now()

	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(baseTarget)
			pr.Out.URL.Path = baseTarget.Path + reqPath
			pr.Out.Host = baseTarget.Host
			pr.Out.Header.Set("Authorization", "Bearer "+apiKey)
		},
		FlushInterval: -1,
		ModifyResponse: func(resp *http.Response) error {
			resp.Header.Set("X-Wire-Model", upstreamModel)
			if done != nil {
				done(time.Since(start).Nanoseconds(), resp.StatusCode)
				done = nil // prevent double-call
			}
			return rewriteContextLengthError(resp)
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			g.log.Error("remote proxy error", "model", modelName, "provider", params.Provider, "error", err)
			if done != nil {
				done(time.Since(start).Nanoseconds(), 502)
				done = nil
			}
			writeError(w, http.StatusBadGateway, "server_error", "upstream error: "+err.Error(), "upstream_error")
		},
	}

	proxy.ServeHTTP(w, r)
}

// estimateTokens approximates the token count of a request body using a
// chars/4 heuristic. It extracts text from messages[].content (chat) and the
// input field (embeddings). Returns 0 on parse failure or empty content.
func estimateTokens(body []byte) int {
	var req struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return 0
	}

	total := 0
	for _, msg := range req.Messages {
		if len(msg.Content) == 0 {
			continue
		}
		var s string
		if err := json.Unmarshal(msg.Content, &s); err == nil {
			total += len(s)
			continue
		}
		var parts []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(msg.Content, &parts); err == nil {
			for _, p := range parts {
				total += len(p.Text)
			}
		}
	}

	if len(req.Input) > 0 {
		var s string
		if err := json.Unmarshal(req.Input, &s); err == nil {
			total += len(s)
		} else {
			var arr []string
			if err := json.Unmarshal(req.Input, &arr); err == nil {
				for _, s := range arr {
					total += len(s)
				}
			}
		}
	}

	return total / 4
}

// rewriteContextLengthError inspects non-2xx upstream responses for a
// context_length_exceeded error code and rewrites the status to 400 to match
// the OpenAI API standard. llama-server returns these as 500.
func rewriteContextLengthError(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		resp.Body = io.NopCloser(bytes.NewReader(nil))
		return nil
	}

	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error.Code == "context_length_exceeded" {
		resp.StatusCode = http.StatusBadRequest
		resp.Status = http.StatusText(http.StatusBadRequest)
	}

	resp.Body = io.NopCloser(bytes.NewReader(body))
	return nil
}

const hfCacheTTL = 30 * time.Second

// scanHFCache returns cached model IDs (org/repo:QUANT) for all GGUF files
// found in the HuggingFace hub cache. Results are cached for 30s to avoid
// repeated filesystem I/O on /v1/models calls.
func (g *Gateway) scanHFCache() []string {
	if g.hfCacheDir == "" {
		return nil
	}
	if ids, ok := g.hfCache.get(hfCacheTTL); ok {
		return ids
	}

	entries, err := os.ReadDir(g.hfCacheDir)
	if err != nil {
		return nil
	}

	var models []string
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || !strings.HasPrefix(name, "models--") {
			continue
		}
		// models--org--repo → org/repo
		parts := strings.SplitN(name, "--", 3)
		if len(parts) != 3 {
			continue
		}
		repo := parts[1] + "/" + parts[2]

		// Walk snapshots for .gguf files.
		snapshotsDir := filepath.Join(g.hfCacheDir, name, "snapshots")
		snapshots, err := os.ReadDir(snapshotsDir)
		if err != nil {
			continue
		}
		seen := make(map[string]bool)
		for _, snap := range snapshots {
			if !snap.IsDir() {
				continue
			}
			ggufs, err := filepath.Glob(filepath.Join(snapshotsDir, snap.Name(), "*.gguf"))
			if err != nil {
				continue
			}
			for _, path := range ggufs {
				base := strings.TrimSuffix(filepath.Base(path), ".gguf")
				// Skip auxiliary files (e.g. mmproj-BF16.gguf).
				if strings.HasPrefix(base, "mmproj") {
					continue
				}
				// Extract quant: look for Q prefix after the last dash-or-dot, but handle
				// compound quants like Q5_K_M. The pattern is: model-name[-size]Q<num>[_<variant>]
				// e.g. "Qwen3-4B-Q8_0" → "Q8_0", "Qwen3-4B-Q5_K_M" → "Q5_K_M"
				quant := extractQuant(base)
				if quant == "" {
					continue
				}
				id := repo + ":" + quant
				if !seen[id] {
					seen[id] = true
					models = append(models, id)
				}
			}
		}
	}
	g.hfCache.set(models, hfCacheTTL)
	return models
}

// extractQuant extracts the quantization suffix from a GGUF filename.
// Handles patterns like "Qwen3-4B-Q8_0", "Qwen3-4B-Q5_K_M", "Phi-4-mini-instruct.Q8_0".
// Returns empty string if no valid quant is found.
func extractQuant(base string) string {
	// Find last Q<num> and stop at -XB boundary (e.g., -4B, -8B)
	for i := len(base) - 1; i >= 0; i-- {
		if i > 0 && base[i] == 'B' && base[i-1] >= '0' && base[i-1] <= '9' {
			break
		}
		if i+1 < len(base) && base[i] == 'Q' && base[i+1] >= '0' && base[i+1] <= '9' {
			return base[i:]
		}
	}
	return ""
}

// pathPriority maps a request path to a queue priority tier.
func pathPriority(path string) queue.Priority {
	switch path {
	case "/v1/completions":
		return queue.PriorityMedium
	case "/v1/embeddings", "/v1/rerank":
		return queue.PriorityLow
	default:
		return queue.PriorityHigh
	}
}

// writeError writes an OpenAI-format JSON error response.
func writeError(w http.ResponseWriter, status int, errType, message, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
		"error": map[string]any{
			"message": message,
			"type":    errType,
			"code":    code,
		},
	})
}
