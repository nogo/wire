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
	"sync/atomic"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/process"
)

// nctxProvider wraps the NCtx lookup; *monitor.Hub satisfies this interface.
type nctxProvider interface {
	NCtx(model string) int
}

// backendManager is the subset of process.Manager the Gateway depends on.
type backendManager interface {
	Loaded(name string) bool
	// LoadOrGet returns the backend address for the named model, loading it on
	// demand if not already running. The caller must call release() when the
	// proxied request completes to decrement the in-flight counter.
	LoadOrGet(ctx context.Context, modelName string, params config.ModelParams) (addr string, release func(), err error)
	// ListModels returns all known models in non-idle states.
	ListModels() []monitor.ModelInfo
}

// Gateway is an HTTP reverse-proxy that routes OpenAI-compatible requests to llama-server processes.
type Gateway struct {
	cfg       atomic.Pointer[config.Config]
	mgr       backendManager
	log       *slog.Logger
	affinity  *affinityTable
	hub       nctxProvider
	remote    *monitor.RemoteTracker
	discovery *monitor.ModelDiscovery
	server    *http.Server
}

// New creates a Gateway backed by the given process manager.
// hub may be nil; when non-nil, /monitor and /monitor/ws routes are registered.
// remote may be nil; when non-nil, remote provider requests are tracked.
func New(cfg *config.Config, mgr *process.Manager, hub *monitor.Hub, remote *monitor.RemoteTracker, discovery *monitor.ModelDiscovery, log *slog.Logger) *Gateway {
	return newGateway(cfg, mgr, hub, remote, discovery, log)
}

func newGateway(cfg *config.Config, mgr backendManager, hub *monitor.Hub, remote *monitor.RemoteTracker, discovery *monitor.ModelDiscovery, log *slog.Logger) *Gateway {
	// Build affinity table from warmup models.
	modelSlots := make(map[string]int, len(cfg.Warmup))
	for _, name := range cfg.Warmup {
		if params, ok := cfg.MatchModel(name); ok {
			modelSlots[name] = params.Slots
		}
	}

	g := &Gateway{mgr: mgr, log: log, affinity: newAffinityTable(modelSlots), remote: remote, discovery: discovery}
	g.cfg.Store(cfg)
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

// handleRoute extracts the model field from the JSON body and proxies to the matching process.
// If the model is known but not loaded, it is started on demand; the HTTP connection
// is held open until the model becomes healthy.
func (g *Gateway) handleRoute(w http.ResponseWriter, r *http.Request) {
	// Tee the body so the decoder and the upstream both see it.
	var captured bytes.Buffer
	tee := io.TeeReader(r.Body, &captured)
	dec := json.NewDecoder(tee)

	var payload struct {
		Model string `json:"model"`
	}
	if err := dec.Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error(), "bad_request")
		return
	}
	if payload.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing model field", "bad_request")
		return
	}

	// Reconstruct full body: captured bytes (read by decoder) + rest not yet read.
	r.Body = io.NopCloser(io.MultiReader(&captured, r.Body))

	currentCfg := g.cfg.Load()
	params, ok := currentCfg.MatchModel(payload.Model)
	if !ok {
		// Fallback: check discovered models from provider listings.
		if g.discovery != nil {
			if dm, found := g.discovery.Snapshot().Lookup(payload.Model); found {
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
	}

	if params.IsRemote() {
		g.handleRemote(w, r, payload.Model, params, currentCfg)
		return
	}

	// LoadOrGet blocks until the model is healthy (or fails). The load context is
	// detached from the caller's request context inside the manager, so a client
	// disconnect does not abort an in-progress load that other requests may share.
	addr, release, err := g.mgr.LoadOrGet(r.Context(), payload.Model, params)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			writeError(w, http.StatusGatewayTimeout, "server_error", "model load timed out: "+err.Error(), "load_timeout")
			return
		}
		writeError(w, http.StatusServiceUnavailable, "server_error", err.Error(), "model_not_loaded")
		return
	}
	defer release()

	// Context window guard: reject requests that would exceed 90% of n_ctx.
	// Best-effort: skipped when hub is nil, n_ctx is unknown, or estimation fails.
	if g.hub != nil {
		if nCtx := g.hub.NCtx(payload.Model); nCtx > 0 {
			if est := estimateTokens(captured.Bytes()); est > 0 {
				if est > nCtx*9/10 {
					writeError(w, http.StatusBadRequest, "invalid_request_error",
						fmt.Sprintf("estimated %d tokens exceeds context window of %d (n_ctx)", est, nCtx),
						"context_length_exceeded")
					return
				}
			}
		}
	}

	// Register affinity entry for the newly loaded model (idempotent for warmup models).
	g.affinity.Register(payload.Model, params.Slots)

	// Slot affinity: read full body, compute prefix hash, inject id_slot if assigned.
	// On any error, fall through with the original body unchanged.
	if allBody, readErr := io.ReadAll(r.Body); readErr == nil {
		hash := prefixHash(allBody)
		if slotID, assigned := g.affinity.Assign(payload.Model, hash); assigned {
			allBody = injectSlot(allBody, slotID)
		}
		r.Body = io.NopCloser(bytes.NewReader(allBody))
		r.ContentLength = int64(len(allBody))
	}

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
		OwnedBy      string   `json:"owned_by"`
		Status       string   `json:"status,omitempty"`
		Capabilities []string `json:"capabilities,omitempty"`
		Local        bool     `json:"local"`
	}
	type modelsResp struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}

	cfg := g.cfg.Load()

	// Local models from the process manager.
	models := g.mgr.ListModels()
	data := make([]modelObj, 0, len(models))
	for _, mi := range models {
		obj := modelObj{
			ID:      mi.ID,
			Object:  "model",
			OwnedBy: "wire",
			Status:  mi.Status,
			Local:   true,
		}
		if p, ok := cfg.MatchModel(mi.ID); ok {
			obj.Capabilities = p.Capabilities
		}
		data = append(data, obj)
	}

	// Remote models from config.
	configuredRemote := make(map[string]bool)
	for pattern, p := range cfg.Models {
		if !p.IsRemote() {
			continue
		}
		configuredRemote[pattern] = true
		data = append(data, modelObj{
			ID:           pattern,
			Object:       "model",
			OwnedBy:      p.Provider,
			Status:       "available",
			Capabilities: p.Capabilities,
			Local:        false,
		})
	}

	// Discovered models from provider listings (skip duplicates with config).
	if g.discovery != nil {
		for _, dm := range g.discovery.Snapshot().AllModels() {
			if configuredRemote[dm.ID] {
				continue
			}
			data = append(data, modelObj{
				ID:      dm.ID,
				Object:  "model",
				OwnedBy: dm.Provider,
				Status:  "available",
				Local:   false,
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: data}) //nolint:errcheck
}

// handleHealth returns 200 if at least one warmup model is healthy, 503 otherwise.
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	for _, name := range g.cfg.Load().Warmup {
		if g.mgr.Loaded(name) {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"}) //nolint:errcheck
			return
		}
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(map[string]string{"status": "degraded"}) //nolint:errcheck
}

// handleRemote proxies a request to a remote OpenAI-compatible provider,
// bypassing the process manager entirely.
func (g *Gateway) handleRemote(w http.ResponseWriter, r *http.Request, modelName string, params config.ModelParams, cfg *config.Config) {
	provider, ok := cfg.Providers[params.Provider]
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "unknown provider: "+params.Provider, "config_error")
		return
	}

	// Rewrite model field if upstream_model is set.
	if params.UpstreamModel != "" {
		allBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "failed to read body", "bad_request")
			return
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(allBody, &obj); err == nil {
			if raw, err := json.Marshal(params.UpstreamModel); err == nil {
				obj["model"] = raw
				if rewritten, err := json.Marshal(obj); err == nil {
					allBody = rewritten
				}
			}
		}
		r.Body = io.NopCloser(bytes.NewReader(allBody))
		r.ContentLength = int64(len(allBody))
	}

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
