package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/process"
)

// backendManager is the subset of process.Manager the Gateway depends on.
type backendManager interface {
	Addr(name string) (string, bool)
	Healthy(name string) bool
	RunningModels() []string
}

// Gateway is an HTTP reverse-proxy that routes OpenAI-compatible requests to llama-server processes.
type Gateway struct {
	cfg      *config.Config
	mgr      backendManager
	log      *slog.Logger
	affinity *affinityTable
	server   *http.Server
}

// New creates a Gateway backed by the given process manager.
// hub may be nil; when non-nil, /monitor and /monitor/ws routes are registered.
func New(cfg *config.Config, mgr *process.Manager, hub *monitor.Hub, log *slog.Logger) *Gateway {
	return newGateway(cfg, mgr, hub, log)
}

func newGateway(cfg *config.Config, mgr backendManager, hub *monitor.Hub, log *slog.Logger) *Gateway {
	// Build affinity table from warmup models.
	modelSlots := make(map[string]int, len(cfg.Warmup))
	for _, name := range cfg.Warmup {
		if params, ok := cfg.MatchModel(name); ok {
			modelSlots[name] = params.Slots
		}
	}

	g := &Gateway{cfg: cfg, mgr: mgr, log: log, affinity: newAffinityTable(modelSlots)}

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

// handleRoute extracts the model field from the JSON body and proxies to the matching process.
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

	if _, ok := g.cfg.MatchModel(payload.Model); !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "unknown model: "+payload.Model, "model_not_found")
		return
	}

	if !g.mgr.Healthy(payload.Model) {
		writeError(w, http.StatusServiceUnavailable, "server_error", "model not loaded", "model_not_loaded")
		return
	}

	addr, ok := g.mgr.Addr(payload.Model)
	if !ok {
		writeError(w, http.StatusServiceUnavailable, "server_error", "model not loaded", "model_not_loaded")
		return
	}

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
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			g.log.Error("proxy error", "model", payload.Model, "error", err)
			writeError(w, http.StatusBadGateway, "server_error", "upstream error: "+err.Error(), "upstream_error")
		},
	}

	proxy.ServeHTTP(w, r)
}

// handleModels returns the list of currently running models.
func (g *Gateway) handleModels(w http.ResponseWriter, _ *http.Request) {
	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	type modelsResp struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}

	running := g.mgr.RunningModels()
	data := make([]modelObj, len(running))
	for i, name := range running {
		data[i] = modelObj{ID: name, Object: "model", OwnedBy: "wire"}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: data}) //nolint:errcheck
}

// handleHealth returns 200 if at least one warmup model is healthy, 503 otherwise.
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	for _, name := range g.cfg.Warmup {
		if g.mgr.Healthy(name) {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"}) //nolint:errcheck
			return
		}
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(map[string]string{"status": "degraded"}) //nolint:errcheck
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
