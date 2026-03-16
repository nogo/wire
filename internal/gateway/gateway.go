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
	"strings"
	"sync"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/process"
)

// backendManager is the subset of process.Manager the Gateway depends on.
type backendManager interface {
	Addr(name string) (string, bool)
	Healthy(name string) bool
}

// Gateway is an HTTP reverse-proxy that routes OpenAI-compatible requests to backends.
type Gateway struct {
	cfg    *config.Config
	mgr    backendManager
	log    *slog.Logger
	rt     *routeTable
	server *http.Server
}

// New creates a Gateway backed by the given process manager.
func New(cfg *config.Config, mgr *process.Manager, log *slog.Logger) *Gateway {
	return newGateway(cfg, mgr, log)
}

func newGateway(cfg *config.Config, mgr backendManager, log *slog.Logger) *Gateway {
	g := &Gateway{cfg: cfg, mgr: mgr, log: log, rt: buildRouteTable(cfg)}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.handleRoute)
	mux.HandleFunc("POST /v1/completions", g.handleRoute)
	mux.HandleFunc("POST /v1/embeddings", g.handleRoute)
	mux.HandleFunc("POST /v1/rerank", g.handleRoute)
	mux.HandleFunc("GET /v1/models", g.handleModels)
	mux.HandleFunc("GET /health", g.handleHealth)

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

// handleRoute extracts the model field from the JSON body and proxies to the matching backend.
func (g *Gateway) handleRoute(w http.ResponseWriter, r *http.Request) {
	if len(g.cfg.Backends) == 0 {
		writeError(w, http.StatusInternalServerError, "server_error", "no backends configured", "no_backends")
		return
	}

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

	backendName, err := g.rt.resolve(payload.Model)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error(), "model_not_found")
		return
	}

	if !g.mgr.Healthy(backendName) {
		writeError(w, http.StatusServiceUnavailable, "server_error", "backend unavailable", "backend_unavailable")
		return
	}

	addr, ok := g.mgr.Addr(backendName)
	if !ok {
		writeError(w, http.StatusServiceUnavailable, "server_error", "backend unavailable", "backend_unavailable")
		return
	}

	target, parseErr := url.Parse(addr)
	if parseErr != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "invalid backend URL", "internal_error")
		return
	}

	rewriter := g.buildReplacer(backendName)

	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Host = target.Host
		},
		FlushInterval: -1,
		ModifyResponse: func(resp *http.Response) error {
			if rewriter != nil {
				resp.Header.Del("Content-Length")
				resp.ContentLength = -1
				resp.Body = io.NopCloser(&replacingReader{r: resp.Body, repl: rewriter})
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			g.log.Error("proxy error", "backend", backendName, "error", err)
			writeError(w, http.StatusBadGateway, "server_error", "upstream error: "+err.Error(), "upstream_error")
		},
	}

	proxy.ServeHTTP(w, r)
}

// handleModels merges GET /v1/models from all backends and rewrites model IDs to backend names.
func (g *Gateway) handleModels(w http.ResponseWriter, r *http.Request) {
	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	type modelsResp struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}

	var (
		mu   sync.Mutex
		data []modelObj
		wg   sync.WaitGroup
	)
	now := time.Now().Unix()

	for name := range g.cfg.Backends {
		name := name
		wg.Add(1)
		go func() {
			defer wg.Done()
			fallback := modelObj{ID: name, Object: "model", Created: now, OwnedBy: "wire"}

			addr, ok := g.mgr.Addr(name)
			if !ok {
				mu.Lock()
				data = append(data, fallback)
				mu.Unlock()
				return
			}

			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()

			req, err := http.NewRequestWithContext(ctx, http.MethodGet, addr+"/v1/models", nil)
			if err != nil {
				mu.Lock()
				data = append(data, fallback)
				mu.Unlock()
				return
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				mu.Lock()
				data = append(data, fallback)
				mu.Unlock()
				return
			}
			defer resp.Body.Close()

			var mr modelsResp
			if err := json.NewDecoder(resp.Body).Decode(&mr); err != nil || len(mr.Data) == 0 {
				mu.Lock()
				data = append(data, fallback)
				mu.Unlock()
				return
			}

			entry := mr.Data[0]
			entry.ID = name
			mu.Lock()
			data = append(data, entry)
			mu.Unlock()
		}()
	}
	wg.Wait()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: data}) //nolint:errcheck
}

// handleHealth returns 200 if at least one backend is healthy, 503 otherwise.
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	for name := range g.cfg.Backends {
		if g.mgr.Healthy(name) {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"}) //nolint:errcheck
			return
		}
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(map[string]string{"status": "degraded"}) //nolint:errcheck
}

// buildReplacer creates a strings.Replacer that rewrites llama-server model name aliases
// in response bodies to the canonical backend name.
func (g *Gateway) buildReplacer(backendName string) *strings.Replacer {
	aliases := g.rt.backendAliases[backendName]
	var pairs []string
	for _, alias := range aliases {
		if alias == backendName {
			continue
		}
		pairs = append(pairs,
			`"model":"`+alias+`"`, `"model":"`+backendName+`"`,
		)
	}
	if len(pairs) == 0 {
		return nil
	}
	return strings.NewReplacer(pairs...)
}

// replacingReader wraps an io.Reader and applies a strings.Replacer to each chunk read.
// Overflow bytes (when replacement expands the chunk) are buffered internally.
type replacingReader struct {
	r    io.Reader
	repl *strings.Replacer
	buf  bytes.Buffer
}

func (rr *replacingReader) Read(p []byte) (int, error) {
	if rr.buf.Len() > 0 {
		return rr.buf.Read(p)
	}
	tmp := make([]byte, len(p))
	n, readErr := rr.r.Read(tmp)
	if n > 0 {
		replaced := []byte(rr.repl.Replace(string(tmp[:n])))
		copy(p, replaced)
		if len(replaced) > len(p) {
			rr.buf.Write(replaced[len(p):])
			return len(p), nil // defer readErr until buf is drained
		}
		return len(replaced), readErr
	}
	return 0, readErr
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
