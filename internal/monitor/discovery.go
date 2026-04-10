package monitor

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"time"
)

// DiscoveredModel is a model found by querying a provider's /v1/models endpoint.
type DiscoveredModel struct {
	ID       string
	Provider string // provider name from config
	OwnedBy  string // owned_by from upstream response
}

// DiscoverySnapshot is a point-in-time copy of all discovered models grouped by provider.
type DiscoverySnapshot struct {
	ByProvider map[string][]DiscoveredModel
}

// Lookup returns the first provider that offers the named model, sorted
// alphabetically by provider name for determinism. Returns false if no
// provider lists the model.
func (s DiscoverySnapshot) Lookup(model string) (DiscoveredModel, bool) {
	var matches []DiscoveredModel
	for _, models := range s.ByProvider {
		for _, m := range models {
			if m.ID == model {
				matches = append(matches, m)
			}
		}
	}
	if len(matches) == 0 {
		return DiscoveredModel{}, false
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Provider < matches[j].Provider
	})
	return matches[0], true
}

// AllModels returns all discovered models as a flat slice.
func (s DiscoverySnapshot) AllModels() []DiscoveredModel {
	var out []DiscoveredModel
	for _, models := range s.ByProvider {
		out = append(out, models...)
	}
	return out
}

// ProviderEndpoint is the info needed to query a provider's model list.
type ProviderEndpoint struct {
	Name    string
	BaseURL string
	APIKey  string
}

// ModelDiscovery periodically fetches /v1/models from each configured provider
// and caches the results. Safe for concurrent use.
type ModelDiscovery struct {
	mu         sync.RWMutex
	byProvider map[string][]DiscoveredModel

	endpoints []ProviderEndpoint
	log       *slog.Logger
	client    *http.Client
	interval  time.Duration

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewModelDiscovery creates a discovery instance for the given provider endpoints.
func NewModelDiscovery(endpoints []ProviderEndpoint, log *slog.Logger) *ModelDiscovery {
	ctx, cancel := context.WithCancel(context.Background())
	eps := make([]ProviderEndpoint, len(endpoints))
	copy(eps, endpoints)
	sort.Slice(eps, func(i, j int) bool { return eps[i].Name < eps[j].Name })

	return &ModelDiscovery{
		byProvider: make(map[string][]DiscoveredModel),
		endpoints:  eps,
		log:        log,
		client:     &http.Client{Timeout: 15 * time.Second},
		interval:   5 * time.Minute,
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start performs an initial fetch, then polls at the configured interval.
func (d *ModelDiscovery) Start() {
	d.refresh()
	d.wg.Add(1)
	go d.pollLoop()
}

// Stop cancels the poll loop and waits for it to exit.
func (d *ModelDiscovery) Stop() {
	d.cancel()
	d.wg.Wait()
}

// Snapshot returns a copy of all discovered models.
func (d *ModelDiscovery) Snapshot() DiscoverySnapshot {
	d.mu.RLock()
	defer d.mu.RUnlock()
	cp := make(map[string][]DiscoveredModel, len(d.byProvider))
	for k, v := range d.byProvider {
		models := make([]DiscoveredModel, len(v))
		copy(models, v)
		cp[k] = models
	}
	return DiscoverySnapshot{ByProvider: cp}
}

func (d *ModelDiscovery) pollLoop() {
	defer d.wg.Done()
	ticker := time.NewTicker(d.interval)
	defer ticker.Stop()
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.refresh()
		}
	}
}

func (d *ModelDiscovery) refresh() {
	for _, ep := range d.endpoints {
		models, err := d.fetchModels(ep)
		if err != nil {
			d.log.Warn("model discovery failed", "provider", ep.Name, "error", err)
			continue
		}
		d.mu.Lock()
		d.byProvider[ep.Name] = models
		d.mu.Unlock()
		d.log.Info("models discovered", "provider", ep.Name, "count", len(models))
	}
}

func (d *ModelDiscovery) fetchModels(ep ProviderEndpoint) ([]DiscoveredModel, error) {
	reqURL := ep.BaseURL + "/v1/models"
	req, err := http.NewRequestWithContext(d.ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+ep.APIKey)

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	models := make([]DiscoveredModel, len(result.Data))
	for i, m := range result.Data {
		models[i] = DiscoveredModel{
			ID:       m.ID,
			Provider: ep.Name,
			OwnedBy:  m.OwnedBy,
		}
	}
	return models, nil
}
