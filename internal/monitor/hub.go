package monitor

import (
	"context"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// HubMessageKind identifies the type of a HubMessage.
type HubMessageKind int

const (
	MsgSnapshot    HubMessageKind = iota // initial state snapshot for a new subscriber
	MsgMetricsTick                       // backend metrics polled from /metrics + /slots
	MsgGPUTick                           // GPU hardware metrics polled from sysfs
	MsgServerEvent                       // structured event relayed from eventParser
	MsgStateChange                       // backend health state changed
	MsgRemoteTick                        // remote provider metrics polled from RemoteTracker
	MsgModelsTick                        // models refreshed from model provider
)

// BackendMetrics is the latest polled state for one managed backend.
type BackendMetrics struct {
	Name                  string
	BuildInfo             string
	RequestsProcessing    int
	RequestsPending       int
	TokensPerSecond       float64
	PromptTokensPerSecond float64
	TokensPredictedTotal  int64
	PromptTokensTotal     int64
	Slots                 []Slot
}

// HubSnapshot is the full current state, delivered to new subscribers.
type HubSnapshot struct {
	Backends []BackendMetrics
	GPUs     []GPUMetrics
	Remote   map[string]RemoteStats
	Models   []ModelInfo
}

// HubMessage is a tagged message sent to subscribers.
type HubMessage struct {
	Kind    HubMessageKind
	Snap    *HubSnapshot           // MsgSnapshot
	Metrics *BackendMetrics        // MsgMetricsTick
	GPUs    []GPUMetrics           // MsgGPUTick
	Event   *ServerEvent           // MsgServerEvent
	Backend string                 // MsgStateChange
	Loaded  bool                   // MsgStateChange
	Remote  map[string]RemoteStats // MsgRemoteTick
	Models  []ModelInfo            // MsgModelsTick
}

// Hub subscribes to the event channel from Manager, polls each backend's HTTP
// endpoints every second, and fans out to subscribers. GPU state is received
// from the VRAM tracker via gpuStateCh instead of polling sysfs directly.
type Hub struct {
	log           *slog.Logger
	events        <-chan ServerEvent
	snapshotFn    func() []BackendStatus
	gpuStateCh    <-chan []GPUMetrics
	remote        *RemoteTracker
	modelProvider func() []ModelInfo

	mu         sync.RWMutex
	metrics    map[string]*BackendMetrics
	gpuData    []GPUMetrics
	remoteData map[string]RemoteStats
	modelsData []ModelInfo

	subsMu sync.Mutex
	subs   map[uint64]chan HubMessage
	nextID atomic.Uint64

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	client *http.Client
}

// SetModelProvider registers a callback that returns the current model list.
func (h *Hub) SetModelProvider(fn func() []ModelInfo) {
	h.modelProvider = fn
}

// NewHub creates a Hub. snapshotFn is called on every poll tick to obtain
// current backend addresses (Manager.Snapshot satisfies this). remote may be
// nil; when non-nil, remote provider metrics are included in the poll loop.
// gpuStateCh delivers GPU metrics from the VRAM tracker (may be nil).
// Call Start to begin the event relay and poll loop.
func NewHub(log *slog.Logger, events <-chan ServerEvent, snapshotFn func() []BackendStatus, remote *RemoteTracker, _ *ModelDiscovery, gpuStateCh <-chan []GPUMetrics) *Hub {
	ctx, cancel := context.WithCancel(context.Background())
	h := &Hub{
		log:        log,
		events:     events,
		snapshotFn: snapshotFn,
		gpuStateCh: gpuStateCh,
		remote:     remote,
		metrics:    make(map[string]*BackendMetrics),
		remoteData: make(map[string]RemoteStats),
		subs:       make(map[uint64]chan HubMessage),
		ctx:        ctx,
		cancel:     cancel,
		client:     &http.Client{Timeout: 3 * time.Second},
	}
	return h
}

// Start launches the event relay and poll goroutines.
func (h *Hub) Start() {
	h.wg.Add(2)
	go h.relayEvents()
	go h.pollLoop()
	if h.gpuStateCh != nil {
		h.wg.Add(1)
		go h.gpuStateLoop()
	}
}

// Stop cancels all goroutines, waits for them to exit, then closes all
// subscriber channels. No goroutines are leaked after Stop returns.
func (h *Hub) Stop() {
	h.cancel()
	h.wg.Wait()

	h.subsMu.Lock()
	for id, ch := range h.subs {
		close(ch)
		delete(h.subs, id)
	}
	h.subsMu.Unlock()
}

// Subscribe registers a subscriber. The returned channel (buffered 64) receives
// a MsgSnapshot immediately, then subsequent messages as they arrive. The
// returned func removes the subscriber; Stop closes the channel.
func (h *Hub) Subscribe() (<-chan HubMessage, func()) {
	ch := make(chan HubMessage, 64)
	id := h.nextID.Add(1)

	h.subsMu.Lock()
	h.subs[id] = ch
	h.subsMu.Unlock()

	snap := h.Snapshot()
	select {
	case ch <- HubMessage{Kind: MsgSnapshot, Snap: &snap}:
	default:
	}

	return ch, func() {
		h.subsMu.Lock()
		delete(h.subs, id)
		h.subsMu.Unlock()
	}
}

// Statuses returns the current runtime state of all managed backends by
// delegating to the snapshotFn provided at construction. Safe to call
// concurrently.
func (h *Hub) Statuses() []BackendStatus {
	return h.snapshotFn()
}

// Snapshot returns a copy of the current full state synchronously.
func (h *Hub) Snapshot() HubSnapshot {
	h.mu.RLock()
	defer h.mu.RUnlock()

	backends := make([]BackendMetrics, 0, len(h.metrics))
	for _, m := range h.metrics {
		cp := *m
		backends = append(backends, cp)
	}
	gpuCopy := make([]GPUMetrics, len(h.gpuData))
	copy(gpuCopy, h.gpuData)

	var remoteCopy map[string]RemoteStats
	if len(h.remoteData) > 0 {
		remoteCopy = make(map[string]RemoteStats, len(h.remoteData))
		maps.Copy(remoteCopy, h.remoteData)
	}

	var modelsCopy []ModelInfo
	if len(h.modelsData) > 0 {
		modelsCopy = make([]ModelInfo, len(h.modelsData))
		copy(modelsCopy, h.modelsData)
	}

	return HubSnapshot{Backends: backends, GPUs: gpuCopy, Remote: remoteCopy, Models: modelsCopy}
}

func (h *Hub) relayEvents() {
	defer h.wg.Done()
	for {
		select {
		case <-h.ctx.Done():
			return
		case ev, ok := <-h.events:
			if !ok {
				return
			}
			ev2 := ev
			h.broadcast(HubMessage{Kind: MsgServerEvent, Event: &ev2})
		}
	}
}

func (h *Hub) pollLoop() {
	defer h.wg.Done()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.poll()
		}
	}
}

func (h *Hub) poll() {
	for _, b := range h.snapshotFn() {
		if b.Port == 0 {
			continue
		}
		m := h.fetchBackend(b)
		h.mu.Lock()
		h.metrics[b.Name] = m
		h.mu.Unlock()
		h.broadcast(HubMessage{Kind: MsgMetricsTick, Metrics: m})
	}

	if h.remote != nil {
		snap := h.remote.Snapshot()
		h.mu.Lock()
		h.remoteData = snap
		h.mu.Unlock()
		if len(snap) > 0 {
			h.broadcast(HubMessage{Kind: MsgRemoteTick, Remote: snap})
		}
	}

	if h.modelProvider != nil {
		models := h.modelProvider()
		h.mu.Lock()
		h.modelsData = models
		h.mu.Unlock()
		if len(models) > 0 {
			h.broadcast(HubMessage{Kind: MsgModelsTick, Models: models})
		}
	}
}

func (h *Hub) fetchBackend(b BackendStatus) *BackendMetrics {
	m := &BackendMetrics{Name: b.Name}
	base := "http://localhost:" + itoa(b.Port)

	if data, err := h.fetch(base + "/metrics"); err == nil {
		g := ParsePrometheusGauges(string(data))
		m.RequestsProcessing = int(g["llamacpp:requests_processing"])
		m.RequestsPending = int(g["llamacpp:requests_deferred"])
		m.TokensPerSecond = g["llamacpp:predicted_tokens_seconds"]
		m.PromptTokensPerSecond = g["llamacpp:prompt_tokens_seconds"]
		m.TokensPredictedTotal = int64(g["llamacpp:tokens_predicted_total"])
		m.PromptTokensTotal = int64(g["llamacpp:prompt_tokens_total"])
	} else {
		h.log.Debug("metrics fetch failed", "backend", b.Name, "error", err)
	}

	if data, err := h.fetch(base + "/slots"); err == nil {
		if slots, err := ParseSlots(data); err == nil {
			m.Slots = slots
		}
	} else {
		h.log.Debug("slots fetch failed", "backend", b.Name, "error", err)
	}

	if data, err := h.fetch(base + "/props"); err == nil {
		if info, err := ParseProps(data); err == nil {
			m.BuildInfo = info
		}
	}

	return m
}

func (h *Hub) fetch(url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(h.ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (h *Hub) broadcast(msg HubMessage) {
	h.subsMu.Lock()
	defer h.subsMu.Unlock()
	for _, ch := range h.subs {
		select {
		case ch <- msg:
		default:
			// Drop to slow consumer.
		}
	}
}

// gpuStateLoop receives GPU metrics from the VRAM tracker and broadcasts them.
func (h *Hub) gpuStateLoop() {
	defer h.wg.Done()
	for {
		select {
		case <-h.ctx.Done():
			return
		case metrics, ok := <-h.gpuStateCh:
			if !ok {
				return
			}
			h.mu.Lock()
			h.gpuData = metrics
			h.mu.Unlock()
			if len(metrics) > 0 {
				h.broadcast(HubMessage{Kind: MsgGPUTick, GPUs: metrics})
			}
		}
	}
}

// NCtx returns the n_ctx value for the named model from the last poll.
// Returns 0 if the model is unknown or has no slots (guard should be skipped).
func (h *Hub) NCtx(model string) int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	m, ok := h.metrics[model]
	if !ok || len(m.Slots) == 0 {
		return 0
	}
	return m.Slots[0].NCtx
}

// itoa converts an int to its decimal string representation without importing
// the strconv package at the call site.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	buf := make([]byte, 0, 10)
	for n > 0 {
		buf = append([]byte{byte('0' + n%10)}, buf...)
		n /= 10
	}
	if neg {
		buf = append([]byte{'-'}, buf...)
	}
	return string(buf)
}
