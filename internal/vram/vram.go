package vram

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/nogo/wire/internal/monitor"
)

// Tracker manages VRAM budget tracking, admission control, and sysfs reconciliation.
type Tracker struct {
	backend      Backend
	hfCache      *HFCache
	estimates    sync.Map // modelRef → uint64 (VRAM estimate bytes)
	reservations sync.Map // modelRef → uint64 (reserved bytes)
	pending      sync.Map // modelRef → uint64 (pending release, waiting for sysfs confirmation)

	gpuStateCh chan []monitor.GPUMetrics // published to Hub

	totalVRAM uint64
	mu        sync.RWMutex

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// New creates a Tracker with the given backend and HF cache scanner.
// Call Discover() then Start() to begin tracking.
func New(backend Backend, hfCache *HFCache) *Tracker {
	ctx, cancel := context.WithCancel(context.Background())
	return &Tracker{
		backend:    backend,
		hfCache:    hfCache,
		gpuStateCh: make(chan []monitor.GPUMetrics, 64),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// HFCache returns the HF cache scanner for manual scanning if needed.
func (t *Tracker) HFCache() *HFCache {
	return t.hfCache
}

// Discover runs the backend auto-detection chain and populates totalVRAM.
// Returns the discovered GPUs.
func (t *Tracker) Discover() []monitor.GPUInfo {
	gpus := t.backend.Discover()
	if t.backend.IsNoop() {
		slog.Info("VRAM tracker: noop backend (CPU-only)")
		return gpus
	}
	t.mu.Lock()
	t.totalVRAM = t.backend.TotalVRAM()
	t.mu.Unlock()
	slog.Info("VRAM tracker: discovered GPUs", "count", len(gpus), "total_vram", t.totalVRAM)
	return gpus
}

// CanFit reports whether estimate bytes would fit in the current VRAM budget.
// Budget = total - (actual_sysfs_used + pending_releases).
func (t *Tracker) CanFit(estimate uint64) bool {
	if t.backend.IsNoop() {
		return true
	}
	t.mu.RLock()
	total := t.totalVRAM
	t.mu.RUnlock()

	actualUsed := t.backend.UsedVRAM()
	pending := t.pendingTotal()

	sum := actualUsed + pending
	if sum >= total {
		return false
	}
	return estimate <= total-sum
}

// FreeVRAM returns the current VRAM budget available for new models:
// total − actual_sysfs_used − pending_releases.
// Returns math.MaxUint64 for noop (CPU-only) backends.
func (t *Tracker) FreeVRAM() uint64 {
	if t.backend.IsNoop() {
		return math.MaxUint64
	}
	t.mu.RLock()
	total := t.totalVRAM
	t.mu.RUnlock()
	used := t.backend.UsedVRAM()
	pending := t.pendingTotal()
	sum := used + pending
	if sum >= total {
		return 0
	}
	return total - sum
}

// Reserve marks VRAM for a model. estimate is the expected VRAM consumption.
// Returns false if the model is already reserved or if it would exceed budget.
func (t *Tracker) Reserve(model string, estimate uint64) bool {
	if t.backend.IsNoop() {
		return true
	}
	if _, loaded := t.reservations.LoadOrStore(model, estimate); loaded {
		return true // already reserved
	}
	if !t.CanFit(estimate) {
		t.reservations.Delete(model)
		return false
	}
	slog.Info("VRAM reserved", "model", model, "bytes", estimate)
	return true
}

// Release marks a model's VRAM as pending release.
// The reservation is moved to pending until sysfs reconciliation confirms the drop.
func (t *Tracker) Release(model string) {
	if t.backend.IsNoop() {
		return
	}
	if v, loaded := t.reservations.LoadAndDelete(model); loaded {
		t.pending.Store(model, v)
		slog.Info("VRAM release pending", "model", model, "bytes", v)
	}
}

// EstimateForModel returns the VRAM estimate for a model reference.
// It checks the HF cache scan first, then tries to stat the path directly.
func (t *Tracker) EstimateForModel(modelRef string) uint64 {
	if t.hfCache != nil {
		if v := t.hfCache.Estimate(modelRef); v > 0 {
			return v
		}
	}
	if v, ok := t.estimates.Load(modelRef); ok {
		return v.(uint64)
	}
	// Try placeholder based on model name.
	if v := PlaceholderVRAM(modelRef); v > 0 {
		t.estimates.Store(modelRef, v)
		return v
	}
	return 0
}

// SetEstimate records a VRAM estimate for a model reference.
func (t *Tracker) SetEstimate(modelRef string, estimate uint64) {
	t.estimates.Store(modelRef, estimate)
}

// GPUStateCh returns a receive-only channel that delivers GPU metrics snapshots.
// Hub subscribes to this channel for MsgGPUTick messages.
func (t *Tracker) GPUStateCh() <-chan []monitor.GPUMetrics {
	return t.gpuStateCh
}

// Start begins the poll loop that reconciles reservations with sysfs readings.
func (t *Tracker) Start() {
	t.wg.Add(1)
	go t.pollLoop()
}

// Stop cancels the poll loop and waits for it to exit.
func (t *Tracker) Stop() {
	t.cancel()
	t.wg.Wait()
	close(t.gpuStateCh)
}

// pendingTotal returns the sum of all pending-release VRAM.
func (t *Tracker) pendingTotal() uint64 {
	var total uint64
	t.pending.Range(func(_, v interface{}) bool {
		total += v.(uint64)
		return true
	})
	return total
}

// reservedTotal returns the sum of all active reservations.
func (t *Tracker) reservedTotal() uint64 {
	var total uint64
	t.reservations.Range(func(_, v interface{}) bool {
		total += v.(uint64)
		return true
	})
	return total
}

func (t *Tracker) pollLoop() {
	defer t.wg.Done()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-t.ctx.Done():
			return
		case <-ticker.C:
			t.reconcile()
			t.pushGPUState()
		}
	}
}

// reconcile compares sysfs actual VRAM used against tracked reservations+pending,
// and clears stale entries where actual usage has dropped.
func (t *Tracker) reconcile() {
	if t.backend.IsNoop() {
		return
	}
	actualUsed := t.backend.UsedVRAM()
	reserved := t.reservedTotal()
	pending := t.pendingTotal()

	// Clear pending releases where sysfs shows the VRAM has been freed.
	if pending > 0 && actualUsed < reserved+pending {
		t.pending.Range(func(key, val interface{}) bool {
			// If actual usage dropped significantly, assume this model's VRAM was freed.
			if actualUsed <= reserved {
				t.pending.Delete(key)
				slog.Info("VRAM release confirmed by sysfs", "model", key)
			}
			return true
		})
	}

	// If sysfs shows less used than we think is reserved, something was
	// externally freed — clear all reservations to stay aligned.
	if actualUsed < reserved {
		t.reservations.Range(func(key, _ interface{}) bool {
			slog.Warn("VRAM mismatch: sysfs lower than reserved, clearing reservation",
				"model", key, "actual_used", actualUsed, "reserved", reserved)
			t.reservations.Delete(key)
			return true
		})
	}
}

// pushGPUState reads GPU metrics and sends them to the Hub channel.
func (t *Tracker) pushGPUState() {
	metrics := t.backend.ReadMetrics()
	select {
	case t.gpuStateCh <- metrics:
	default:
		// Drop if Hub consumer is slow.
	}
}
