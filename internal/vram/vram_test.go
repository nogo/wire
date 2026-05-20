package vram

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nogo/wire/internal/monitor"
)

type mockBackend struct {
	discoverCalls atomic.Int32
	metricsCalls  atomic.Int32
	totalVRAM     uint64
	usedVRAM      atomic.Uint64
	isNoop        bool
	gpus          []monitor.GPUInfo
}

func (m *mockBackend) Discover() []monitor.GPUInfo {
	m.discoverCalls.Add(1)
	return m.gpus
}

func (m *mockBackend) ReadMetrics() []monitor.GPUMetrics {
	m.metricsCalls.Add(1)
	return []monitor.GPUMetrics{{
		Index:     0,
		Name:      "Mock GPU",
		VRAMTotal: m.totalVRAM,
		VRAMUsed:  m.usedVRAM.Load(),
	}}
}
func (m *mockBackend) TotalVRAM() uint64 { return m.totalVRAM }
func (m *mockBackend) UsedVRAM() uint64  { return m.usedVRAM.Load() }
func (m *mockBackend) IsNoop() bool      { return m.isNoop }

func newTestTracker(backend Backend, hfCacheDir string) (*Tracker, *HFCache) {
	hf := NewHFCache(hfCacheDir)
	tr := New(backend, hf)
	return tr, hf
}

func TestNoopBackend(t *testing.T) {
	backend := NoopBackend{}
	if !backend.IsNoop() {
		t.Error("IsNoop should return true")
	}
	if backend.TotalVRAM() != 0 {
		t.Error("noop total VRAM should be 0")
	}
}

func TestTracker_CanFit_NoopBackend(t *testing.T) {
	backend := NoopBackend{}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()
	if !tracker.CanFit(1 << 60) {
		t.Error("noop backend should always return CanFit=true")
	}
}

func TestTracker_CanFit_Budget(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 8 << 30,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	backend.usedVRAM.Store(2 << 30)

	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	// Budget = total - (actual_used + pending) = 8 - (2 + 0) = 6 GiB free
	if !tracker.CanFit(4 << 30) {
		t.Error("4 GiB should fit in 6 GiB free")
	}
	if !tracker.CanFit(6 << 30) {
		t.Error("6 GiB should fit in 6 GiB free")
	}
	if tracker.CanFit(7 << 30) {
		t.Error("7 GiB should not fit in 6 GiB free")
	}

	// Reserve 1 GiB → now in reservations map (not pending yet)
	tracker.Reserve("model-a", 1<<30)
	// After Reserve, the model is in reservations. CanFit doesn't count reservations.
	// CanFit counts pending (released but not yet confirmed by sysfs).
	// So budget is still 8 - (2 + 0) = 6 GiB.
	if !tracker.CanFit(6 << 30) {
		t.Error("6 GiB should still fit (reservations not counted in CanFit)")
	}

	// Release → pending = 1 GiB. Budget = 8 - (2 + 1) = 5 GiB.
	tracker.Release("model-a")
	if !tracker.CanFit(5 << 30) {
		t.Error("5 GiB should fit: 8 - (2 + 1 pending) = 5 GiB")
	}
	if tracker.CanFit(6 << 30) {
		t.Error("6 GiB should not fit: budget is 5 GiB")
	}
}

func TestTracker_Reserve_Release(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 4 << 30,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	if !tracker.Reserve("model-a", 2<<30) {
		t.Fatal("reservation should succeed")
	}
	if !tracker.Reserve("model-a", 2<<30) {
		t.Error("duplicate reserve should return true")
	}
}

func TestTracker_Reconciliation(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 8 << 30,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 2<<30)
	tracker.Release("model-a")
	backend.usedVRAM.Store(0)
	tracker.reconcile()

	found := false
	tracker.pending.Range(func(key, _ interface{}) bool {
		found = true
		return false
	})
	if found {
		t.Error("pending should be empty after reconciliation")
	}
}

func TestTracker_GPUStateCh(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 8 << 30,
		usedVRAM:  atomic.Uint64{},
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()
	tracker.Start()
	defer tracker.Stop()

	ch := tracker.GPUStateCh()
	select {
	case metrics := <-ch:
		if len(metrics) != 1 {
			t.Errorf("expected 1 GPU metric, got %d", len(metrics))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for GPU state on channel")
	}
}

func TestEstimateVRAM(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "model.gguf")
	data := make([]byte, 100<<20)
	os.WriteFile(path, data, 0o644)

	est, err := EstimateVRAM(path)
	if err != nil {
		t.Fatalf("EstimateVRAM failed: %v", err)
	}
	want := uint64(float64(100<<20) * 1.3)
	if est != want {
		t.Errorf("EstimateVRAM: got %d, want %d", est, want)
	}
}

func TestPlaceholderVRAM(t *testing.T) {
	// Use actual computed values
	size7b := knownModelSizes["7b"]
	want7b := uint64(float64(size7b) * vramMultiplier)
	size70b := knownModelSizes["70b"]
	want70b := uint64(float64(size70b) * vramMultiplier)
	tests := []struct {
		name string
		want uint64
	}{
		{"llama3:7b", want7b},
		{"llama3:70b", want70b},
		{"unknown-model", 0},
	}
	for _, tt := range tests {
		got := PlaceholderVRAM(tt.name)
		if got != tt.want {
			t.Errorf("PlaceholderVRAM(%q): got %d, want %d", tt.name, got, tt.want)
		}
	}
}

func TestHFCache_Scan(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := filepath.Join(cacheDir, "models", "llama3-7b")
	os.MkdirAll(modelDir, 0o755)
	ggufPath := filepath.Join(modelDir, "model.gguf")
	data := make([]byte, 2<<30)
	os.WriteFile(ggufPath, data, 0o644)

	hf := NewHFCache(cacheDir)
	results := hf.Scan()

	if len(results) != 1 {
		t.Fatalf("expected 1 estimate, got %d", len(results))
	}
	est, ok := results["llama3-7b"]
	if !ok {
		t.Fatal("expected estimate for llama3-7b")
	}
	fileSize := int64(2 << 30)
	want := uint64(float64(fileSize) * vramMultiplier)
	if est != want {
		t.Errorf("estimate: got %d, want %d", est, want)
	}

	if got := hf.Estimate("llama3-7b"); got != want {
		t.Errorf("Estimate: got %d, want %d", got, want)
	}
	if got := hf.Estimate("nonexistent"); got != 0 {
		t.Errorf("Estimate for nonexistent: got %d, want 0", got)
	}
}

func TestHFCache_EmptyDir(t *testing.T) {
	hf := NewHFCache("/nonexistent/path")
	results := hf.Scan()
	if len(results) != 0 {
		t.Errorf("expected 0 results for missing dir, got %d", len(results))
	}
}

func TestResolveModelPath(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := filepath.Join(cacheDir, "models", "deepseek-r1")
	os.MkdirAll(filepath.Join(modelDir, "blobs"), 0o755)
	ggufPath := filepath.Join(modelDir, "blobs", "model.gguf")
	os.WriteFile(ggufPath, []byte("fake"), 0o644)

	got := ResolveModelPath(cacheDir, "deepseek-r1")
	if got != ggufPath {
		t.Errorf("ResolveModelPath: got %q, want %q", got, ggufPath)
	}

	got = ResolveModelPath(cacheDir, "nonexistent")
	if got != "" {
		t.Errorf("ResolveModelPath for nonexistent: got %q, want empty", got)
	}
}

func TestTracker_EstimateForModel(t *testing.T) {
	backend := &mockBackend{isNoop: true}
	tracker, hf := newTestTracker(backend, "")
	tracker.Discover()

	tracker.SetEstimate("custom-model", 12345)
	if got := tracker.EstimateForModel("custom-model"); got != 12345 {
		t.Errorf("EstimateForModel: got %d, want 12345", got)
	}

	hf.estimates.Store("cached-model", uint64(99999))
	if got := tracker.EstimateForModel("cached-model"); got != 99999 {
		t.Errorf("EstimateForModel from HF cache: got %d, want 99999", got)
	}

	got := tracker.EstimateForModel("llama3:7b")
	if got == 0 {
		t.Error("EstimateForModel should return placeholder for 7b model")
	}
}

func Test_findGGUF(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "a", "b")
	os.MkdirAll(nested, 0o755)
	gguf := filepath.Join(nested, "model.gguf")
	os.WriteFile(gguf, []byte("x"), 0o644)

	got := findGGUF(dir)
	if got != gguf {
		t.Errorf("findGGUF: got %q, want %q", got, gguf)
	}

	empty := t.TempDir()
	if got := findGGUF(empty); got != "" {
		t.Errorf("findGGUF on empty dir: got %q, want empty", got)
	}
}

func TestPendingTotal(t *testing.T) {
	backend := &mockBackend{isNoop: true}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.pending.Store("a", uint64(100))
	tracker.pending.Store("b", uint64(200))

	got := tracker.pendingTotal()
	if got != 300 {
		t.Errorf("pendingTotal: got %d, want 300", got)
	}
}

func TestReservedTotal(t *testing.T) {
	backend := &mockBackend{isNoop: true}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.reservations.Store("a", uint64(100))
	tracker.reservations.Store("b", uint64(200))

	got := tracker.reservedTotal()
	if got != 300 {
		t.Errorf("reservedTotal: got %d, want 300", got)
	}
}

func TestReconcileClearsStaleReservations(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 8 << 30,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	backend.usedVRAM.Store(1 << 30)

	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.reservations.Store("stale-model", uint64(4<<30))
	tracker.reconcile()

	found := false
	tracker.reservations.Range(func(key, _ interface{}) bool {
		found = true
		return false
	})
	if found {
		t.Error("reservation should be cleared after reconciliation")
	}
}

func TestTracker_CanFit_OverBudget(t *testing.T) {
	backend := &mockBackend{
		totalVRAM: 8 << 30,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	// sysfs reports more used than totalVRAM — must not underflow to maxUint64
	backend.usedVRAM.Store(10 << 30)

	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	if tracker.CanFit(1) {
		t.Error("CanFit should return false when usedVRAM > totalVRAM")
	}
}

// TestReconcile_PendingClearedWhenSysfsDropsToReserved covers the common workload
// case: two models are released (pending) while a third stays loaded (reserved).
// Once sysfs confirms both released models have freed their VRAM (actualUsed == reserved),
// the pending map must be emptied and the reserved entry must survive.
func TestReconcile_PendingClearedWhenSysfsDropsToReserved(t *testing.T) {
	const total = 8 << 30
	backend := &mockBackend{
		totalVRAM: total,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 2<<30)
	tracker.Reserve("model-b", 2<<30)
	tracker.Reserve("model-c", 2<<30)
	backend.usedVRAM.Store(6 << 30) // all three loaded

	tracker.Release("model-a")
	tracker.Release("model-b")
	// sysfs now shows only model-c's footprint — a and b freed
	backend.usedVRAM.Store(2 << 30)

	tracker.reconcile()

	var pendingKeys []interface{}
	tracker.pending.Range(func(k, _ interface{}) bool {
		pendingKeys = append(pendingKeys, k)
		return true
	})
	if len(pendingKeys) != 0 {
		t.Errorf("pending should be empty after sysfs confirms drop, got keys: %v", pendingKeys)
	}

	if _, ok := tracker.reservations.Load("model-c"); !ok {
		t.Error("model-c reservation must survive reconcile")
	}
}

// TestReconcile_PendingStaysWhenSysfsUnchanged verifies that reconcile does not
// prematurely clear a pending entry when the GPU has not yet freed the VRAM
// (sysfs still reflects the released model's footprint).
func TestReconcile_PendingStaysWhenSysfsUnchanged(t *testing.T) {
	const total = 8 << 30
	backend := &mockBackend{
		totalVRAM: total,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 2<<30)
	tracker.Reserve("model-b", 2<<30)
	backend.usedVRAM.Store(4 << 30) // both loaded

	// Release model-a but sysfs hasn't moved — GPU hasn't freed it yet.
	tracker.Release("model-a")
	backend.usedVRAM.Store(4 << 30)

	tracker.reconcile()

	if _, ok := tracker.pending.Load("model-a"); !ok {
		t.Error("model-a pending entry must not be cleared while sysfs still shows its footprint")
	}
}

// TestReconcile_MultipleStaleReservations verifies that all reservations are wiped
// when sysfs reports less than the combined reserved total — the external OOM-kill
// (or manual unload) scenario with more than one model affected.
func TestReconcile_MultipleStaleReservations(t *testing.T) {
	const total = 12 << 30
	backend := &mockBackend{
		totalVRAM: total,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 4<<30)
	tracker.Reserve("model-b", 4<<30)
	tracker.Reserve("model-c", 4<<30)
	// sysfs shows far less — all three were killed externally
	backend.usedVRAM.Store(1 << 30)

	tracker.reconcile()

	var reservedKeys []interface{}
	tracker.reservations.Range(func(k, _ interface{}) bool {
		reservedKeys = append(reservedKeys, k)
		return true
	})
	if len(reservedKeys) != 0 {
		t.Errorf("all stale reservations must be cleared, remaining: %v", reservedKeys)
	}
}

// TestReconcile_FreeVRAMReflectsPendingThenClears verifies that FreeVRAM correctly
// accounts for pending VRAM before reconcile and recovers the full budget afterward.
func TestReconcile_FreeVRAMReflectsPendingThenClears(t *testing.T) {
	const total = 8 << 30
	backend := &mockBackend{
		totalVRAM: total,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 4<<30)
	backend.usedVRAM.Store(4 << 30)

	tracker.Release("model-a")
	// sysfs still shows 4GiB (not freed yet); pending=4GiB, reserved=0
	// FreeVRAM = 8 - (4 + 4) = 0
	if got := tracker.FreeVRAM(); got != 0 {
		t.Errorf("FreeVRAM before reconcile: got %d, want 0", got)
	}

	// GPU frees the VRAM.
	backend.usedVRAM.Store(0)
	tracker.reconcile()

	// FreeVRAM = 8 - (0 + 0) = 8GiB
	if got := tracker.FreeVRAM(); got != total {
		t.Errorf("FreeVRAM after reconcile: got %d, want %d", got, uint64(total))
	}
}

// TestReconcile_AdmissionUnblockedAfterPendingClears verifies that a reservation
// that was blocked (because pending VRAM inflated the budget) succeeds once
// reconcile confirms the pending model's VRAM was freed.
func TestReconcile_AdmissionUnblockedAfterPendingClears(t *testing.T) {
	const total = 4 << 30
	backend := &mockBackend{
		totalVRAM: total,
		gpus:      []monitor.GPUInfo{{Index: 0, Name: "Mock"}},
	}
	tracker, _ := newTestTracker(backend, "")
	tracker.Discover()

	tracker.Reserve("model-a", 3<<30)
	backend.usedVRAM.Store(3 << 30)

	tracker.Release("model-a")
	// pending=3GiB, actualUsed=3GiB → budget = 4-(3+3) underflows → CanFit(2GiB)=false
	if tracker.CanFit(2 << 30) {
		t.Error("CanFit should be false while pending VRAM is still counted")
	}

	// GPU confirms the release.
	backend.usedVRAM.Store(0)
	tracker.reconcile()

	if !tracker.CanFit(2 << 30) {
		t.Error("CanFit should be true after reconcile clears pending")
	}
	if !tracker.Reserve("model-b", 2<<30) {
		t.Error("Reserve should succeed after pending is cleared")
	}
}
