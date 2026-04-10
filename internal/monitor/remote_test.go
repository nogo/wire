package monitor

import (
	"sync"
	"testing"
)

func TestRemoteTracker_StartEnd(t *testing.T) {
	tr := NewRemoteTracker()

	done := tr.Start("model-a", "openrouter")

	snap := tr.Snapshot()
	if snap["model-a"].InFlight != 1 {
		t.Errorf("in_flight = %d, want 1", snap["model-a"].InFlight)
	}

	done(50_000_000, 200) // 50ms, success

	snap = tr.Snapshot()
	s := snap["model-a"]
	if s.InFlight != 0 {
		t.Errorf("in_flight after done = %d, want 0", s.InFlight)
	}
	if s.Requests != 1 {
		t.Errorf("requests = %d, want 1", s.Requests)
	}
	if s.Errors != 0 {
		t.Errorf("errors = %d, want 0", s.Errors)
	}
	if s.Provider != "openrouter" {
		t.Errorf("provider = %q, want %q", s.Provider, "openrouter")
	}
	avg := s.AvgLatencyMs()
	if avg < 49 || avg > 51 {
		t.Errorf("avg latency = %.1f ms, want ~50", avg)
	}
}

func TestRemoteTracker_Errors(t *testing.T) {
	tr := NewRemoteTracker()

	done := tr.Start("model-a", "test")
	done(10_000_000, 500) // error

	snap := tr.Snapshot()
	if snap["model-a"].Errors != 1 {
		t.Errorf("errors = %d, want 1", snap["model-a"].Errors)
	}
}

func TestRemoteTracker_MultipleModels(t *testing.T) {
	tr := NewRemoteTracker()

	doneA := tr.Start("model-a", "prov-a")
	doneB := tr.Start("model-b", "prov-b")
	doneA(10_000_000, 200)
	doneB(20_000_000, 200)

	snap := tr.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("snapshot has %d entries, want 2", len(snap))
	}
	if snap["model-a"].Provider != "prov-a" {
		t.Errorf("model-a provider = %q", snap["model-a"].Provider)
	}
	if snap["model-b"].Provider != "prov-b" {
		t.Errorf("model-b provider = %q", snap["model-b"].Provider)
	}
}

func TestRemoteTracker_Concurrent(t *testing.T) {
	tr := NewRemoteTracker()
	const n = 100

	var wg sync.WaitGroup
	wg.Add(n)
	for range n {
		go func() {
			defer wg.Done()
			done := tr.Start("model", "prov")
			done(1_000_000, 200)
		}()
	}
	wg.Wait()

	snap := tr.Snapshot()
	s := snap["model"]
	if s.Requests != n {
		t.Errorf("requests = %d, want %d", s.Requests, n)
	}
	if s.InFlight != 0 {
		t.Errorf("in_flight = %d, want 0", s.InFlight)
	}
}

func TestRemoteTracker_SnapshotIsCopy(t *testing.T) {
	tr := NewRemoteTracker()
	done := tr.Start("model", "prov")
	done(1_000_000, 200)

	snap1 := tr.Snapshot()

	done2 := tr.Start("model", "prov")
	done2(1_000_000, 200)

	snap2 := tr.Snapshot()

	if snap1["model"].Requests != 1 {
		t.Error("snap1 should be a copy and not affected by later calls")
	}
	if snap2["model"].Requests != 2 {
		t.Error("snap2 should reflect both calls")
	}
}
