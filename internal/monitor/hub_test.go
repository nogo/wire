package monitor

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// newTestHub creates a Hub wired to a fake snapshotFn and the provided event
// channel. No GPU discovery runs (discoverGPUs returns empty on most hosts).
func newTestHub(t *testing.T, events chan ServerEvent, backends []BackendStatus) *Hub {
	t.Helper()
	h := NewHub(slog.Default(), events, func() []BackendStatus { return backends })
	h.gpus = nil // disable GPU polling in unit tests
	return h
}

func TestHub_Subscribe_getsSnapshot(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)
	h.Start()
	defer h.Stop()

	ch, unsub := h.Subscribe()
	defer unsub()

	select {
	case msg := <-ch:
		if msg.Kind != MsgSnapshot {
			t.Errorf("first message kind: got %v, want MsgSnapshot", msg.Kind)
		}
		if msg.Snap == nil {
			t.Error("Snap must not be nil")
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for snapshot")
	}
}

func TestHub_EventRelay(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)
	h.Start()
	defer h.Stop()

	ch, unsub := h.Subscribe()
	defer unsub()

	// Drain the snapshot.
	<-ch

	ev := ServerEvent{Kind: EventRaw, Backend: "local", Message: "hello"}
	events <- ev

	select {
	case msg := <-ch:
		if msg.Kind != MsgServerEvent {
			t.Errorf("kind: got %v, want MsgServerEvent", msg.Kind)
		}
		if msg.Event == nil || msg.Event.Message != "hello" {
			t.Errorf("Event: %+v", msg.Event)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for server event")
	}
}

func TestHub_FanOut(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)
	h.Start()
	defer h.Stop()

	ch1, unsub1 := h.Subscribe()
	ch2, unsub2 := h.Subscribe()
	defer unsub1()
	defer unsub2()

	<-ch1
	<-ch2

	events <- ServerEvent{Kind: EventRaw, Message: "broadcast"}

	for _, ch := range []<-chan HubMessage{ch1, ch2} {
		select {
		case msg := <-ch:
			if msg.Kind != MsgServerEvent {
				t.Errorf("kind: got %v, want MsgServerEvent", msg.Kind)
			}
		case <-time.After(time.Second):
			t.Fatal("timeout waiting for broadcast")
		}
	}
}

func TestHub_Unsubscribe(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)
	h.Start()
	defer h.Stop()

	ch, unsub := h.Subscribe()
	<-ch // drain snapshot

	unsub()

	// Events sent after unsubscribe must not appear on ch.
	events <- ServerEvent{Kind: EventRaw, Message: "after-unsub"}

	select {
	case msg := <-ch:
		t.Errorf("received unexpected message after unsubscribe: %+v", msg)
	case <-time.After(100 * time.Millisecond):
		// expected
	}
}

func TestHub_Stop_closesChannels(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)
	h.Start()

	ch, _ := h.Subscribe()
	<-ch // drain snapshot

	h.Stop()

	// Channel must be closed after Stop.
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected channel to be closed, got value")
		}
	case <-time.After(time.Second):
		t.Fatal("channel not closed after Stop")
	}
}

func TestHub_Snapshot(t *testing.T) {
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, nil)

	snap := h.Snapshot()
	if snap.Backends == nil {
		t.Error("Backends must not be nil")
	}
	if snap.GPUs == nil {
		t.Error("GPUs must not be nil")
	}
}

func TestHub_PollMetrics(t *testing.T) {
	slots, _ := json.Marshal([]Slot{{ID: 0, State: SlotIdle}})
	metrics := `# TYPE llama_kv_cache_usage_ratio gauge
llama_kv_cache_usage_ratio 0.5
# TYPE llama_requests_processing gauge
llama_requests_processing 1
`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			w.Write([]byte(metrics)) //nolint:errcheck
		case "/slots":
			w.Write(slots) //nolint:errcheck
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	// Extract port from test server URL.
	var port int
	fmt.Sscanf(srv.URL, "http://127.0.0.1:%d", &port)

	backends := []BackendStatus{{Name: "local", Port: port, Healthy: true}}
	events := make(chan ServerEvent, 8)
	h := newTestHub(t, events, backends)
	h.Start()
	defer h.Stop()

	ch, unsub := h.Subscribe()
	defer unsub()
	<-ch // snapshot

	// Wait for a MsgMetricsTick.
	deadline := time.After(3 * time.Second)
	for {
		select {
		case msg := <-ch:
			if msg.Kind == MsgMetricsTick {
				if msg.Metrics == nil {
					t.Fatal("Metrics is nil")
				}
				if msg.Metrics.KVCacheUsageRatio != 0.5 {
					t.Errorf("KVCacheUsageRatio: got %v, want 0.5", msg.Metrics.KVCacheUsageRatio)
				}
				if msg.Metrics.RequestsProcessing != 1 {
					t.Errorf("RequestsProcessing: got %d, want 1", msg.Metrics.RequestsProcessing)
				}
				if len(msg.Metrics.Slots) != 1 {
					t.Errorf("Slots: got %d, want 1", len(msg.Metrics.Slots))
				}
				return
			}
		case <-deadline:
			t.Fatal("timeout waiting for MsgMetricsTick")
		}
	}
}
