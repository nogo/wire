package monitor

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nhooyr.io/websocket"
)

func TestHandleWS_SnapshotOnConnect(t *testing.T) {
	events := make(chan ServerEvent, 8)
	hub := NewHub(slog.Default(), events, func() []BackendStatus {
		return []BackendStatus{{
			Name:    "test",
			Model:   "some-model",
			Healthy: true,
			Port:    12345,
		}}
	})
	hub.gpus = nil
	hub.Start()
	defer hub.Stop()

	srv := httptest.NewServer(http.HandlerFunc(HandleWS(hub)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	wsURL := strings.Replace(srv.URL, "http://", "ws://", 1)
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}

	var msg map[string]json.RawMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}

	var typ string
	if err := json.Unmarshal(msg["type"], &typ); err != nil || typ != "snapshot" {
		t.Errorf("first message type: got %q, want %q", typ, "snapshot")
	}

	var backends map[string]json.RawMessage
	if err := json.Unmarshal(msg["backends"], &backends); err != nil {
		t.Fatalf("backends field: %v", err)
	}
	if _, ok := backends["test"]; !ok {
		t.Error("expected backend 'test' in snapshot")
	}

	var gpus []json.RawMessage
	if err := json.Unmarshal(msg["gpus"], &gpus); err != nil {
		t.Fatalf("gpus field: %v", err)
	}
	// No GPUs in test environment.
	if len(gpus) != 0 {
		t.Errorf("gpus: got %d, want 0", len(gpus))
	}
}

func TestHandleWS_MetricsForwarded(t *testing.T) {
	events := make(chan ServerEvent, 8)
	hub := NewHub(slog.Default(), events, func() []BackendStatus { return nil })
	hub.gpus = nil
	hub.Start()
	defer hub.Stop()

	srv := httptest.NewServer(http.HandlerFunc(HandleWS(hub)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	wsURL := strings.Replace(srv.URL, "http://", "ws://", 1)
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	// Drain snapshot.
	if _, _, err := conn.Read(ctx); err != nil {
		t.Fatalf("read snapshot: %v", err)
	}

	// Inject a metrics message after a short delay to allow the handler to
	// reach its select loop.
	go func() {
		time.Sleep(30 * time.Millisecond)
		hub.broadcast(HubMessage{
			Kind: MsgMetricsTick,
			Metrics: &BackendMetrics{
				Name:            "local",
				BuildInfo:       "b8705-test",
				TokensPerSecond: 42.0,
				Slots: []Slot{
					{ID: 0, NCtx: 4096, TokensPredicted: 2048},
				},
			},
		})
	}()

	// Read until we get a metrics message (poll ticks may also arrive).
	deadline := time.After(3 * time.Second)
	for {
		var msg map[string]json.RawMessage
		_, data, readErr := conn.Read(ctx)
		if readErr != nil {
			t.Fatalf("read: %v", readErr)
		}
		if err := json.Unmarshal(data, &msg); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		var typ string
		json.Unmarshal(msg["type"], &typ) //nolint:errcheck
		if typ != "metrics" {
			select {
			case <-deadline:
				t.Fatal("timeout waiting for metrics message")
			default:
				continue
			}
		}

		var ctxPct float64
		if err := json.Unmarshal(msg["ctx_usage_pct"], &ctxPct); err != nil {
			t.Fatalf("ctx_usage_pct: %v", err)
		}
		if ctxPct != 50.0 {
			t.Errorf("ctx_usage_pct: got %v, want 50.0", ctxPct)
		}
		var buildInfo string
		if err := json.Unmarshal(msg["build_info"], &buildInfo); err != nil {
			t.Fatalf("build_info: %v", err)
		}
		if buildInfo != "b8705-test" {
			t.Errorf("build_info: got %q, want %q", buildInfo, "b8705-test")
		}
		return
	}
}

func TestHandleWS_HubStop_ClosesConnection(t *testing.T) {
	events := make(chan ServerEvent, 8)
	hub := NewHub(slog.Default(), events, func() []BackendStatus { return nil })
	hub.gpus = nil
	hub.Start()

	srv := httptest.NewServer(http.HandlerFunc(HandleWS(hub)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	wsURL := strings.Replace(srv.URL, "http://", "ws://", 1)
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	// Drain snapshot.
	if _, _, err := conn.Read(ctx); err != nil {
		t.Fatalf("read snapshot: %v", err)
	}

	// Stop the hub — the websocket handler must close the connection.
	hub.Stop()

	// The next read should fail because the server closed the connection.
	readCtx, readCancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer readCancel()
	_, _, err = conn.Read(readCtx)
	if err == nil {
		t.Error("expected connection close after hub stop, got nil error")
	}
}

func TestHandlePage(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/monitor", nil)
	w := httptest.NewRecorder()
	HandlePage(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if ct != "text/html; charset=utf-8" {
		t.Errorf("content-type: got %q, want %q", ct, "text/html; charset=utf-8")
	}
	if w.Body.Len() == 0 {
		t.Error("response body is empty")
	}
}
