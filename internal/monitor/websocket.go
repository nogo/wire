package monitor

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"nhooyr.io/websocket"
)

// HandleWS returns an HTTP handler that upgrades to WebSocket and streams hub
// messages as JSON. It sends a snapshot on connect, then forwards all
// subsequent hub messages until the client disconnects or the hub stops.
func HandleWS(hub *Hub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
		})
		if err != nil {
			return
		}
		defer conn.CloseNow()

		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()

		// Subscribe before reading the snapshot so no events are lost between
		// the two operations.
		ch, unsub := hub.Subscribe()
		defer unsub()

		// Drain the initial MsgSnapshot that Subscribe pushes to the channel.
		var hubSnap *HubSnapshot
		select {
		case msg := <-ch:
			if msg.Snap != nil {
				hubSnap = msg.Snap
			}
		case <-time.After(time.Second):
		}
		if hubSnap == nil {
			s := hub.Snapshot()
			hubSnap = &s
		}

		// Send the snapshot as the first wire message.
		snap := buildSnapshotMsg(hubSnap, hub.Statuses())
		if err := writeJSON(ctx, conn, snap); err != nil {
			return
		}

		// Discard goroutine: reads from the client only to detect close frames
		// and network drops. Cancels ctx so the main loop exits.
		go func() {
			defer cancel()
			for {
				if _, _, err := conn.Read(ctx); err != nil {
					return
				}
			}
		}()

		// Stream hub messages.
		for {
			select {
			case <-ctx.Done():
				return
			case msg, ok := <-ch:
				if !ok {
					// Hub was stopped.
					conn.Close(websocket.StatusGoingAway, "server stopping")
					return
				}
				out := toWireMsg(msg)
				if out == nil {
					continue
				}
				wCtx, wCancel := context.WithTimeout(ctx, 10*time.Second)
				writeErr := writeJSON(wCtx, conn, out)
				wCancel()
				if writeErr != nil {
					return
				}
			}
		}
	}
}

func writeJSON(ctx context.Context, conn *websocket.Conn, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, data)
}

// snapMsg is the initial state snapshot sent on connect.
type snapMsg struct {
	Type     string                     `json:"type"`
	Backends map[string]backendSnapInfo `json:"backends"`
	GPUs     []GPUMetrics               `json:"gpus"`
}

type backendSnapInfo struct {
	Model     string  `json:"model"`
	Status    string  `json:"status"`
	PID       int     `json:"pid"`
	Port      int     `json:"port"`
	UptimeSec float64 `json:"uptime_sec"`
	Restarts  int     `json:"restarts"`
}

func buildSnapshotMsg(snap *HubSnapshot, statuses []BackendStatus) snapMsg {
	backends := make(map[string]backendSnapInfo, len(statuses))
	for _, s := range statuses {
		status := "unhealthy"
		if s.Healthy {
			status = "healthy"
		}
		uptime := 0.0
		if !s.StartedAt.IsZero() {
			uptime = time.Since(s.StartedAt).Seconds()
		}
		backends[s.Name] = backendSnapInfo{
			Model:     s.Model,
			Status:    status,
			PID:       s.PID,
			Port:      s.Port,
			UptimeSec: uptime,
			Restarts:  s.Restarts,
		}
	}
	gpus := snap.GPUs
	if gpus == nil {
		gpus = []GPUMetrics{}
	}
	return snapMsg{Type: "snapshot", Backends: backends, GPUs: gpus}
}

// gpuWireMsg is the periodic GPU metrics update.
type gpuWireMsg struct {
	Type string       `json:"type"`
	GPUs []GPUMetrics `json:"gpus"`
}

// metricsWireMsg is the periodic per-backend metrics update.
type metricsWireMsg struct {
	Type       string          `json:"type"`
	Backend    string          `json:"backend"`
	BuildInfo  string          `json:"build_info,omitempty"`
	CtxUsePct  float64         `json:"ctx_usage_pct"`
	SlotsBusy  int             `json:"slots_busy"`
	SlotsTotal int             `json:"slots_total"`
	QueueDepth int             `json:"queue_depth"`
	GenTokS    float64         `json:"gen_tok_s"`
	PromptTokS float64         `json:"prompt_tok_s"`
	Slots      []slotWireEntry `json:"slots"`
}

type slotWireEntry struct {
	ID      int    `json:"id"`
	State   string `json:"state"`
	Tokens  int    `json:"tokens"`
	CtxUsed int    `json:"ctx_used"`
}

// eventWireMsg is a structured server event.
type eventWireMsg struct {
	Type       string  `json:"type"`
	Backend    string  `json:"backend"`
	Kind       string  `json:"kind"`
	Slot       int     `json:"slot"`
	Tokens     int     `json:"tokens"`
	TokS       float64 `json:"tok_s"`
	DurationMs float64 `json:"duration_ms"`
	Message    string  `json:"message"`
}

// stateWireMsg is a backend health state transition.
type stateWireMsg struct {
	Type    string `json:"type"`
	Backend string `json:"backend"`
	Status  string `json:"status"`
	Model   string `json:"model"`
}

// toWireMsg converts a HubMessage to the JSON-serialisable wire type.
// Returns nil for message kinds that have no wire representation.
func toWireMsg(msg HubMessage) any {
	switch msg.Kind {
	case MsgGPUTick:
		gpus := msg.GPUs
		if gpus == nil {
			gpus = []GPUMetrics{}
		}
		return gpuWireMsg{Type: "gpu", GPUs: gpus}

	case MsgMetricsTick:
		if msg.Metrics == nil {
			return nil
		}
		m := msg.Metrics
		slotsBusy := 0
		totalCtx := 0
		usedCtx := 0
		slots := make([]slotWireEntry, 0, len(m.Slots))
		for _, s := range m.Slots {
			state := "idle"
			if s.State == SlotProcessing {
				state = "generating"
				slotsBusy++
			}
			totalCtx += s.NCtx
			usedCtx += s.TokensPredicted
			slots = append(slots, slotWireEntry{
				ID:      s.ID,
				State:   state,
				Tokens:  s.TokensPredicted,
				CtxUsed: s.NCtx,
			})
		}
		ctxPct := 0.0
		if totalCtx > 0 {
			ctxPct = float64(usedCtx) / float64(totalCtx) * 100
		}
		return metricsWireMsg{
			Type:       "metrics",
			Backend:    m.Name,
			BuildInfo:  m.BuildInfo,
			CtxUsePct:  ctxPct,
			SlotsBusy:  slotsBusy,
			SlotsTotal: len(m.Slots),
			QueueDepth: m.RequestsPending,
			GenTokS:    m.TokensPerSecond,
			PromptTokS: m.PromptTokensPerSecond,
			Slots:      slots,
		}

	case MsgServerEvent:
		if msg.Event == nil {
			return nil
		}
		e := msg.Event
		return eventWireMsg{
			Type:       "event",
			Backend:    e.Backend,
			Kind:       eventKindStr(e.Kind),
			Slot:       e.Slot,
			Tokens:     e.Tokens,
			TokS:       e.TokPerSec,
			DurationMs: e.DurationMs,
			Message:    e.Message,
		}

	case MsgStateChange:
		status := "unhealthy"
		if msg.Healthy {
			status = "healthy"
		}
		return stateWireMsg{
			Type:    "state",
			Backend: msg.Backend,
			Status:  status,
		}

	default:
		return nil
	}
}

func eventKindStr(k EventKind) string {
	switch k {
	case EventSlotAssign:
		return "slot_assign"
	case EventSlotRelease:
		return "slot_release"
	case EventPromptEval:
		return "prompt_eval"
	case EventGenEval:
		return "gen_eval"
	case EventModelLoad:
		return "model_load"
	case EventVRAMAlloc:
		return "vram_alloc"
	case EventError:
		return "error"
	case EventRaw:
		return "raw"
	default:
		return "unknown"
	}
}
