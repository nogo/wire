package monitor

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/coder/websocket"
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
	Type      string                        `json:"type"`
	Locals    map[string]localSnapInfo      `json:"locals"`
	GPUs      []GPUMetrics                  `json:"gpus"`
	Remote    map[string]remoteSnapInfo     `json:"remote,omitempty"`
	Discovery map[string]discoveryWireEntry `json:"discovery,omitempty"`
	Models    []ModelInfo                   `json:"models,omitempty"`
}

type localSnapInfo struct {
	Model     string  `json:"model"`
	Status    string  `json:"status"`
	PID       int     `json:"pid"`
	Port      int     `json:"port"`
	UptimeSec float64 `json:"uptime_sec"`
	Restarts  int     `json:"restarts"`
}

type remoteSnapInfo struct {
	Provider  string  `json:"provider"`
	InFlight  int64   `json:"in_flight"`
	Requests  int64   `json:"requests"`
	Errors    int64   `json:"errors"`
	LatencyMs float64 `json:"latency_ms"`
}

func buildSnapshotMsg(snap *HubSnapshot, statuses []BackendStatus) snapMsg {
	locals := make(map[string]localSnapInfo, len(statuses))
	for _, s := range statuses {
		status := "unloaded"
		if s.Loaded {
			status = "loaded"
		}
		uptime := 0.0
		if !s.StartedAt.IsZero() {
			uptime = time.Since(s.StartedAt).Seconds()
		}
		locals[s.Name] = localSnapInfo{
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

	var remote map[string]remoteSnapInfo
	if len(snap.Remote) > 0 {
		remote = make(map[string]remoteSnapInfo, len(snap.Remote))
		for name, rs := range snap.Remote {
			remote[name] = remoteSnapInfo{
				Provider:  rs.Provider,
				InFlight:  rs.InFlight,
				Requests:  rs.Requests,
				Errors:    rs.Errors,
				LatencyMs: rs.AvgLatencyMs(),
			}
		}
	}

	var discovery map[string]discoveryWireEntry
	if len(snap.Models) > 0 {
		discovery = buildModelsEntries(snap.Models)
	}

	return snapMsg{Type: "snapshot", Locals: locals, GPUs: gpus, Remote: remote, Discovery: discovery, Models: snap.Models}
}

func buildModelsEntries(models []ModelInfo) map[string]discoveryWireEntry {
	out := make(map[string]discoveryWireEntry)
	byProvider := make(map[string][]string)
	for _, m := range models {
		if m.Type == "remote" && m.Provider != "" {
			byProvider[m.Provider] = append(byProvider[m.Provider], m.ID)
		}
	}
	for provider, ids := range byProvider {
		out[provider] = discoveryWireEntry{
			ModelCount: len(ids),
			Models:     ids,
		}
	}
	return out
}

// gpuWireMsg is the periodic GPU metrics update.
type gpuWireMsg struct {
	Type string       `json:"type"`
	GPUs []GPUMetrics `json:"gpus"`
}

// metricsWireMsg is the periodic per-local-model metrics update.
type metricsWireMsg struct {
	Type       string          `json:"type"`
	Local      string          `json:"local"`
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
	Local      string  `json:"local"`
	Kind       string  `json:"kind"`
	Slot       int     `json:"slot"`
	Tokens     int     `json:"tokens"`
	TokS       float64 `json:"tok_s"`
	DurationMs float64 `json:"duration_ms"`
	Message    string  `json:"message"`
}

// stateWireMsg is a local model health state transition.
type stateWireMsg struct {
	Type   string `json:"type"`
	Local  string `json:"local"`
	Status string `json:"status"`
	Model  string `json:"model"`
}

// remoteWireMsg is the periodic remote provider metrics update.
type remoteWireMsg struct {
	Type   string                    `json:"type"`
	Remote map[string]remoteSnapInfo `json:"remote"`
}

// discoveryWireEntry is one provider's discovered model list.
type discoveryWireEntry struct {
	ModelCount int      `json:"model_count"`
	Models     []string `json:"models"`
}

// modelsWireMsg is the periodic models update.
type modelsWireMsg struct {
	Type   string      `json:"type"`
	Models []ModelInfo `json:"models"`
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
			Local:      m.Name,
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
			Local:      e.Backend,
			Kind:       eventKindStr(e.Kind),
			Slot:       e.Slot,
			Tokens:     e.Tokens,
			TokS:       e.TokPerSec,
			DurationMs: e.DurationMs,
			Message:    e.Message,
		}

	case MsgStateChange:
		status := "unloaded"
		if msg.Loaded {
			status = "loaded"
		}
		return stateWireMsg{
			Type:   "state",
			Local:  msg.Backend,
			Status: status,
		}

	case MsgRemoteTick:
		if len(msg.Remote) == 0 {
			return nil
		}
		remote := make(map[string]remoteSnapInfo, len(msg.Remote))
		for name, rs := range msg.Remote {
			remote[name] = remoteSnapInfo{
				Provider:  rs.Provider,
				InFlight:  rs.InFlight,
				Requests:  rs.Requests,
				Errors:    rs.Errors,
				LatencyMs: rs.AvgLatencyMs(),
			}
		}
		return remoteWireMsg{Type: "remote", Remote: remote}

	case MsgModelsTick:
		if len(msg.Models) == 0 {
			return nil
		}
		return modelsWireMsg{Type: "models", Models: msg.Models}

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
