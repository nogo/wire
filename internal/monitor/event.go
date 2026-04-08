package monitor

import "time"

// EventKind identifies the type of a ServerEvent.
type EventKind int

const (
	EventSlotAssign  EventKind = iota // slot N is processing
	EventSlotRelease                  // slot N released
	EventPromptEval                   // prompt eval time = ...
	EventGenEval                      // eval time = ... (generation)
	EventModelLoad                    // offloading N layers to GPU
	EventVRAMAlloc                    // buffer size = N MiB
	EventError                        // stderr error line
	EventRaw                          // unrecognized line
)

// ServerEvent is a structured event emitted by eventParser from a managed
// llama-server stderr stream.
type ServerEvent struct {
	Time       time.Time
	Backend    string
	Kind       EventKind
	Slot       int // -1 for system-level events
	Tokens     int
	TokPerSec  float64
	DurationMs float64
	Message    string // raw line for EventRaw, detail for others
}

// BackendStatus is a point-in-time snapshot of a managed backend's runtime state.
type BackendStatus struct {
	Name      string
	Model     string
	Port      int
	PID       int
	Healthy   bool
	Status    string // "loading", "healthy", "draining", "idle"
	StartedAt time.Time
	Restarts  int
}

// ModelInfo is a lightweight model descriptor returned by ListModels.
type ModelInfo struct {
	ID     string
	Status string // "loading", "healthy", "draining"
}
