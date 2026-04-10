package monitor

import (
	"sync"
	"sync/atomic"
)

// RemoteStats holds telemetry for a single remote model.
type RemoteStats struct {
	Provider  string
	InFlight  int64 // current concurrent requests
	Requests  int64 // total completed requests
	Errors    int64 // total non-2xx responses
	LatencyNs int64 // cumulative latency in nanoseconds (for averaging)
}

// AvgLatencyMs returns the average request latency in milliseconds.
func (s RemoteStats) AvgLatencyMs() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.LatencyNs) / float64(s.Requests) / 1e6
}

// RemoteTracker tracks in-flight requests and latency for remote provider models.
// Safe for concurrent use.
type RemoteTracker struct {
	mu    sync.Mutex
	stats map[string]*remoteEntry
}

type remoteEntry struct {
	provider  string
	inFlight  atomic.Int64
	requests  atomic.Int64
	errors    atomic.Int64
	latencyNs atomic.Int64
}

// NewRemoteTracker creates a new tracker.
func NewRemoteTracker() *RemoteTracker {
	return &RemoteTracker{stats: make(map[string]*remoteEntry)}
}

// entry returns or creates the entry for the given model.
func (t *RemoteTracker) entry(model, provider string) *remoteEntry {
	t.mu.Lock()
	defer t.mu.Unlock()
	e, ok := t.stats[model]
	if !ok {
		e = &remoteEntry{provider: provider}
		t.stats[model] = e
	}
	return e
}

// Start records the beginning of a remote request. Returns a function to call
// when the request completes, passing the duration in nanoseconds and the HTTP
// status code.
func (t *RemoteTracker) Start(model, provider string) func(durationNs int64, statusCode int) {
	e := t.entry(model, provider)
	e.inFlight.Add(1)
	return func(durationNs int64, statusCode int) {
		e.inFlight.Add(-1)
		e.requests.Add(1)
		e.latencyNs.Add(durationNs)
		if statusCode < 200 || statusCode >= 300 {
			e.errors.Add(1)
		}
	}
}

// Snapshot returns a copy of all tracked remote model stats.
func (t *RemoteTracker) Snapshot() map[string]RemoteStats {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]RemoteStats, len(t.stats))
	for model, e := range t.stats {
		out[model] = RemoteStats{
			Provider:  e.provider,
			InFlight:  e.inFlight.Load(),
			Requests:  e.requests.Load(),
			Errors:    e.errors.Load(),
			LatencyNs: e.latencyNs.Load(),
		}
	}
	return out
}
