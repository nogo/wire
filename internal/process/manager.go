package process

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/vram"
)

// modelState represents the lifecycle state of a managed model process.
type modelState int

const (
	stateIdle      modelState = iota // not loaded or failed
	stateLoading                     // llama-server starting up
	stateLoaded                      // loaded and serving requests
	stateUnloading                   // unloading: SIGTERM sent, waiting for exit
)

func (s modelState) String() string {
	switch s {
	case stateIdle:
		return "idle"
	case stateLoading:
		return "loading"
	case stateLoaded:
		return "loaded"
	case stateUnloading:
		return "unloading"
	default:
		return "unknown"
	}
}

type procState struct {
	mu        sync.Mutex
	modelName string
	modelPath string // resolved GGUF path; empty means use -hf modelName
	params    config.ModelParams
	port      int
	proc      *exec.Cmd
	startedAt time.Time
	restarts  int
	isWarmup  bool

	// State machine. state and ready are always updated together under mu.
	state   modelState
	ready   chan struct{} // closed when loading completes (success or failure); nil when idle
	loadErr error         // non-nil if the most recent load attempt failed

	// In-flight tracking and LRU.
	inflight atomic.Int64
	lastUsed time.Time

	// Per-model lifecycle control.
	cancel context.CancelFunc
	wg     sync.WaitGroup // Done when the supervisor goroutine exits
}

// addInflight increments the in-flight counter and updates lastUsed.
// Returns the address, release function, and no error.
func (ps *procState) addInflight(port int) (addr string, release func(), err error) {
	ps.inflight.Add(1)
	ps.mu.Lock()
	ps.lastUsed = time.Now()
	ps.mu.Unlock()
	return fmt.Sprintf("http://localhost:%d", port), func() { ps.inflight.Add(-1) }, nil
}

// Manager supervises llama-server child processes.
type Manager struct {
	cfg        *config.Config
	log        *slog.Logger
	cmdBuilder func(ctx context.Context, modelName string, modelPath string, params config.ModelParams) *exec.Cmd

	// healthTimeout overrides the 120s waitHealthy timeout. Zero means 120s.
	// Set in tests to speed up failure scenarios.
	healthTimeout time.Duration

	// vram is the optional VRAM tracker for admission control. nil = count-only.
	vram *vram.Tracker

	// drainTimeout is the max time to wait for in-flight requests to complete
	// before sending SIGTERM. Zero means 30s.
	drainTimeout time.Duration

	mu      sync.RWMutex
	procs   map[string]*procState
	ctx     context.Context    // manager root context (detached from any HTTP request)
	wg      sync.WaitGroup     // Done when all supervisors exit
	cancel  context.CancelFunc // cancels m.ctx
	eventCh chan monitor.ServerEvent
}

// New creates a Manager. It does not start any processes; call Start.
func New(cfg *config.Config, log *slog.Logger) *Manager {
	m := new(Manager)
	m.cfg = cfg
	m.log = log
	m.procs = make(map[string]*procState)
	m.eventCh = make(chan monitor.ServerEvent, 256)
	m.cmdBuilder = func(ctx context.Context, modelName string, modelPath string, params config.ModelParams) *exec.Cmd {
		return exec.CommandContext(ctx, "llama-server", buildArgs(modelName, modelPath, params)...)
	}
	return m
}

// SetCmdBuilder replaces the default llama-server launcher. Used in tests to
// inject a fake process builder without starting real llama-server instances.
func (m *Manager) SetCmdBuilder(fn func(ctx context.Context, modelName string, modelPath string, params config.ModelParams) *exec.Cmd) {
	m.cmdBuilder = fn
}

// SetVRAMTracker attaches a VRAM tracker for VRAM-aware admission control.
// When nil (default), only the max_models count limit is enforced.
func (m *Manager) SetVRAMTracker(t *vram.Tracker) { m.vram = t }

// effectiveDrainTimeout returns the configured drain timeout or the default 30s.
func (m *Manager) effectiveDrainTimeout() time.Duration {
	if m.drainTimeout > 0 {
		return m.drainTimeout
	}
	return 30 * time.Second
}

// drainAndCancel waits up to drainTimeout for in-flight requests to complete,
// then cancels the model's context to initiate SIGTERM. Must NOT hold m.mu.
func (m *Manager) drainAndCancel(ps *procState) {
	deadline := time.Now().Add(m.effectiveDrainTimeout())
	for ps.inflight.Load() > 0 {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			m.log.Warn("drain timeout exceeded, forcing shutdown", "model", ps.modelName)
			break
		}
		sleep := 10 * time.Millisecond
		if remaining < sleep {
			sleep = remaining
		}
		time.Sleep(sleep)
	}
	ps.cancel()
}

// Events returns a receive-only channel of structured events from all managed processes.
func (m *Manager) Events() <-chan monitor.ServerEvent {
	return m.eventCh
}

// Snapshot returns a point-in-time snapshot of all managed processes.
func (m *Manager) Snapshot() []monitor.BackendStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]monitor.BackendStatus, 0, len(m.procs))
	for _, ps := range m.procs {
		ps.mu.Lock()
		pid := 0
		if ps.proc != nil && ps.proc.Process != nil {
			pid = ps.proc.Process.Pid
		}
		status := monitor.BackendStatus{
			Name:      ps.modelName,
			Model:     ps.modelName,
			Port:      ps.port,
			PID:       pid,
			Loaded:    ps.state == stateLoaded,
			Status:    ps.state.String(),
			StartedAt: ps.startedAt,
			Restarts:  ps.restarts,
		}
		ps.mu.Unlock()
		result = append(result, status)
	}
	return result
}

// RunningModels returns model names of all currently healthy processes.
func (m *Manager) RunningModels() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var names []string
	for name, ps := range m.procs {
		ps.mu.Lock()
		healthy := ps.state == stateLoaded
		ps.mu.Unlock()
		if healthy {
			names = append(names, name)
		}
	}
	return names
}

// ListModels returns all known models in any state except idle.
func (m *Manager) ListModels() []monitor.ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]monitor.ModelInfo, 0, len(m.procs))
	for _, ps := range m.procs {
		ps.mu.Lock()
		state := ps.state
		name := ps.modelName
		ps.mu.Unlock()
		if state != stateIdle {
			result = append(result, monitor.ModelInfo{ID: name, Status: state.String()})
		}
	}
	return result
}

// Start initialises the manager's root context. Warmup models are loaded
// through the gateway's admission path (Gateway.Warmup), not here.
func (m *Manager) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.ctx = ctx
	m.cancel = cancel
	return nil
}

// startSupervisor creates a procState and launches the supervisor goroutine.
// Must NOT hold m.mu.
func (m *Manager) startSupervisor(modelName string, params config.ModelParams, isWarmup bool) error {
	modelCtx, modelCancel := context.WithCancel(m.ctx)
	ps := &procState{
		modelName: modelName,
		params:    params,
		state:     stateLoading,
		ready:     make(chan struct{}),
		cancel:    modelCancel,
		isWarmup:  isWarmup,
	}

	m.mu.Lock()
	m.procs[modelName] = ps
	m.mu.Unlock()

	m.wg.Add(1)
	ps.wg.Add(1)
	go m.supervise(modelCtx, modelName)
	return nil
}

// Reload applies newCfg to the running manager. Models whose params changed are
// drained and restarted; models no longer matching any pattern are drained and
// removed; new warmup entries are started. Models with unchanged params keep
// serving without interruption.
//
// Global fields that require a process restart (e.g. listen address) are logged
// as warnings but not applied.
func (m *Manager) Reload(newCfg *config.Config) {
	if m.cfg.Listen != newCfg.Listen {
		m.log.Warn("listen address changed; requires restart to take effect",
			"old", m.cfg.Listen, "new", newCfg.Listen)
	}

	type drainOp struct {
		name      string
		ps        *procState
		newParams config.ModelParams
		restart   bool
		isWarmup  bool
	}

	newWarmupSet := make(map[string]bool, len(newCfg.Warmup))
	for _, name := range newCfg.Warmup {
		newWarmupSet[name] = true
	}

	m.mu.Lock()
	m.cfg = newCfg

	var ops []drainOp
	for name, ps := range m.procs {
		ps.mu.Lock()
		state := ps.state
		oldParams := ps.params
		wasWarmup := ps.isWarmup
		ps.mu.Unlock()

		if state != stateLoaded && state != stateLoading {
			continue
		}

		newParams, ok := newCfg.MatchModel(name)
		if !ok {
			m.log.Info("draining removed model", "model", name)
			ps.mu.Lock()
			ps.state = stateUnloading
			ps.mu.Unlock()
			ops = append(ops, drainOp{name: name, ps: ps})
			continue
		}

		if !oldParams.Equal(newParams) {
			m.log.Info("reloading model", "model", name)
			ps.mu.Lock()
			ps.state = stateUnloading
			ps.mu.Unlock()
			ops = append(ops, drainOp{
				name:      name,
				ps:        ps,
				newParams: newParams,
				restart:   true,
				isWarmup:  wasWarmup || newWarmupSet[name],
			})
			continue
		}

		if wasWarmup && !newWarmupSet[name] {
			m.log.Info("draining removed warmup model", "model", name)
			ps.mu.Lock()
			ps.state = stateUnloading
			ps.mu.Unlock()
			ops = append(ops, drainOp{name: name, ps: ps})
		}
	}
	m.mu.Unlock()

	// Cancel all affected models simultaneously, then wait for each to exit.
	// This runs outside m.mu to avoid deadlocking with the supervisor goroutine.
	for i := range ops {
		ops[i].ps.cancel()
	}
	for i := range ops {
		ops[i].ps.wg.Wait()
	}

	m.mu.Lock()
	for i := range ops {
		delete(m.procs, ops[i].name)
	}
	m.mu.Unlock()

	// Restart models whose params changed.
	for _, op := range ops {
		if op.restart {
			if err := m.startSupervisor(op.name, op.newParams, op.isWarmup); err != nil {
				m.log.Error("failed to restart model after reload", "model", op.name, "error", err)
			}
		}
	}

	// Start new warmup models not currently in procs.
	for _, name := range newCfg.Warmup {
		m.mu.RLock()
		_, exists := m.procs[name]
		m.mu.RUnlock()
		if exists {
			continue
		}
		params, ok := newCfg.MatchModel(name)
		if !ok {
			continue
		}
		m.log.Info("starting new warmup model", "model", name)
		if err := m.startSupervisor(name, params, true); err != nil {
			m.log.Error("failed to start new warmup model", "model", name, "error", err)
		}
	}
}

// Stop cancels all managed processes and waits for them to exit.
func (m *Manager) Stop() error {
	if m.cancel != nil {
		m.cancel()
	}
	m.wg.Wait()
	return nil
}

// Loaded reports whether a model's process is ready to serve requests.
func (m *Manager) Loaded(name string) bool {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()
	if !ok {
		return false
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return ps.state == stateLoaded
}

// Addr returns the base URL for a healthy model's process.
func (m *Manager) Addr(name string) (string, bool) {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()
	if !ok {
		return "", false
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.port == 0 || ps.state != stateLoaded {
		return "", false
	}
	return fmt.Sprintf("http://localhost:%d", ps.port), true
}

// LoadOrGet returns a LoadTicket for the named model, loading it on demand if
// necessary. The ticket's Ready() channel receives the backend address when the
// model is healthy, or is closed on failure. Concurrent calls for the same
// unloaded model coalesce — one process starts and all waiters get the result.
//
// modelPath is the resolved GGUF file path; empty means use -hf modelName.
// The caller must call ticket.Release() (typically via defer) when the proxied
// request completes to decrement the in-flight counter.
func (m *Manager) LoadOrGet(modelName string, modelPath string, params config.ModelParams) *LoadTicket {
	ticket := newTicket()

	// Fast path: check under read lock to avoid goroutine overhead for
	// the common already-loaded case.
	m.mu.RLock()
	ps, ok := m.procs[modelName]
	m.mu.RUnlock()
	if ok {
		ps.mu.Lock()
		state := ps.state
		port := ps.port
		ready := ps.ready
		ps.mu.Unlock()

		switch state {
		case stateLoaded:
			addr, rel, _ := ps.addInflight(port)
			ticket.Succeed(addr, rel)
			return ticket
		case stateLoading:
			go func() {
				<-ready
				addr, rel, err := m.checkLoaded(modelName)
				if err != nil {
					ticket.Fail(err)
				} else {
					ticket.Succeed(addr, rel)
				}
			}()
			return ticket
		case stateUnloading:
			ticket.Fail(fmt.Errorf("%w: model %q", ErrModelDraining, modelName))
			return ticket
		}
	}

	var estimate uint64
	if m.vram != nil {
		estimate = m.vram.EstimateForModel(modelName)
	}
	go m.startLoad(ticket, modelName, modelPath, params, estimate)
	return ticket
}

// startLoad is the slow path of LoadOrGet: acquires the write lock, evicts if
// needed, and either joins an in-progress load or starts a new supervisor.
func (m *Manager) startLoad(ticket *LoadTicket, modelName string, modelPath string, params config.ModelParams, estimate uint64) {
	m.mu.Lock()

	// Re-check state under write lock; may have changed since fast-path check.
	if ps, ok := m.procs[modelName]; ok {
		ps.mu.Lock()
		state := ps.state
		ready := ps.ready
		port := ps.port
		ps.mu.Unlock()

		switch state {
		case stateLoaded:
			m.mu.Unlock()
			addr, rel, _ := ps.addInflight(port)
			ticket.Succeed(addr, rel)
			return
		case stateLoading:
			m.mu.Unlock()
			<-ready
			addr, rel, err := m.checkLoaded(modelName)
			if err != nil {
				ticket.Fail(err)
			} else {
				ticket.Succeed(addr, rel)
			}
			return
		case stateUnloading:
			m.mu.Unlock()
			ticket.Fail(fmt.Errorf("%w: model %q", ErrModelDraining, modelName))
			return
		case stateIdle:
			delete(m.procs, modelName)
		}
	}

	if err := m.evictIfNeeded(modelName, estimate); err != nil {
		m.mu.Unlock()
		ticket.Fail(err)
		return
	}

	// evictIfNeeded releases and re-acquires m.mu; re-check for concurrent loads.
	if ps2, ok := m.procs[modelName]; ok {
		ps2.mu.Lock()
		state2 := ps2.state
		ready2 := ps2.ready
		port2 := ps2.port
		ps2.mu.Unlock()
		switch state2 {
		case stateLoaded:
			m.mu.Unlock()
			addr, rel, _ := ps2.addInflight(port2)
			ticket.Succeed(addr, rel)
			return
		case stateLoading:
			m.mu.Unlock()
			<-ready2
			addr, rel, err := m.checkLoaded(modelName)
			if err != nil {
				ticket.Fail(err)
			} else {
				ticket.Succeed(addr, rel)
			}
			return
		}
	}

	// Start a new supervisor.
	modelCtx, modelCancel := context.WithCancel(m.ctx)
	ps := &procState{
		modelName: modelName,
		modelPath: modelPath,
		params:    params,
		state:     stateLoading,
		ready:     make(chan struct{}),
		cancel:    modelCancel,
		isWarmup:  false,
	}
	m.procs[modelName] = ps
	ready := ps.ready
	m.mu.Unlock()

	m.publishEvent(monitor.ServerEvent{
		Time:    time.Now(),
		Backend: modelName,
		Kind:    monitor.EventModelLoad,
		Message: "loading on demand",
	})

	m.wg.Add(1)
	ps.wg.Add(1)
	go m.supervise(modelCtx, modelName)

	<-ready
	addr, rel, err := m.checkLoaded(modelName)
	if err != nil {
		ticket.Fail(err)
	} else {
		ticket.Succeed(addr, rel)
	}
}

// checkLoaded reads the outcome of a completed load (after ready has closed).
func (m *Manager) checkLoaded(modelName string) (addr string, release func(), err error) {
	m.mu.RLock()
	ps, ok := m.procs[modelName]
	m.mu.RUnlock()

	if !ok {
		return "", func() {}, fmt.Errorf("model %q load failed", modelName)
	}

	ps.mu.Lock()
	state := ps.state
	loadErr := ps.loadErr
	port := ps.port
	ps.mu.Unlock()

	if state != stateLoaded {
		if loadErr != nil {
			return "", func() {}, fmt.Errorf("model %q load failed: %w", modelName, loadErr)
		}
		return "", func() {}, fmt.Errorf("model %q load failed", modelName)
	}

	return ps.addInflight(port)
}

// evictIfNeeded evicts the minimum set of idle models needed to admit a new
// model, enforcing both the max_models count limit and the VRAM budget.
//
// Must be called with m.mu write-locked. It releases and re-acquires m.mu
// while draining evicted supervisors, so the caller must re-validate procs
// state after it returns.
func (m *Manager) evictIfNeeded(modelName string, estimate uint64) error {
	type candidate struct {
		name     string
		ps       *procState
		lastUsed time.Time
		vramEst  uint64
	}
	var candidates []candidate
	running := 0

	for name, ps := range m.procs {
		ps.mu.Lock()
		state := ps.state
		lu := ps.lastUsed
		ps.mu.Unlock()

		if state != stateIdle {
			running++
		}
		if state == stateLoaded && ps.inflight.Load() == 0 {
			vramEst := uint64(0)
			if m.vram != nil {
				vramEst = m.vram.EstimateForModel(name)
			}
			candidates = append(candidates, candidate{name: name, ps: ps, lastUsed: lu, vramEst: vramEst})
		}
	}

	needCountEvict := m.cfg.MaxModels > 0 && running >= m.cfg.MaxModels
	needVRAMEvict := m.vram != nil && estimate > 0 && m.vram.FreeVRAM() < estimate

	if !needCountEvict && !needVRAMEvict {
		return nil
	}
	if len(candidates) == 0 {
		if needVRAMEvict && !needCountEvict {
			return fmt.Errorf("insufficient VRAM for %q; all loaded models have in-flight requests", modelName)
		}
		return fmt.Errorf("at capacity (%d/%d models); all loaded models have in-flight requests", running, m.cfg.MaxModels)
	}

	// Sort LRU: oldest first (evict least-recently-used first).
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].lastUsed.Before(candidates[j].lastUsed)
	})

	// Greedily select the minimum number of victims (LRU order) that satisfies
	// both the count constraint and the VRAM budget.
	var victims []candidate
	countSatisfied := !needCountEvict
	vramSatisfied := !needVRAMEvict
	free := uint64(0)
	if m.vram != nil {
		free = m.vram.FreeVRAM()
	}
	vramFreed := uint64(0)

	for _, c := range candidates {
		victims = append(victims, c)
		running--
		vramFreed += c.vramEst

		if !countSatisfied && running < m.cfg.MaxModels {
			countSatisfied = true
		}
		if !vramSatisfied && free+vramFreed >= estimate {
			vramSatisfied = true
		}
		if countSatisfied && vramSatisfied {
			break
		}
	}

	if !countSatisfied || !vramSatisfied {
		return fmt.Errorf("at capacity; cannot free enough resources for model %q", modelName)
	}

	// Mark all victims as unloading and release their VRAM reservations before
	// dropping the lock — new requests will be rejected immediately.
	for i := range victims {
		victims[i].ps.mu.Lock()
		victims[i].ps.state = stateUnloading
		victims[i].ps.mu.Unlock()
		if m.vram != nil {
			m.vram.Release(victims[i].name)
		}
		m.log.Info("evicting model (LRU)", "model", victims[i].name)
		m.publishEvent(monitor.ServerEvent{
			Time:    time.Now(),
			Backend: victims[i].name,
			Kind:    monitor.EventModelLoad,
			Message: "unloading (LRU eviction)",
		})
	}

	// Release the manager lock while draining all victims concurrently.
	m.mu.Unlock()
	for _, v := range victims {
		m.drainAndCancel(v.ps)
		v.ps.wg.Wait()
	}
	m.mu.Lock()
	for _, v := range victims {
		delete(m.procs, v.name)
	}
	return nil
}

// publishEvent sends an event to the buffered channel, dropping it if full.
func (m *Manager) publishEvent(evt monitor.ServerEvent) {
	select {
	case m.eventCh <- evt:
	default:
	}
}

// healthCheckTimeout returns the effective health check timeout.
func (m *Manager) effectiveHealthTimeout() time.Duration {
	if m.healthTimeout > 0 {
		return m.healthTimeout
	}
	return 120 * time.Second
}

// supervise is the per-model restart loop. It runs until ctx is cancelled
// or (for on-demand models) until the first load attempt fails or the process
// exits after becoming healthy.
func (m *Manager) supervise(ctx context.Context, modelName string) {
	defer m.wg.Done()

	m.mu.RLock()
	ps := m.procs[modelName]
	m.mu.RUnlock()

	// Ensure state is idle and the ready channel is closed on exit (for any
	// reason). This unblocks callers waiting in LoadOrGet and lets Loaded()
	// return false after the supervisor exits.
	defer func() {
		ps.wg.Done()
		ps.mu.Lock()
		ready := ps.ready
		ps.ready = nil
		// Only preserve stateUnloading; evictIfNeeded owns that transition.
		if ps.state != stateUnloading {
			ps.state = stateIdle
		}
		ps.mu.Unlock()
		if ready != nil {
			close(ready)
		}
	}()

	const (
		maxRetries  = 5
		retryWindow = 60 * time.Second
		backoff     = 2 * time.Second
	)

	var (
		attempts    int
		windowStart = time.Now()
		firstLoad   = true
	)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if time.Since(windowStart) > retryWindow {
			attempts = 0
			windowStart = time.Now()
		}
		if attempts >= maxRetries {
			m.log.Error("max restart retries exceeded, giving up", "model", modelName)
			return
		}

		if attempts > 0 && ps.isWarmup {
			// Before retrying a warmup model, create a fresh ready channel so
			// new callers waiting in LoadOrGet see the correct loading state.
			ps.mu.Lock()
			ps.state = stateLoading
			ps.loadErr = nil
			ps.ready = make(chan struct{})
			ps.mu.Unlock()

			m.log.Info("restarting model", "model", modelName, "attempt", attempts)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}

		attempts++

		ps.mu.Lock()
		if !firstLoad {
			ps.restarts++
		}
		ps.startedAt = time.Now()
		params := ps.params
		modelPath := ps.modelPath
		ps.mu.Unlock()

		cmd, parser, err := m.launch(ctx, modelName, modelPath, params)
		if err != nil {
			m.log.Error("failed to launch llama-server", "model", modelName, "error", err)
			if !ps.isWarmup {
				m.failLoad(ps, modelName, err)
				return
			}
			continue
		}

		// Wait for llama-server to report its listening port via stdout.
		var port int
		select {
		case port = <-parser.PortCh:
		case <-ctx.Done():
			cmd.Process.Kill()
			cmd.Wait() //nolint:errcheck
			return
		case <-time.After(m.effectiveHealthTimeout()):
			m.log.Error("timed out waiting for listening port", "model", modelName)
			cmd.Process.Kill()
			cmd.Wait() //nolint:errcheck
			if !ps.isWarmup {
				m.failLoad(ps, modelName, fmt.Errorf("timed out waiting for listening port"))
				return
			}
			continue
		}

		ps.mu.Lock()
		ps.port = port
		ps.proc = cmd
		ps.mu.Unlock()

		loadStart := time.Now()
		if err := waitHealthy(ctx, port, m.effectiveHealthTimeout()); err != nil {
			m.log.Error("health check failed on start", "model", modelName, "error", err)
			cmd.Process.Kill()
			cmd.Wait() //nolint:errcheck
			ps.mu.Lock()
			ps.proc = nil
			ps.mu.Unlock()
			if !ps.isWarmup {
				m.failLoad(ps, modelName, err)
				return
			}
			continue
		}

		loadDur := time.Since(loadStart)

		// Transition to healthy and signal all waiters by closing ready.
		ps.mu.Lock()
		ps.state = stateLoaded
		if firstLoad {
			ps.lastUsed = time.Now()
		}
		ready := ps.ready
		ps.ready = nil
		ps.mu.Unlock()

		if ready != nil {
			close(ready)
		}
		firstLoad = false

		if m.vram != nil {
			if est := m.vram.EstimateForModel(modelName); est > 0 {
				if !m.vram.Reserve(modelName, est) {
					m.log.Warn("VRAM reservation failed after model load", "model", modelName)
				}
			}
		}

		m.log.Info("model healthy", "model", modelName, "port", port, "load_duration", loadDur)
		m.publishEvent(monitor.ServerEvent{
			Time:       time.Now(),
			Backend:    modelName,
			Kind:       monitor.EventModelLoad,
			DurationMs: float64(loadDur.Milliseconds()),
			Message:    "loaded",
		})

		exitCh := make(chan error, 1)
		go func() { exitCh <- cmd.Wait() }()

		select {
		case <-ctx.Done():
			<-exitCh
			return

		case exitErr := <-exitCh:
			ps.mu.Lock()
			ps.state = stateIdle
			ps.proc = nil
			ps.mu.Unlock()

			if ctx.Err() != nil {
				return
			}

			oomKilled := false
			if exitErr != nil {
				if ee, ok := errors.AsType[*exec.ExitError](exitErr); ok {
					// OOM killer sends SIGKILL — detect via WaitStatus.
					if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signal() == syscall.SIGKILL {
						oomKilled = true
						m.log.Error("process killed by SIGKILL (likely OOM)", "model", modelName)
					} else {
						m.log.Warn("process exited unexpectedly", "model", modelName, "exit_code", ee.ExitCode())
					}
				} else {
					m.log.Warn("process exited unexpectedly", "model", modelName, "error", exitErr)
				}
			}

			if !ps.isWarmup {
				// On-demand model crashed; remove and stop.
				m.mu.Lock()
				delete(m.procs, modelName)
				m.mu.Unlock()
				return
			}

			// OOM: back off longer to let the system recover before reloading.
			if oomKilled {
				oomDelay := 30 * time.Second
				m.log.Warn("delaying restart after OOM kill", "model", modelName, "delay", oomDelay)
				select {
				case <-ctx.Done():
					return
				case <-time.After(oomDelay):
				}
			}
		}
		// Warmup model: fall through to retry loop.
	}
}

// failLoad signals a failed on-demand load to all waiters and cleans up.
func (m *Manager) failLoad(ps *procState, modelName string, err error) {
	m.log.Error("model load failed", "model", modelName, "error", err)
	m.publishEvent(monitor.ServerEvent{
		Time:    time.Now(),
		Backend: modelName,
		Kind:    monitor.EventError,
		Message: "load failed: " + err.Error(),
	})

	ps.mu.Lock()
	ps.loadErr = err
	ps.state = stateIdle
	ready := ps.ready
	ps.ready = nil
	ps.mu.Unlock()

	if ready != nil {
		close(ready)
	}

	m.mu.Lock()
	delete(m.procs, modelName)
	m.mu.Unlock()
}

// launch starts a llama-server process with --port 0 and wires its
// stdout/stderr to the event parser. The actual listening port is sent
// to parser.PortCh once llama-server prints its "listening on" line.
func (m *Manager) launch(ctx context.Context, modelName string, modelPath string, params config.ModelParams) (*exec.Cmd, *monitor.EventParser, error) {
	cmd := m.cmdBuilder(ctx, modelName, modelPath, params)

	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return cmd.Process.Signal(syscall.SIGTERM)
	}
	cmd.WaitDelay = 10 * time.Second

	parser := monitor.NewEventParser(m.log, modelName, m.eventCh)
	cmd.Stdout = parser
	cmd.Stderr = parser

	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}

	setOOMScore(cmd.Process.Pid, m.log)

	return cmd, parser, nil
}

// setOOMScore writes 1000 to /proc/<pid>/oom_score_adj so that the kernel
// prefers killing llama-server over host processes under memory pressure.
func setOOMScore(pid int, log *slog.Logger) {
	path := fmt.Sprintf("/proc/%d/oom_score_adj", pid)
	if err := os.WriteFile(path, []byte("1000"), 0); err != nil {
		log.Warn("failed to set OOM score; process unprotected", "pid", pid, "err", err)
	}
}

// buildArgs constructs the llama-server command-line arguments for a model.
// Port 0 tells llama-server to pick a free port; the actual port is parsed
// from stdout by the event parser.
// When modelPath is non-empty, -m <path> is used; otherwise -hf <modelName>.
func buildArgs(modelName string, modelPath string, params config.ModelParams) []string {
	slots := params.Slots
	if slots <= 0 {
		slots = 1
	}
	var args []string
	if modelPath != "" {
		args = []string{"-m", modelPath, "--port", "0", "--no-webui", "--metrics", "--parallel", strconv.Itoa(slots)}
	} else {
		args = []string{"-hf", modelName, "--port", "0", "--no-webui", "--metrics", "--parallel", strconv.Itoa(slots)}
	}
	if params.GPULayers > 0 {
		args = append(args, "-ngl", strconv.Itoa(params.GPULayers))
	}
	if params.Context > 0 {
		args = append(args, "-c", strconv.Itoa(params.Context))
	}
	if params.Pooling != "" {
		args = append(args, "--embedding", "--pooling", params.Pooling)
	}
	if params.KVType != "" {
		args = append(args, "--cache-type-k", params.KVType, "--cache-type-v", params.KVType)
	}
	return append(args, params.Extra...)
}

// waitHealthy polls GET /health on the given port every second until it
// returns 200 OK, the context is cancelled, or the timeout elapses.
func waitHealthy(ctx context.Context, port int, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	url := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 5 * time.Second}

	// Use a shorter poll interval when the timeout is short (test scenarios).
	interval := time.Second
	if timeout < time.Second {
		interval = timeout / 5
		if interval < 10*time.Millisecond {
			interval = 10 * time.Millisecond
		}
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			resp, err := client.Get(url) //nolint:noctx
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
		}
	}
}
