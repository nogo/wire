package process

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os/exec"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
)

// modelState represents the lifecycle state of a managed model process.
type modelState int

const (
	stateIdle     modelState = iota // not loaded or failed
	stateLoading                    // llama-server starting up
	stateHealthy                    // serving requests
	stateDraining                   // SIGTERM sent, waiting for exit
)

func (s modelState) String() string {
	switch s {
	case stateIdle:
		return "idle"
	case stateLoading:
		return "loading"
	case stateHealthy:
		return "loaded"
	case stateDraining:
		return "unloading"
	default:
		return "unknown"
	}
}

type procState struct {
	mu        sync.Mutex
	modelName string
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

// Manager supervises llama-server child processes.
type Manager struct {
	cfg        *config.Config
	log        *slog.Logger
	cmdBuilder func(ctx context.Context, modelName string, params config.ModelParams) *exec.Cmd

	// healthTimeout overrides the 120s waitHealthy timeout. Zero means 120s.
	// Set in tests to speed up failure scenarios.
	healthTimeout time.Duration

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
	m.cmdBuilder = func(ctx context.Context, modelName string, params config.ModelParams) *exec.Cmd {
		return exec.CommandContext(ctx, "llama-server", buildArgs(modelName, params)...)
	}
	return m
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
			Healthy:   ps.state == stateHealthy,
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
		healthy := ps.state == stateHealthy
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

// Start launches supervisor goroutines for all warmup models.
// It returns immediately; health checks happen asynchronously.
func (m *Manager) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.ctx = ctx
	m.cancel = cancel

	for _, modelName := range m.cfg.Warmup {
		params, ok := m.cfg.MatchModel(modelName)
		if !ok {
			cancel()
			return fmt.Errorf("warmup model %q does not match any pattern", modelName)
		}
		if err := m.startSupervisor(modelName, params, true); err != nil {
			cancel()
			return err
		}
	}
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

// Stop cancels all managed processes and waits for them to exit.
func (m *Manager) Stop() error {
	if m.cancel != nil {
		m.cancel()
	}
	m.wg.Wait()
	return nil
}

// Healthy reports whether a model's process is ready to serve requests.
func (m *Manager) Healthy(name string) bool {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()
	if !ok {
		return false
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return ps.state == stateHealthy
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
	if ps.port == 0 || ps.state != stateHealthy {
		return "", false
	}
	return fmt.Sprintf("http://localhost:%d", ps.port), true
}

// LoadOrGet returns the address of the named model, loading it on demand if
// necessary. Concurrent calls for the same unloaded model coalesce — one
// process starts and all waiters get the address when it becomes healthy.
//
// The caller must call the returned release function (typically via defer) when
// the proxied request completes. This decrements the in-flight counter and
// allows the model to be evicted once idle.
//
// Errors:
//   - context.Canceled / context.DeadlineExceeded: caller's context expired
//   - "model is shutting down": model is being drained
//   - "at capacity …": max_models reached with all models in use
//   - "model … load failed: …": llama-server failed to start or become healthy
func (m *Manager) LoadOrGet(ctx context.Context, modelName string, params config.ModelParams) (addr string, release func(), err error) {
	noop := func() {}

	m.mu.Lock()

	if ps, ok := m.procs[modelName]; ok {
		ps.mu.Lock()
		state := ps.state
		ready := ps.ready
		port := ps.port
		ps.mu.Unlock()

		switch state {
		case stateHealthy:
			ps.inflight.Add(1)
			ps.mu.Lock()
			ps.lastUsed = time.Now()
			ps.mu.Unlock()
			m.mu.Unlock()
			return fmt.Sprintf("http://localhost:%d", port), func() { ps.inflight.Add(-1) }, nil

		case stateLoading:
			// Another goroutine is already loading this model. Wait for it.
			m.mu.Unlock()
			select {
			case <-ready:
			case <-ctx.Done():
				return "", noop, ctx.Err()
			}
			return m.checkLoaded(modelName)

		case stateDraining:
			m.mu.Unlock()
			return "", noop, fmt.Errorf("model %q is shutting down", modelName)

		case stateIdle:
			// Stale failed entry; clean up and reload.
			delete(m.procs, modelName)
		}
	}

	// Model not present (or was idle): check capacity, evict if needed, then load.
	// evictIfNeeded may release and re-acquire m.mu internally.
	if err := m.evictIfNeeded(); err != nil {
		m.mu.Unlock()
		return "", noop, err
	}

	// Re-check: another goroutine may have loaded this model while we evicted.
	if ps2, ok := m.procs[modelName]; ok {
		ps2.mu.Lock()
		state2 := ps2.state
		ready2 := ps2.ready
		port2 := ps2.port
		ps2.mu.Unlock()
		switch state2 {
		case stateHealthy:
			ps2.inflight.Add(1)
			ps2.mu.Lock()
			ps2.lastUsed = time.Now()
			ps2.mu.Unlock()
			m.mu.Unlock()
			return fmt.Sprintf("http://localhost:%d", port2), func() { ps2.inflight.Add(-1) }, nil
		case stateLoading:
			m.mu.Unlock()
			select {
			case <-ready2:
			case <-ctx.Done():
				return "", noop, ctx.Err()
			}
			return m.checkLoaded(modelName)
		}
	}

	// Create a new procState and start the supervisor.
	modelCtx, modelCancel := context.WithCancel(m.ctx)
	ps := &procState{
		modelName: modelName,
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

	select {
	case <-ready:
	case <-ctx.Done():
		return "", noop, ctx.Err()
	}

	return m.checkLoaded(modelName)
}

// checkLoaded reads the outcome of a completed load (after ready has closed).
func (m *Manager) checkLoaded(modelName string) (addr string, release func(), err error) {
	noop := func() {}

	m.mu.RLock()
	ps, ok := m.procs[modelName]
	m.mu.RUnlock()

	if !ok {
		return "", noop, fmt.Errorf("model %q load failed", modelName)
	}

	ps.mu.Lock()
	state := ps.state
	loadErr := ps.loadErr
	port := ps.port
	ps.mu.Unlock()

	if state != stateHealthy {
		if loadErr != nil {
			return "", noop, fmt.Errorf("model %q load failed: %w", modelName, loadErr)
		}
		return "", noop, fmt.Errorf("model %q load failed", modelName)
	}

	ps.inflight.Add(1)
	ps.mu.Lock()
	ps.lastUsed = time.Now()
	ps.mu.Unlock()
	return fmt.Sprintf("http://localhost:%d", port), func() { ps.inflight.Add(-1) }, nil
}

// evictIfNeeded evicts the LRU idle healthy model if we are at capacity.
//
// IMPORTANT: must be called with m.mu write-locked. It temporarily releases
// and re-acquires m.mu while waiting for the evicted supervisor to exit,
// so the caller must re-validate procs state after it returns.
func (m *Manager) evictIfNeeded() error {
	// Count non-idle models.
	running := 0
	for _, ps := range m.procs {
		ps.mu.Lock()
		s := ps.state
		ps.mu.Unlock()
		if s != stateIdle {
			running++
		}
	}
	if running < m.cfg.MaxModels {
		return nil
	}

	// Find LRU healthy model with no in-flight requests.
	type candidate struct {
		name     string
		ps       *procState
		lastUsed time.Time
	}
	var candidates []candidate
	for name, ps := range m.procs {
		ps.mu.Lock()
		state := ps.state
		lu := ps.lastUsed
		ps.mu.Unlock()
		if state == stateHealthy && ps.inflight.Load() == 0 {
			candidates = append(candidates, candidate{name: name, ps: ps, lastUsed: lu})
		}
	}
	if len(candidates) == 0 {
		return fmt.Errorf("at capacity (%d/%d models); all models have in-flight requests", running, m.cfg.MaxModels)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].lastUsed.Before(candidates[j].lastUsed)
	})

	victim := candidates[0]
	victim.ps.mu.Lock()
	victim.ps.state = stateDraining
	victim.ps.mu.Unlock()

	// Release the manager lock while waiting for the supervisor to exit.
	m.mu.Unlock()

	m.log.Info("evicting model (LRU)", "model", victim.name)
	m.publishEvent(monitor.ServerEvent{
		Time:    time.Now(),
		Backend: victim.name,
		Kind:    monitor.EventModelLoad,
		Message: "draining (LRU eviction)",
	})

	victim.ps.cancel()  // SIGTERM via context cancellation
	victim.ps.wg.Wait() // wait for supervisor goroutine to exit

	m.mu.Lock() // re-acquire for the caller
	delete(m.procs, victim.name)
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
	// reason). This unblocks callers waiting in LoadOrGet and lets Healthy()
	// return false after the supervisor exits.
	defer func() {
		ps.wg.Done()
		ps.mu.Lock()
		ready := ps.ready
		ps.ready = nil
		// Only preserve stateDraining; evictIfNeeded owns that transition.
		if ps.state != stateDraining {
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
		ps.mu.Unlock()

		cmd, parser, err := m.launch(ctx, modelName, params)
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
		ps.state = stateHealthy
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

			if exitErr != nil {
				if ee, ok := errors.AsType[*exec.ExitError](exitErr); ok {
					m.log.Warn("process exited unexpectedly", "model", modelName, "exit_code", ee.ExitCode())
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
func (m *Manager) launch(ctx context.Context, modelName string, params config.ModelParams) (*exec.Cmd, *monitor.EventParser, error) {
	cmd := m.cmdBuilder(ctx, modelName, params)

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
	return cmd, parser, nil
}

// buildArgs constructs the llama-server command-line arguments for a model.
// Port 0 tells llama-server to pick a free port; the actual port is parsed
// from stdout by the event parser.
func buildArgs(modelName string, params config.ModelParams) []string {
	slots := params.Slots
	if slots <= 0 {
		slots = 1
	}
	args := []string{
		"-hf", modelName,
		"--port", "0",
		"--no-webui",
		"--metrics",
		"--parallel", strconv.Itoa(slots),
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
