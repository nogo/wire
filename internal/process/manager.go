package process

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
)

type procState struct {
	mu        sync.Mutex
	modelName string
	params    config.ModelParams
	port      int
	proc      *exec.Cmd
	healthy   bool
	startedAt time.Time
	restarts  int
}

// Manager supervises llama-server child processes for warmup models.
type Manager struct {
	cfg        *config.Config
	log        *slog.Logger
	cmdBuilder func(ctx context.Context, modelName string, params config.ModelParams, port int) *exec.Cmd

	mu      sync.RWMutex
	procs   map[string]*procState
	wg      sync.WaitGroup
	cancel  context.CancelFunc
	eventCh chan monitor.ServerEvent
}

// New creates a Manager. It does not start any processes; call Start.
func New(cfg *config.Config, log *slog.Logger) *Manager {
	m := new(Manager)
	m.cfg = cfg
	m.log = log
	m.procs = make(map[string]*procState)
	m.eventCh = make(chan monitor.ServerEvent, 256)
	m.cmdBuilder = func(ctx context.Context, modelName string, params config.ModelParams, port int) *exec.Cmd {
		return exec.CommandContext(ctx, "llama-server", buildArgs(modelName, params, port)...)
	}
	return m
}

// Events returns a receive-only channel of structured events from all managed
// processes. The channel is buffered (256); sends are non-blocking and events
// are dropped when the channel is full.
func (m *Manager) Events() <-chan monitor.ServerEvent {
	return m.eventCh
}

// Snapshot returns a point-in-time copy of the runtime state of all managed
// processes. It acquires read locks and is safe to call concurrently.
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
			Healthy:   ps.healthy,
			StartedAt: ps.startedAt,
			Restarts:  ps.restarts,
		}
		ps.mu.Unlock()
		result = append(result, status)
	}
	return result
}

// RunningModels returns the model names of all currently healthy processes.
func (m *Manager) RunningModels() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var names []string
	for name, ps := range m.procs {
		ps.mu.Lock()
		healthy := ps.healthy
		ps.mu.Unlock()
		if healthy {
			names = append(names, name)
		}
	}
	return names
}

// Start launches supervisor goroutines for all warmup models.
// It returns immediately; health checks happen asynchronously.
func (m *Manager) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.cancel = cancel

	for _, modelName := range m.cfg.Warmup {
		params, ok := m.cfg.MatchModel(modelName)
		if !ok {
			cancel()
			return fmt.Errorf("warmup model %q does not match any pattern", modelName)
		}
		ps := &procState{
			modelName: modelName,
			params:    params,
		}

		m.mu.Lock()
		m.procs[modelName] = ps
		m.mu.Unlock()

		m.wg.Add(1)
		go m.supervise(ctx, modelName)
	}
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
	return ps.healthy
}

// Addr returns the base URL for a model's process.
func (m *Manager) Addr(name string) (string, bool) {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()
	if !ok {
		return "", false
	}
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.port == 0 {
		return "", false
	}
	return fmt.Sprintf("http://localhost:%d", ps.port), true
}

// supervise is the per-model restart loop. It runs until ctx is cancelled
// or max retries are exceeded.
func (m *Manager) supervise(ctx context.Context, modelName string) {
	defer m.wg.Done()

	m.mu.RLock()
	ps := m.procs[modelName]
	m.mu.RUnlock()

	defer func() {
		ps.mu.Lock()
		ps.healthy = false
		ps.proc = nil
		ps.mu.Unlock()
	}()

	const (
		maxRetries  = 5
		retryWindow = 60 * time.Second
		backoff     = 2 * time.Second
	)

	var (
		attempts    int
		windowStart = time.Now()
		launched    bool
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

		if attempts > 0 {
			m.log.Info("restarting model", "model", modelName, "attempt", attempts)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}

		attempts++

		port, err := freePort()
		if err != nil {
			m.log.Error("port allocation failed", "model", modelName, "error", err)
			continue
		}

		ps.mu.Lock()
		ps.port = port
		ps.healthy = false
		params := ps.params
		if launched {
			ps.restarts++
		}
		ps.startedAt = time.Now()
		ps.mu.Unlock()
		launched = true

		cmd, err := m.launch(ctx, modelName, params, port)
		if err != nil {
			m.log.Error("failed to launch llama-server", "model", modelName, "error", err)
			continue
		}

		ps.mu.Lock()
		ps.proc = cmd
		ps.mu.Unlock()

		if err := waitHealthy(ctx, port, 120*time.Second); err != nil {
			m.log.Error("health check failed on start", "model", modelName, "error", err)
			cmd.Process.Kill()
			cmd.Wait() //nolint:errcheck
			ps.mu.Lock()
			ps.proc = nil
			ps.mu.Unlock()
			continue
		}

		ps.mu.Lock()
		ps.healthy = true
		ps.mu.Unlock()
		m.log.Info("model healthy", "model", modelName, "port", port)

		exitCh := make(chan error, 1)
		go func() { exitCh <- cmd.Wait() }()

		select {
		case <-ctx.Done():
			<-exitCh
			return

		case exitErr := <-exitCh:
			ps.mu.Lock()
			ps.healthy = false
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
		}
		// Fall through to retry.
	}
}

// launch starts a llama-server process and wires its stdout/stderr to slog.
func (m *Manager) launch(ctx context.Context, modelName string, params config.ModelParams, port int) (*exec.Cmd, error) {
	cmd := m.cmdBuilder(ctx, modelName, params, port)

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
		return nil, err
	}
	return cmd, nil
}

// buildArgs constructs the llama-server command-line arguments for a model.
func buildArgs(modelName string, params config.ModelParams, port int) []string {
	slots := params.Slots
	if slots <= 0 {
		slots = 1
	}
	args := []string{
		"-hf", modelName,
		"--port", strconv.Itoa(port),
		"--no-webui",
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

// freePort finds a free TCP port by binding to :0 and immediately releasing it.
func freePort() (int, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port, nil
}

// waitHealthy polls GET /health on the given port every second until it
// returns 200 OK, the context is cancelled, or the timeout elapses.
func waitHealthy(ctx context.Context, port int, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	url := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 5 * time.Second}
	ticker := time.NewTicker(time.Second)
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
