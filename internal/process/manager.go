package process

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
)

type swapRequest struct {
	model  string
	doneCh chan error
}

type procState struct {
	mu        sync.Mutex
	backend   config.Backend
	port      int
	proc      *os.Process
	healthy   bool
	swapCh    chan swapRequest
	startedAt time.Time
	restarts  int
}

// Manager supervises llama-server child processes for managed backends and
// health-checks external backends.
type Manager struct {
	cfg        *config.Config
	log        *slog.Logger
	cmdBuilder func(ctx context.Context, b config.Backend, port int) *exec.Cmd

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
	m.cmdBuilder = func(ctx context.Context, b config.Backend, port int) *exec.Cmd {
		return exec.CommandContext(ctx, "llama-server", buildArgs(b, port)...)
	}
	return m
}

// Events returns a receive-only channel of structured events from all managed
// backends. The channel is buffered (256); sends are non-blocking and events
// are dropped when the channel is full.
func (m *Manager) Events() <-chan monitor.ServerEvent {
	return m.eventCh
}

// Snapshot returns a point-in-time copy of the runtime state of all managed
// backends. It acquires read locks and is safe to call concurrently.
func (m *Manager) Snapshot() []monitor.BackendStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]monitor.BackendStatus, 0, len(m.procs))
	for name, ps := range m.procs {
		ps.mu.Lock()
		pid := 0
		if ps.proc != nil {
			pid = ps.proc.Pid
		}
		status := monitor.BackendStatus{
			Name:      name,
			Model:     ps.backend.Model,
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

// Start launches supervisor goroutines for all managed backends.
// It returns immediately; health checks happen asynchronously.
func (m *Manager) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.cancel = cancel

	for name, b := range m.cfg.Backends {
		if !b.IsManaged() {
			continue
		}
		ps := new(procState)
		ps.backend = b
		ps.swapCh = make(chan swapRequest, 1)

		m.mu.Lock()
		m.procs[name] = ps
		m.mu.Unlock()

		m.wg.Add(1)
		go m.supervise(ctx, name)
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

// Healthy reports whether a backend is ready to serve requests.
// For external backends it performs a best-effort GET /health.
func (m *Manager) Healthy(name string) bool {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()

	if ok {
		ps.mu.Lock()
		defer ps.mu.Unlock()
		return ps.healthy
	}

	b, ok := m.cfg.Backends[name]
	if !ok || !b.IsExternal() {
		return false
	}

	resp, err := http.Get(b.URL + "/health") //nolint:noctx
	if err != nil {
		// Best-effort: some external APIs don't implement /health.
		return true
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Addr returns the base URL for a backend.
// For managed backends it returns "http://localhost:<port>".
// For external backends it returns Backend.URL.
func (m *Manager) Addr(name string) (string, bool) {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()

	if ok {
		ps.mu.Lock()
		defer ps.mu.Unlock()
		if ps.port == 0 {
			return "", false
		}
		return fmt.Sprintf("http://localhost:%d", ps.port), true
	}

	b, ok := m.cfg.Backends[name]
	if !ok || !b.IsExternal() {
		return "", false
	}
	return b.URL, true
}

// Swap hot-swaps the model for a managed backend.
// It stops the running process, updates the model, starts a new process, and
// waits up to 120 seconds for the health check to pass.
func (m *Manager) Swap(ctx context.Context, name, model string) error {
	m.mu.RLock()
	ps, ok := m.procs[name]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("backend %q is not a managed backend", name)
	}

	doneCh := make(chan error, 1)
	select {
	case ps.swapCh <- swapRequest{model: model, doneCh: doneCh}:
	case <-ctx.Done():
		return ctx.Err()
	}

	select {
	case err := <-doneCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// supervise is the per-backend restart loop. It runs until ctx is cancelled
// or max retries are exceeded.
func (m *Manager) supervise(ctx context.Context, name string) {
	defer m.wg.Done()

	m.mu.RLock()
	ps := m.procs[name]
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
		launched    bool // false on the very first launch
	)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Reset the retry counter when the window has elapsed.
		if time.Since(windowStart) > retryWindow {
			attempts = 0
			windowStart = time.Now()
		}

		if attempts >= maxRetries {
			m.log.Error("max restart retries exceeded, giving up", "backend", name)
			return
		}

		if attempts > 0 {
			m.log.Info("restarting backend", "backend", name, "attempt", attempts)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}

		attempts++

		port, err := freePort()
		if err != nil {
			m.log.Error("port allocation failed", "backend", name, "error", err)
			continue
		}

		ps.mu.Lock()
		ps.port = port
		ps.healthy = false
		backend := ps.backend
		if launched {
			ps.restarts++
		}
		ps.startedAt = time.Now()
		ps.mu.Unlock()
		launched = true

		cmd, err := m.launch(ctx, name, backend, port)
		if err != nil {
			m.log.Error("failed to launch llama-server", "backend", name, "error", err)
			continue
		}

		ps.mu.Lock()
		ps.proc = cmd.Process
		ps.mu.Unlock()

		if err := waitHealthy(ctx, port, 120*time.Second); err != nil {
			m.log.Error("health check failed on start", "backend", name, "error", err)
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
		m.log.Info("backend healthy", "backend", name, "port", port)

		exitCh := make(chan error, 1)
		go func() { exitCh <- cmd.Wait() }()

	monitor:
		for {
			select {
			case <-ctx.Done():
				// cmd.Cancel sends SIGTERM; cmd.WaitDelay gives 10s then SIGKILL.
				<-exitCh
				return

			case req := <-ps.swapCh:
				ps.mu.Lock()
				ps.healthy = false
				ps.backend.Model = req.model
				backend = ps.backend
				ps.mu.Unlock()

				cmd.Process.Signal(syscall.SIGTERM) //nolint:errcheck
				select {
				case <-exitCh:
				case <-time.After(10 * time.Second):
					cmd.Process.Kill() //nolint:errcheck
					<-exitCh
				}

				ps.mu.Lock()
				ps.proc = nil
				ps.mu.Unlock()

				newPort, portErr := freePort()
				if portErr != nil {
					req.doneCh <- portErr
					break monitor
				}

				ps.mu.Lock()
				ps.port = newPort
				ps.restarts++
				ps.startedAt = time.Now()
				ps.mu.Unlock()

				newCmd, launchErr := m.launch(ctx, name, backend, newPort)
				if launchErr != nil {
					req.doneCh <- launchErr
					break monitor
				}

				ps.mu.Lock()
				ps.proc = newCmd.Process
				ps.mu.Unlock()

				if healthErr := waitHealthy(ctx, newPort, 120*time.Second); healthErr != nil {
					newCmd.Process.Kill() //nolint:errcheck
					newCmd.Wait()         //nolint:errcheck
					ps.mu.Lock()
					ps.proc = nil
					ps.mu.Unlock()
					req.doneCh <- healthErr
					break monitor
				}

				ps.mu.Lock()
				ps.healthy = true
				ps.mu.Unlock()
				m.log.Info("swap complete", "backend", name, "port", newPort, "model", req.model)
				req.doneCh <- nil

				cmd = newCmd
				exitCh = make(chan error, 1)
				go func() { exitCh <- cmd.Wait() }()

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
						m.log.Warn("process exited unexpectedly", "backend", name, "exit_code", ee.ExitCode())
					} else {
						m.log.Warn("process exited unexpectedly", "backend", name, "error", exitErr)
					}
				}
				break monitor
			}
		}
		// Fall through to retry.
	}
}

// launch starts a llama-server process and wires its stdout/stderr to slog.
func (m *Manager) launch(ctx context.Context, name string, b config.Backend, port int) (*exec.Cmd, error) {
	cmd := m.cmdBuilder(ctx, b, port)

	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return cmd.Process.Signal(syscall.SIGTERM)
	}
	cmd.WaitDelay = 10 * time.Second

	parser := monitor.NewEventParser(m.log, name, m.eventCh)
	cmd.Stdout = parser
	cmd.Stderr = parser

	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// buildArgs constructs the llama-server command-line arguments from a Backend config.
func buildArgs(b config.Backend, port int) []string {
	args := []string{
		"-hf", b.Model,
		"--port", strconv.Itoa(port),
		"--no-webui",
	}
	if b.GPULayers > 0 {
		args = append(args, "-ngl", strconv.Itoa(b.GPULayers))
	}
	if b.Context > 0 {
		args = append(args, "-c", strconv.Itoa(b.Context))
	}
	if b.Pooling != "" {
		args = append(args, "--embedding", "--pooling", b.Pooling)
	}
	return append(args, b.Extra...)
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
