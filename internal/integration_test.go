package internal_test

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/gateway"
	"github.com/nogo/wire/internal/process"
)

// ---------------------------------------------------------------------------
// Subprocess helper: fake llama-server for integration tests.
//
// When GO_WANT_HELPER_PROCESS=1, the test binary acts as a minimal HTTP server
// that satisfies the Manager's health check and proxies any other request with
// a stub JSON response. This avoids starting real llama-server processes.
// ---------------------------------------------------------------------------

// TestHotReload_HelperProcess is NOT a real test — it is a subprocess target.
// It runs as the fake llama-server when invoked via integrationCmdBuilder.
func TestHotReload_HelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"fake","choices":[]}`)) //nolint:errcheck
	})

	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		fmt.Fprintf(os.Stderr, "listen: %v\n", err)
		os.Exit(2)
	}
	actualPort := listener.Addr().(*net.TCPAddr).Port
	// Must match the regex in monitor/parser.go: "server is listening on http://[\d.]+:(\d+)"
	fmt.Fprintf(os.Stdout, "main: server is listening on http://127.0.0.1:%d\n", actualPort)

	srv := &http.Server{Handler: mux}
	go srv.Serve(listener) //nolint:errcheck

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	<-sigCh

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(ctx) //nolint:errcheck
	os.Exit(0)
}

// integrationCmdBuilder returns a cmdBuilder that runs the current test binary
// as a fake llama-server subprocess via TestHotReload_HelperProcess.
func integrationCmdBuilder(ctx context.Context, _ string, _ config.ModelParams) *exec.Cmd {
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestHotReload_HelperProcess")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	return cmd
}

// waitModelLoaded polls until the named model is healthy or the deadline passes.
func waitModelLoaded(t *testing.T, m *process.Manager, name string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if m.Loaded(name) {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("model %q did not become healthy within %v", name, timeout)
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

// TestHotReload_ParamChangeTriggersRestart verifies that changing model params
// drains the old process and starts a new one with the updated configuration.
func TestHotReload_ParamChangeTriggersRestart(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Context: 4096, Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := process.New(cfg, slog.Default())
	m.SetCmdBuilder(integrationCmdBuilder)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer m.Stop() //nolint:errcheck

	waitModelLoaded(t, m, "org/model-a", 10*time.Second)

	addrBefore, ok := m.Addr("org/model-a")
	if !ok {
		t.Fatal("Addr: expected ok=true before Reload")
	}

	// Reload with changed context size — must trigger a drain and restart.
	newCfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Context: 8192, Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m.Reload(newCfg)

	// A new process should start and become healthy.
	waitModelLoaded(t, m, "org/model-a", 10*time.Second)

	addrAfter, ok := m.Addr("org/model-a")
	if !ok {
		t.Fatal("Addr: expected ok=true after Reload")
	}
	// The new process listens on a different OS-assigned port.
	if addrBefore == addrAfter {
		t.Error("addr unchanged after param change: expected new process on a different port")
	}
}

// TestHotReload_UnchangedConfigIsNoop verifies that Reload with identical model
// params does not restart the process — the same backend continues serving.
func TestHotReload_UnchangedConfigIsNoop(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Context: 4096, Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := process.New(cfg, slog.Default())
	m.SetCmdBuilder(integrationCmdBuilder)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer m.Stop() //nolint:errcheck

	waitModelLoaded(t, m, "org/model-a", 10*time.Second)

	addrBefore, ok := m.Addr("org/model-a")
	if !ok {
		t.Fatal("Addr: expected ok=true before Reload")
	}

	// Reload with identical params — must be a no-op for this model.
	sameCfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Context: 4096, Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m.Reload(sameCfg)

	if !m.Loaded("org/model-a") {
		t.Fatal("model should still be loaded after no-op Reload")
	}
	addrAfter, ok := m.Addr("org/model-a")
	if !ok {
		t.Fatal("Addr: expected ok=true after no-op Reload")
	}
	if addrBefore != addrAfter {
		t.Errorf("addr changed after no-op Reload: %q → %q (model restarted unexpectedly)", addrBefore, addrAfter)
	}
}

// TestHotReload_InvalidConfigDiscarded verifies that Watch silently discards
// invalid config writes and still delivers the next valid config on the channel.
func TestHotReload_InvalidConfigDiscarded(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	// Write an initial valid config so the watcher directory exists on startup.
	initialYAML := "models:\n  \"org/model\":\n    context: 4096\n"
	if err := os.WriteFile(path, []byte(initialYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("WIRE_CONFIG", path)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch, err := config.Watch(ctx, slog.Default())
	if err != nil {
		t.Fatalf("Watch: %v", err)
	}

	// Write invalid YAML (warmup entry without a matching model pattern).
	// config.Load will reject this with a validation error.
	invalidYAML := "models:\n  \"org/model-A\":\n    context: 4096\nwarmup:\n  - \"org/model-B\"\n"
	if err := os.WriteFile(path, []byte(invalidYAML), 0o600); err != nil {
		t.Fatal(err)
	}

	// Nothing should arrive: invalid config must be silently discarded.
	select {
	case cfg, ok := <-ch:
		if ok {
			t.Errorf("received config after invalid YAML write, expected nothing: %+v", cfg)
		} else {
			t.Fatal("watch channel closed unexpectedly")
		}
	case <-time.After(1500 * time.Millisecond):
		// correct: invalid config was discarded
	}

	// Write a valid config — it must arrive on the channel.
	validYAML := "models:\n  \"org/model\":\n    context: 8192\n"
	if err := os.WriteFile(path, []byte(validYAML), 0o600); err != nil {
		t.Fatal(err)
	}

	select {
	case cfg := <-ch:
		if cfg == nil {
			t.Fatal("received nil config after valid write")
		}
		if cfg.Models["org/model"].Context != 8192 {
			t.Errorf("context: got %d, want 8192", cfg.Models["org/model"].Context)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for valid config to arrive after invalid write")
	}
}

// TestHotReload_GatewayRoutesAfterUpdate verifies that Gateway.UpdateConfig
// causes the gateway to recognise models in the new config: requests for
// previously unknown models return 404 before the update and a non-404 after.
func TestHotReload_GatewayRoutesAfterUpdate(t *testing.T) {
	// Find a free port for the gateway HTTP server.
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	listenAddr := fmt.Sprintf(":%d", port)
	baseURL := fmt.Sprintf("http://localhost:%d", port)

	cfg := &config.Config{
		Listen:    listenAddr,
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 2,
	}

	mgr := process.New(cfg, slog.Default())
	mgr.SetCmdBuilder(integrationCmdBuilder)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := mgr.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer mgr.Stop() //nolint:errcheck

	gw := gateway.New(cfg, mgr, nil, nil, nil, slog.Default())

	gwCtx, gwCancel := context.WithCancel(ctx)
	defer gwCancel()
	go gw.ListenAndServe(gwCtx) //nolint:errcheck

	// Wait for the gateway to accept connections (503 is fine — no warmup models).
	shortClient := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if resp, err := shortClient.Get(baseURL + "/health"); err == nil {
			resp.Body.Close()
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Before UpdateConfig: org/model-b is unknown → 404 model_not_found.
	resp, err := shortClient.Post(
		baseURL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"org/model-b","messages":[]}`),
	)
	if err != nil {
		t.Fatalf("request before update: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("before UpdateConfig: got %d, want 404", resp.StatusCode)
	}

	// UpdateConfig to add org/model-b.
	newCfg := &config.Config{
		Listen: listenAddr,
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		MaxModels: 2,
	}
	gw.UpdateConfig(newCfg)

	// After UpdateConfig: org/model-b is recognised. The model loads on demand
	// via the fake subprocess; use a generous timeout for the subprocess to start.
	longClient := &http.Client{Timeout: 15 * time.Second}
	resp2, err := longClient.Post(
		baseURL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"org/model-b","messages":[]}`),
	)
	if err != nil {
		t.Fatalf("request after update: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode == http.StatusNotFound {
		t.Errorf("after UpdateConfig: got 404 for org/model-b, expected model to be recognised (got non-404 before update)")
	}
}
