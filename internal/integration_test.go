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
	"strconv"
	"strings"
	"sync"
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
//
// Supported extra flags (after "--"):
//
//	--health-delay-ms=N   delay GET /health by N milliseconds before responding 200
// ---------------------------------------------------------------------------

// TestHotReload_HelperProcess is NOT a real test — it is a subprocess target.
// It runs as the fake llama-server when invoked via integrationCmdBuilder.
func TestHotReload_HelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	// Parse flags from args after "--".
	args := os.Args
	for i, a := range args {
		if a == "--" {
			args = args[i+1:]
			break
		}
	}

	healthDelayMs := 0
	for _, a := range args {
		if strings.HasPrefix(a, "--health-delay-ms=") {
			ms, _ := strconv.Atoi(strings.TrimPrefix(a, "--health-delay-ms="))
			healthDelayMs = ms
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		if healthDelayMs > 0 {
			time.Sleep(time.Duration(healthDelayMs) * time.Millisecond)
		}
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
func integrationCmdBuilder(ctx context.Context, _ string, _ string, _ config.ModelParams) *exec.Cmd {
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestHotReload_HelperProcess")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	return cmd
}

// integrationCmdBuilderWithDelay is like integrationCmdBuilder but the health
// endpoint sleeps for delayMs milliseconds before responding 200. Used to
// exercise the "concurrent requests coalesce" scenario.
func integrationCmdBuilderWithDelay(delayMs int) func(context.Context, string, string, config.ModelParams) *exec.Cmd {
	return func(ctx context.Context, _ string, _ string, _ config.ModelParams) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0],
			"-test.run=TestHotReload_HelperProcess",
			"--",
			fmt.Sprintf("--health-delay-ms=%d", delayMs),
		)
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
		return cmd
	}
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

// waitGatewayReady polls the gateway /health endpoint until the server accepts
// connections (any HTTP response is sufficient — 503 means degraded but alive).
func waitGatewayReady(t *testing.T, baseURL string, timeout time.Duration) {
	t.Helper()
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if resp, err := client.Get(baseURL + "/health"); err == nil {
			resp.Body.Close()
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("gateway at %s did not become ready within %v", baseURL, timeout)
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

	// Warmup goes through gateway; load model-a explicitly here.
	var addrBefore string
	{
		ticket := m.LoadOrGet("org/model-a", "", cfg.Models["org/model-a"])
		select {
		case addr, ok := <-ticket.Ready():
			if !ok {
				t.Fatalf("LoadOrGet model-a: %v", ticket.Err())
			}
			addrBefore = addr
			ticket.Release()
		case <-time.After(10 * time.Second):
			t.Fatal("timeout loading model-a")
		}
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

	// Warmup goes through gateway; load model-a explicitly here.
	var addrBefore string
	{
		ticket := m.LoadOrGet("org/model-a", "", cfg.Models["org/model-a"])
		select {
		case addr, ok := <-ticket.Ready():
			if !ok {
				t.Fatalf("LoadOrGet model-a: %v", ticket.Err())
			}
			addrBefore = addr
			ticket.Release()
		case <-time.After(10 * time.Second):
			t.Fatal("timeout loading model-a")
		}
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

	// Write an invalid config: a warmup entry that matches multiple model patterns
	// triggers a validation error in config.Load.
	invalidYAML := "models:\n  \"org/model-A\":\n    context: 4096\n  \"org/model-*\":\n    context: 4096\nwarmup:\n  - \"org/model-A\"\n"
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

	gw := gateway.New(cfg, mgr, nil, nil, nil, nil, slog.Default())

	gwCtx, gwCancel := context.WithCancel(ctx)
	defer gwCancel()
	go gw.ListenAndServe(gwCtx) //nolint:errcheck

	// Wait for the gateway to accept connections (503 is fine — no warmup models).
	waitGatewayReady(t, baseURL, 5*time.Second)

	// Before UpdateConfig: org/model-b is unknown → 404 model_not_found.
	shortClient := &http.Client{Timeout: 2 * time.Second}
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

// ---------------------------------------------------------------------------
// New integration tests: coalesced loading, queue priority routing, drain+requeue
// ---------------------------------------------------------------------------

// TestIntegration_CoalescedLoading verifies that concurrent gateway HTTP requests
// for an unloaded model coalesce to a single process start. The 300ms health
// delay ensures all goroutines issue their requests before the model becomes
// healthy, exercising the queue's coalescing path end-to-end.
// OOMScoreAdj is verified on the loaded process.
func TestIntegration_CoalescedLoading(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 1,
	}
	m := process.New(cfg, slog.Default())
	// 300ms health delay ensures all goroutines arrive before the model becomes healthy.
	m.SetCmdBuilder(integrationCmdBuilderWithDelay(300))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer m.Stop() //nolint:errcheck

	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	gwCfg := &config.Config{
		Listen:    fmt.Sprintf(":%d", port),
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 1,
	}
	gw := gateway.New(gwCfg, m, nil, nil, nil, nil, slog.Default())

	gwCtx, gwCancel := context.WithCancel(ctx)
	defer gwCancel()
	go gw.ListenAndServe(gwCtx) //nolint:errcheck

	baseURL := fmt.Sprintf("http://localhost:%d", port)
	waitGatewayReady(t, baseURL, 5*time.Second)

	const n = 10
	var wg sync.WaitGroup
	wg.Add(n)
	codes := make([]int, n)

	for i := range n {
		go func(i int) {
			defer wg.Done()
			client := &http.Client{Timeout: 15 * time.Second}
			resp, err := client.Post(
				baseURL+"/v1/chat/completions", "application/json",
				strings.NewReader(`{"model":"org/model-a","messages":[]}`),
			)
			if err != nil {
				t.Errorf("goroutine %d: %v", i, err)
				return
			}
			resp.Body.Close()
			codes[i] = resp.StatusCode
		}(i)
	}
	wg.Wait()

	for i, code := range codes {
		if code != http.StatusOK {
			t.Errorf("goroutine %d: status %d, want 200", i, code)
		}
	}

	// Only one model process should have been started.
	if got := len(m.RunningModels()); got != 1 {
		t.Errorf("RunningModels: got %d, want 1 (coalescing failed)", got)
	}

	// Verify OOMScoreAdj on the running llama-server process.
	for _, bs := range m.Snapshot() {
		if bs.Name != "org/model-a" || bs.PID <= 0 {
			continue
		}
		path := fmt.Sprintf("/proc/%d/oom_score_adj", bs.PID)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("read oom_score_adj for pid %d: %v", bs.PID, err)
			break
		}
		if got := strings.TrimSpace(string(raw)); got != "1000" {
			t.Errorf("oom_score_adj = %q, want 1000", got)
		}
		break
	}
}

// TestIntegration_QueuePriorityRouting verifies that requests through different
// endpoint paths (chat=high, completions=medium, embeddings=low) all succeed
// end-to-end through the gateway's priority-aware queue.
//
// Budget enforcement (low-prio timing out before high-prio when model is slow)
// is covered exhaustively by queue unit tests (TestPriorityBudgets,
// TestPriorityOrdering). This test validates the full routing stack.
func TestIntegration_QueuePriorityRouting(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 4}},
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

	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	gwCfg := &config.Config{
		Listen:    fmt.Sprintf(":%d", port),
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 4}},
		MaxModels: 1,
	}
	gw := gateway.New(gwCfg, m, nil, nil, nil, nil, slog.Default())

	gwCtx, gwCancel := context.WithCancel(ctx)
	defer gwCancel()
	go gw.ListenAndServe(gwCtx) //nolint:errcheck

	baseURL := fmt.Sprintf("http://localhost:%d", port)
	waitGatewayReady(t, baseURL, 5*time.Second)

	client := &http.Client{Timeout: 15 * time.Second}

	paths := []struct {
		path string
		body string
	}{
		{"/v1/chat/completions", `{"model":"org/model-a","messages":[]}`},
		{"/v1/completions", `{"model":"org/model-a","prompt":"hello"}`},
		{"/v1/embeddings", `{"model":"org/model-a","input":"hello"}`},
	}

	for _, tc := range paths {
		resp, err := client.Post(baseURL+tc.path, "application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Errorf("POST %s: %v", tc.path, err)
			continue
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("POST %s: status %d, want 200", tc.path, resp.StatusCode)
		}
	}
}

// TestIntegration_DrainAndRequeue verifies the full drain+re-queue cycle:
// a model is loaded, drained via Reload, then re-added to the config and
// successfully loaded again via the ticket API.
func TestIntegration_DrainAndRequeue(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
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

	// Initial load via ticket API.
	ticket := m.LoadOrGet("org/model-a", "", cfg.Models["org/model-a"])
	select {
	case addr, ok := <-ticket.Ready():
		if !ok {
			t.Fatalf("initial load failed: %v", ticket.Err())
		}
		if addr == "" {
			t.Fatal("initial load: got empty addr")
		}
		ticket.Release()
	case <-time.After(10 * time.Second):
		t.Fatal("timeout loading model-a")
	}

	if !m.Loaded("org/model-a") {
		t.Fatal("model-a should be loaded before drain")
	}

	// Drain by removing model-a from config.
	emptyCfg := &config.Config{
		Models:    map[string]config.ModelParams{},
		MaxModels: 1,
	}
	m.Reload(emptyCfg)

	if m.Loaded("org/model-a") {
		t.Error("model-a should be drained after Reload removed it")
	}

	// Verify model-a is absent from RunningModels.
	for _, name := range m.RunningModels() {
		if name == "org/model-a" {
			t.Error("model-a still appears in RunningModels after drain")
		}
	}

	// Re-add model-a to config; new request should trigger a fresh load.
	m.Reload(cfg)

	ticket2 := m.LoadOrGet("org/model-a", "", cfg.Models["org/model-a"])
	select {
	case addr, ok := <-ticket2.Ready():
		if !ok {
			t.Fatalf("re-queue failed: %v", ticket2.Err())
		}
		if addr == "" {
			t.Error("re-queue: got empty addr")
		}
		ticket2.Release()
	case <-time.After(10 * time.Second):
		t.Fatal("timeout re-loading model-a after drain")
	}

	if !m.Loaded("org/model-a") {
		t.Error("model-a should be loaded after re-queue")
	}
}
