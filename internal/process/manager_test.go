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
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/vram"
)

// ---------------------------------------------------------------------------
// Subprocess helper: fake llama-server
//
// Tests that need a real process invoke the test binary itself with
// GO_WANT_HELPER_PROCESS=1 and -test.run=TestHelperProcess.
// The subprocess then acts as a minimal llama-server on the requested port.
//
// Supported extra flags (after "--"):
//   --health-delay-ms=N   delay GET /health by N milliseconds before responding 200
//   --fail                exit immediately without starting the HTTP server
// ---------------------------------------------------------------------------

// TestHelperProcess is NOT a real test — it is a subprocess target.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	// Locate args after "--"
	args := os.Args
	for i, a := range args {
		if a == "--" {
			args = args[i+1:]
			break
		}
	}

	if len(args) == 0 || args[0] != "llama-server" {
		fmt.Fprintln(os.Stderr, "expected llama-server as first arg")
		os.Exit(2)
	}

	// Parse flags from remaining args.
	port := "8080"
	healthDelayMs := 0
	fail := false
	for i, a := range args[1:] {
		if a == "--port" && i+1 < len(args[1:]) {
			port = args[1:][i+1]
		}
		if strings.HasPrefix(a, "--health-delay-ms=") {
			ms, _ := strconv.Atoi(strings.TrimPrefix(a, "--health-delay-ms="))
			healthDelayMs = ms
		}
		if a == "--fail" {
			fail = true
		}
	}

	if fail {
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if healthDelayMs > 0 {
			time.Sleep(time.Duration(healthDelayMs) * time.Millisecond)
		}
		w.WriteHeader(http.StatusOK)
	})

	// Use port 0 to let the OS pick a free port, then print the listening
	// line that wire's event parser expects.
	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		fmt.Fprintf(os.Stderr, "listen: %v\n", err)
		os.Exit(2)
	}
	actualPort := listener.Addr().(*net.TCPAddr).Port
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

// testCmdBuilder returns a cmdBuilder that runs the test binary as a fake
// llama-server subprocess.
func testCmdBuilder(ctx context.Context, modelName string, params config.ModelParams) *exec.Cmd {
	args := append(
		[]string{"-test.run=TestHelperProcess", "--", "llama-server"},
		buildArgs(modelName, params)...,
	)
	cmd := exec.CommandContext(ctx, os.Args[0], args...)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	return cmd
}

// testCmdBuilderWithDelay is like testCmdBuilder but the health endpoint
// sleeps for delayMs milliseconds before responding 200. Used to exercise
// the "concurrent requests coalesce" and "context cancellation" scenarios.
func testCmdBuilderWithDelay(delayMs int) func(context.Context, string, config.ModelParams) *exec.Cmd {
	return func(ctx context.Context, modelName string, params config.ModelParams) *exec.Cmd {
		args := append(
			[]string{"-test.run=TestHelperProcess", "--", "llama-server"},
			buildArgs(modelName, params)...,
		)
		args = append(args, fmt.Sprintf("--health-delay-ms=%d", delayMs))
		cmd := exec.CommandContext(ctx, os.Args[0], args...)
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
		return cmd
	}
}

// failCmdBuilder returns a cmdBuilder whose process always exits immediately
// without starting an HTTP server — simulating a model that fails to load.
func failCmdBuilder(ctx context.Context, modelName string, params config.ModelParams) *exec.Cmd {
	args := append(
		[]string{"-test.run=TestHelperProcess", "--", "llama-server"},
		buildArgs(modelName, params)...,
	)
	args = append(args, "--fail")
	cmd := exec.CommandContext(ctx, os.Args[0], args...)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	return cmd
}

// newTestManager creates a Manager wired to the fake llama-server subprocess.
func newTestManager(cfg *config.Config) *Manager {
	m := New(cfg, slog.Default())
	m.cmdBuilder = testCmdBuilder
	return m
}

// loadOrGetWait is a test helper that calls LoadOrGet and blocks until the
// ticket resolves or ctx is cancelled, mimicking the old synchronous signature.
func loadOrGetWait(ctx context.Context, m *Manager, modelName string, params config.ModelParams) (addr string, release func(), err error) {
	ticket := m.LoadOrGet(modelName, params)
	select {
	case a, ok := <-ticket.Ready():
		if !ok {
			return "", func() {}, ticket.Err()
		}
		return a, ticket.Release, nil
	case <-ctx.Done():
		return "", func() {}, ctx.Err()
	}
}

// mockVRAMBackend is a static VRAM backend for testing VRAM-aware eviction.
type mockVRAMBackend struct {
	total uint64
	used  uint64
}

func (b *mockVRAMBackend) Discover() []monitor.GPUInfo       { return nil }
func (b *mockVRAMBackend) ReadMetrics() []monitor.GPUMetrics { return nil }
func (b *mockVRAMBackend) TotalVRAM() uint64                 { return b.total }
func (b *mockVRAMBackend) UsedVRAM() uint64                  { return b.used }
func (b *mockVRAMBackend) IsNoop() bool                      { return false }

// ---------------------------------------------------------------------------
// Unit tests — pure functions
// ---------------------------------------------------------------------------

func TestBuildArgs_minimal(t *testing.T) {
	params := config.ModelParams{}
	got := buildArgs("org/model", params)
	want := []string{"-hf", "org/model", "--port", "0", "--no-webui", "--metrics", "--parallel", "1"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs minimal: got %v, want %v", got, want)
	}
}

func TestBuildArgs_full(t *testing.T) {
	params := config.ModelParams{
		GPULayers: 32,
		Context:   4096,
		Slots:     4,
		Pooling:   "mean",
		Extra:     []string{"--flash-attn"},
	}
	got := buildArgs("org/model", params)
	want := []string{
		"-hf", "org/model",
		"--port", "0",
		"--no-webui",
		"--metrics",
		"--parallel", "4",
		"-ngl", "32",
		"-c", "4096",
		"--embedding", "--pooling", "mean",
		"--flash-attn",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs full: got %v, want %v", got, want)
	}
}

func TestBuildArgs_speculative(t *testing.T) {
	params := config.ModelParams{
		SpecType:   "ngram-mod",
		DraftMax:   16,
		DraftMin:   2,
		SpecNgramN: 4,
		SpecNgramM: 3,
	}
	got := buildArgs("org/model", params)
	want := []string{
		"-hf", "org/model",
		"--port", "0",
		"--no-webui",
		"--metrics",
		"--parallel", "1",
		"--spec-type", "ngram-mod",
		"--draft-max", "16",
		"--draft-min", "2",
		"--spec-ngram-size-n", "4",
		"--spec-ngram-size-m", "3",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs speculative: got %v, want %v", got, want)
	}
}

func TestBuildArgs_speculative_empty(t *testing.T) {
	params := config.ModelParams{}
	args := buildArgs("org/model", params)
	specFlags := []string{"--spec-type", "--draft-max", "--draft-min", "--spec-ngram-size-n", "--spec-ngram-size-m"}
	for _, f := range specFlags {
		for _, a := range args {
			if a == f {
				t.Errorf("unexpected spec flag %q in args: %v", f, args)
			}
		}
	}
}

func TestBuildArgs_speculative_partial(t *testing.T) {
	params := config.ModelParams{SpecType: "ngram-mod"}
	got := buildArgs("org/model", params)
	want := []string{
		"-hf", "org/model",
		"--port", "0",
		"--no-webui",
		"--metrics",
		"--parallel", "1",
		"--spec-type", "ngram-mod",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs speculative partial: got %v, want %v", got, want)
	}
}

func TestBuildArgs_zeroGPULayersOmitted(t *testing.T) {
	params := config.ModelParams{GPULayers: 0, Context: 0}
	args := buildArgs("org/model", params)
	for _, a := range args {
		if a == "-ngl" || a == "-c" {
			t.Errorf("unexpected flag %q in args: %v", a, args)
		}
	}
}

// ---------------------------------------------------------------------------
// Manager tests
// ---------------------------------------------------------------------------

func TestManager_AddrMissing(t *testing.T) {
	m := New(&config.Config{}, slog.Default())
	_, ok := m.Addr("nonexistent")
	if ok {
		t.Error("Addr: expected ok=false for missing model")
	}
}

func TestManager_LoadedMissing(t *testing.T) {
	m := New(&config.Config{}, slog.Default())
	if m.Loaded("nonexistent") {
		t.Error("Loaded: expected false for unknown model")
	}
}

func TestLaunch_OOMScoreAdj(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	m := New(&config.Config{}, slog.Default())
	m.SetCmdBuilder(testCmdBuilder)

	cmd, _, err := m.launch(ctx, "org/test-model", config.ModelParams{})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	defer func() {
		cancel()
		cmd.Wait() //nolint:errcheck
	}()

	pid := cmd.Process.Pid
	path := fmt.Sprintf("/proc/%d/oom_score_adj", pid)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	got := strings.TrimSpace(string(raw))
	if got != "1000" {
		t.Errorf("oom_score_adj = %q, want \"1000\"", got)
	}
}

func TestManager_StartStop(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/test-model": {Slots: 1},
		},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Warmup goes through the gateway; load the model explicitly here.
	addr, release, err := loadOrGetWait(ctx, m, "org/test-model", cfg.Models["org/test-model"])
	if err != nil {
		t.Fatalf("LoadOrGet: %v", err)
	}
	release()

	if !m.Loaded("org/test-model") {
		t.Fatal("model did not become healthy within 10s")
	}
	if addr == "" {
		t.Error("Addr: returned empty string")
	}

	running := m.RunningModels()
	if len(running) != 1 || running[0] != "org/test-model" {
		t.Errorf("RunningModels: got %v, want [org/test-model]", running)
	}

	if err := m.Stop(); err != nil {
		t.Errorf("Stop: %v", err)
	}

	if m.Loaded("org/test-model") {
		t.Error("Loaded: expected false after Stop")
	}
}

// ---------------------------------------------------------------------------
// LoadOrGet tests
// ---------------------------------------------------------------------------

// TestLoadOrGet_Coalescing verifies that 10 concurrent requests for the same
// unloaded model coalesce to a single process start.
func TestLoadOrGet_Coalescing(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 1,
	}
	m := New(cfg, slog.Default())
	// 200 ms startup delay to ensure all goroutines arrive before healthy.
	m.cmdBuilder = testCmdBuilderWithDelay(200)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	params := config.ModelParams{Slots: 1}
	const n = 10
	addrs := make([]string, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			addr, release, err := loadOrGetWait(ctx, m, "org/model-a", params)
			if err != nil {
				t.Errorf("goroutine %d: LoadOrGet error: %v", i, err)
				return
			}
			defer release()
			addrs[i] = addr
		}(i)
	}
	wg.Wait()

	// All goroutines must have gotten a non-empty addr.
	first := addrs[0]
	if first == "" {
		t.Fatal("got empty addr")
	}
	for i, a := range addrs {
		if a != first {
			t.Errorf("goroutine %d: got addr %q, want %q (coalescing failed)", i, a, first)
		}
	}

	// Only one process should have been started.
	if len(m.RunningModels()) != 1 {
		t.Errorf("RunningModels: expected 1, got %d", len(m.RunningModels()))
	}
}

// TestLoadOrGet_LRUEviction verifies that when max_models=1 and model A is
// loaded, requesting model B evicts A and loads B. Requesting A again evicts B
// and reloads A.
func TestLoadOrGet_LRUEviction(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Load model A explicitly; warmup goes through the gateway admission path.
	_, releaseA0, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a (initial): %v", err)
	}
	releaseA0()

	// Request model B: should evict A and load B.
	addrB, releaseB, err := loadOrGetWait(ctx, m, "org/model-b", cfg.Models["org/model-b"])
	if err != nil {
		t.Fatalf("LoadOrGet model-b: %v", err)
	}
	if addrB == "" {
		t.Fatal("got empty addr for model-b")
	}
	releaseB()

	if m.Loaded("org/model-a") {
		t.Error("model-a should have been evicted")
	}
	if !m.Loaded("org/model-b") {
		t.Error("model-b should be healthy")
	}

	// Request model A again: should evict B and reload A.
	addrA2, releaseA2, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a (reload): %v", err)
	}
	if addrA2 == "" {
		t.Fatal("got empty addr for model-a reload")
	}
	releaseA2()

	if m.Loaded("org/model-b") {
		t.Error("model-b should have been evicted on second request")
	}
	if !m.Loaded("org/model-a") {
		t.Error("model-a should be healthy after reload")
	}
}

// TestLoadOrGet_InFlightProtection verifies that a model with an active
// in-flight request cannot be evicted; the caller gets a 503-style error.
func TestLoadOrGet_InFlightProtection(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Hold an in-flight request against model A (also loads it).
	_, releaseA, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	// Do NOT call releaseA yet — this keeps inflight > 0.

	// Requesting model B at max_models=1 must fail because A has in-flight requests.
	_, _, err = loadOrGetWait(ctx, m, "org/model-b", cfg.Models["org/model-b"])
	if err == nil {
		t.Fatal("expected error when trying to evict a model with in-flight requests, got nil")
	}

	// Now release A; the next request for B should succeed.
	releaseA()

	_, releaseB, err := loadOrGetWait(ctx, m, "org/model-b", cfg.Models["org/model-b"])
	if err != nil {
		t.Fatalf("LoadOrGet model-b after release: %v", err)
	}
	releaseB()
}

// TestLoadOrGet_ContextCancellation verifies that cancelling the requesting
// context does not abort the model load. A second request for the same model
// (with a valid context) should join the in-progress load.
func TestLoadOrGet_ContextCancellation(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 1,
	}
	m := New(cfg, slog.Default())
	// 400 ms delay so we can cancel the first request mid-load.
	m.cmdBuilder = testCmdBuilderWithDelay(400)

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	params := config.ModelParams{Slots: 1}

	// First request: cancel the context almost immediately.
	cancelCtx, cancelFn := context.WithCancel(bgCtx)
	cancelFn() // cancel before LoadOrGet returns

	_, _, err := loadOrGetWait(cancelCtx, m, "org/model-a", params)
	if err == nil {
		t.Log("first request: LoadOrGet returned before context cancelled (load was fast)")
	} else if !errors.Is(err, context.Canceled) {
		t.Fatalf("first request: expected context.Canceled, got %v", err)
	}

	// The supervisor should still be running (using the manager's detached context).
	// A second request with a valid context should succeed.
	addr, release, err := loadOrGetWait(bgCtx, m, "org/model-a", params)
	if err != nil {
		t.Fatalf("second LoadOrGet: %v", err)
	}
	defer release()
	if addr == "" {
		t.Fatal("second LoadOrGet: got empty addr")
	}
}

// TestLoadOrGet_LoadFailure verifies that when a model fails to start, all
// concurrent waiters receive an error and the model is removed from procs.
func TestLoadOrGet_LoadFailure(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/bad-model": {Slots: 1}},
		MaxModels: 2,
	}
	m := New(cfg, slog.Default())
	m.cmdBuilder = failCmdBuilder            // process exits immediately
	m.healthTimeout = 500 * time.Millisecond // short timeout so test is fast

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	params := config.ModelParams{Slots: 1}
	const n = 5
	var wg sync.WaitGroup
	errs := make([]error, n)
	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			_, _, errs[i] = loadOrGetWait(bgCtx, m, "org/bad-model", params)
		}(i)
	}
	wg.Wait()

	// All waiters must have received an error.
	for i, err := range errs {
		if err == nil {
			t.Errorf("goroutine %d: expected load error, got nil", i)
		}
	}

	// Model must not appear in ListModels (state idle = excluded).
	for _, mi := range m.ListModels() {
		if mi.ID == "org/bad-model" {
			t.Errorf("failed model still visible in ListModels with status %q", mi.Status)
		}
	}
}

// TestLoadOrGet_CapacityLimit verifies that when max_models=2 and both loaded
// models have in-flight requests, a third request returns an error.
func TestLoadOrGet_CapacityLimit(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		MaxModels: 2,
	}
	m := newTestManager(cfg)

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	params := config.ModelParams{Slots: 1}

	// Load both models and hold in-flight requests.
	_, releaseA, err := loadOrGetWait(bgCtx, m, "org/model-a", params)
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	_, releaseB, err := loadOrGetWait(bgCtx, m, "org/model-b", params)
	if err != nil {
		t.Fatalf("LoadOrGet model-b: %v", err)
	}

	// Third model with both in-flight: must fail.
	_, _, err = loadOrGetWait(bgCtx, m, "org/model-c", config.ModelParams{Slots: 1})
	if err == nil {
		t.Fatal("expected capacity error for third model with all models busy, got nil")
	}

	releaseA()
	releaseB()
}

// ---------------------------------------------------------------------------
// Reload tests
// ---------------------------------------------------------------------------

// TestReload_ParamsChanged verifies that a model with changed params is drained
// and restarted by Reload.
func TestReload_ParamsChanged(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	_, rel, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	rel()

	// New config: same model, different params.
	newCfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 2}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m.Reload(newCfg)

	// Model should restart and become healthy again.
	waitLoadedModel(t, m, "org/model-a", 10*time.Second)
}

// TestReload_ParamsUnchanged verifies that a model with unchanged params is NOT
// restarted by Reload — same port (same process) before and after.
func TestReload_ParamsUnchanged(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	addrBefore, rel, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	rel()

	// Same params — should be a no-op for this model.
	newCfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m.Reload(newCfg)

	if !m.Loaded("org/model-a") {
		t.Fatal("model should still be loaded after no-op Reload")
	}
	addrAfter, ok := m.Addr("org/model-a")
	if !ok {
		t.Fatal("Addr: expected ok=true after Reload")
	}
	if addrBefore != addrAfter {
		t.Errorf("addr changed after no-op Reload: %q → %q (model was restarted unexpectedly)", addrBefore, addrAfter)
	}
}

// TestReload_RemovedModel verifies that a running model not present in the new
// config is drained and removed by Reload.
func TestReload_RemovedModel(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	_, rel, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	rel()

	// New config: model-a removed entirely.
	newCfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-b": {Slots: 1}},
		Warmup:    []string{},
		MaxModels: 1,
	}
	m.Reload(newCfg)

	if m.Loaded("org/model-a") {
		t.Error("model-a should have been drained after Reload removed it")
	}

	m.mu.RLock()
	_, stillInProcs := m.procs["org/model-a"]
	m.mu.RUnlock()
	if stillInProcs {
		t.Error("model-a should have been deleted from procs")
	}
}

// TestReload_NewWarmupModel verifies that a new warmup model in the updated
// config is started by Reload.
func TestReload_NewWarmupModel(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		Warmup:    []string{"org/model-a"},
		MaxModels: 2,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	_, rel, err := loadOrGetWait(ctx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	rel()

	// New config: add model-b to warmup.
	newCfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		Warmup:    []string{"org/model-a", "org/model-b"},
		MaxModels: 2,
	}
	m.Reload(newCfg)

	// model-b should be started and eventually healthy.
	waitLoadedModel(t, m, "org/model-b", 10*time.Second)
}

// TestLoadOrGet_TicketAsync verifies that LoadOrGet returns a ticket immediately
// without blocking even when the model has not yet loaded.
func TestLoadOrGet_TicketAsync(t *testing.T) {
	cfg := &config.Config{
		Models:    map[string]config.ModelParams{"org/model-a": {Slots: 1}},
		MaxModels: 1,
	}
	m := New(cfg, slog.Default())
	m.cmdBuilder = testCmdBuilderWithDelay(200) // 200ms health delay

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	start := time.Now()
	ticket := m.LoadOrGet("org/model-a", config.ModelParams{Slots: 1})
	elapsed := time.Since(start)
	if elapsed > 50*time.Millisecond {
		t.Errorf("LoadOrGet took %v; expected near-immediate return", elapsed)
	}

	select {
	case addr, ok := <-ticket.Ready():
		if !ok {
			t.Fatalf("ticket failed: %v", ticket.Err())
		}
		if addr == "" {
			t.Error("got empty addr")
		}
		ticket.Release()
	case <-time.After(10 * time.Second):
		t.Fatal("ticket did not resolve within 10s")
	}
}

// TestLoadOrGet_Drain verifies drain behavior: once a model is in stateUnloading,
// new requests fail immediately; drainAndCancel waits for the inflight counter
// to reach zero (or the timeout) before sending SIGTERM.
func TestLoadOrGet_Drain(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
		},
		Warmup:    []string{"org/model-a"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)
	m.drainTimeout = 500 * time.Millisecond

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Load then release model-a so it is idle (inflight=0, evictable).
	_, releaseA, err := loadOrGetWait(bgCtx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	releaseA()

	// Request model-b: evicts idle model-a and loads model-b.
	_, releaseB, err := loadOrGetWait(bgCtx, m, "org/model-b", cfg.Models["org/model-b"])
	if err != nil {
		t.Fatalf("LoadOrGet model-b (after eviction): %v", err)
	}
	releaseB()

	if m.Loaded("org/model-a") {
		t.Error("model-a should have been evicted")
	}
	if !m.Loaded("org/model-b") {
		t.Error("model-b should be loaded after eviction")
	}

	// Inject a synthetic proc in stateUnloading to verify that new requests
	// to a draining model are rejected without waiting.
	m.mu.Lock()
	synth := &procState{
		modelName: "org/draining",
		state:     stateUnloading,
		ready:     make(chan struct{}),
		cancel:    func() {},
	}
	m.procs["org/draining"] = synth
	m.mu.Unlock()

	tick := m.LoadOrGet("org/draining", config.ModelParams{})
	select {
	case _, ok := <-tick.Ready():
		if ok {
			t.Error("expected failure for unloading model, got address")
			tick.Release()
		}
	case <-time.After(200 * time.Millisecond):
		t.Error("ticket for draining model should fail immediately")
	}
	if tick.Err() == nil {
		t.Error("expected non-nil error for unloading model")
	}

	// Verify drainAndCancel respects the timeout: create a proc with inflight=1
	// and confirm cancel() is called after drainTimeout.
	ps2 := &procState{modelName: "org/timeout-test", cancel: func() {}}
	ps2.inflight.Add(1)
	start := time.Now()
	m.drainTimeout = 100 * time.Millisecond
	m.drainAndCancel(ps2)
	elapsed := time.Since(start)
	if elapsed < 90*time.Millisecond {
		t.Errorf("drainAndCancel returned in %v; expected to wait ~100ms for timeout", elapsed)
	}
}

// TestLoadOrGet_CascadingVRAMEviction verifies that when VRAM is insufficient
// for a new model, the minimum set of LRU idle models is evicted to make room.
// Models A and B are loaded without a VRAM tracker. Then a tracker is attached
// with a mock backend that reports only 2 GiB free, less than model-c's 3 GiB
// estimate. The cascading eviction must select model-a (LRU) and leave model-b.
func TestLoadOrGet_CascadingVRAMEviction(t *testing.T) {
	const (
		totalVRAM = 8 << 30 // 8 GiB total
		modelEst  = 3 << 30 // 3 GiB per model
		// mock backend reports 6 GiB used → 2 GiB free (not enough for model-c)
		mockUsed = 6 << 30
	)

	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/model-a": {Slots: 1},
			"org/model-b": {Slots: 1},
			"org/model-c": {Slots: 1},
		},
		MaxModels: 10, // no count-based limit; only VRAM drives eviction
	}

	m := newTestManager(cfg)

	bgCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.Start(bgCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Load model-a first (becomes LRU candidate) — no tracker yet.
	_, releaseA, err := loadOrGetWait(bgCtx, m, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	releaseA()

	// Load model-b (more recently used) — still no tracker.
	_, releaseB, err := loadOrGetWait(bgCtx, m, "org/model-b", cfg.Models["org/model-b"])
	if err != nil {
		t.Fatalf("LoadOrGet model-b: %v", err)
	}
	releaseB()

	// Now attach a VRAM tracker. The mock backend reports 6 GiB used, leaving
	// only 2 GiB free — insufficient for model-c (3 GiB estimate).
	backend := &mockVRAMBackend{total: totalVRAM, used: mockUsed}
	tracker := vram.New(backend, nil)
	tracker.Discover()
	tracker.SetEstimate("org/model-a", modelEst)
	tracker.SetEstimate("org/model-b", modelEst)
	tracker.SetEstimate("org/model-c", modelEst)
	m.SetVRAMTracker(tracker)

	// Request model-c: 3 GiB needed, only 2 GiB free → must evict model-a (LRU).
	_, releaseC, err := loadOrGetWait(bgCtx, m, "org/model-c", cfg.Models["org/model-c"])
	if err != nil {
		t.Fatalf("LoadOrGet model-c: %v", err)
	}
	releaseC()

	if m.Loaded("org/model-a") {
		t.Error("model-a should have been evicted (LRU)")
	}
	if !m.Loaded("org/model-b") {
		t.Error("model-b should still be loaded (more recently used)")
	}
	if !m.Loaded("org/model-c") {
		t.Error("model-c should be loaded")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// waitLoadedModel polls until the named model is healthy or the deadline passes.
func waitLoadedModel(t *testing.T, m *Manager, name string, timeout time.Duration) {
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
