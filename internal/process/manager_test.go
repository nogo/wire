package process

import (
	"context"
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

func TestManager_StartStop(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelParams{
			"org/test-model": {Slots: 1},
		},
		Warmup:    []string{"org/test-model"},
		MaxModels: 1,
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Poll until healthy (up to 10s for the subprocess to spin up).
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if m.Loaded("org/test-model") {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if !m.Loaded("org/test-model") {
		t.Fatal("model did not become healthy within 10s")
	}

	addr, ok := m.Addr("org/test-model")
	if !ok {
		t.Fatal("Addr: expected ok=true after healthy")
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
			addr, release, err := m.LoadOrGet(ctx, "org/model-a", params)
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

	// Wait for warmup model A to become healthy.
	waitLoadedModel(t, m, "org/model-a", 10*time.Second)

	// Request model B: should evict A and load B.
	addrB, releaseB, err := m.LoadOrGet(ctx, "org/model-b", cfg.Models["org/model-b"])
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
	addrA2, releaseA2, err := m.LoadOrGet(ctx, "org/model-a", cfg.Models["org/model-a"])
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

	waitLoadedModel(t, m, "org/model-a", 10*time.Second)

	// Hold an in-flight request against model A.
	_, releaseA, err := m.LoadOrGet(ctx, "org/model-a", cfg.Models["org/model-a"])
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	// Do NOT call releaseA yet — this keeps inflight > 0.

	// Requesting model B at max_models=1 must fail because A has in-flight requests.
	_, _, err = m.LoadOrGet(ctx, "org/model-b", cfg.Models["org/model-b"])
	if err == nil {
		t.Fatal("expected error when trying to evict a model with in-flight requests, got nil")
	}

	// Now release A; the next request for B should succeed.
	releaseA()

	_, releaseB, err := m.LoadOrGet(ctx, "org/model-b", cfg.Models["org/model-b"])
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

	_, _, err := m.LoadOrGet(cancelCtx, "org/model-a", params)
	if err == nil {
		t.Log("first request: LoadOrGet returned before context cancelled (load was fast)")
	} else if err != context.Canceled {
		t.Fatalf("first request: expected context.Canceled, got %v", err)
	}

	// The supervisor should still be running (using the manager's detached context).
	// A second request with a valid context should succeed.
	addr, release, err := m.LoadOrGet(bgCtx, "org/model-a", params)
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
			_, _, errs[i] = m.LoadOrGet(bgCtx, "org/bad-model", params)
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
	_, releaseA, err := m.LoadOrGet(bgCtx, "org/model-a", params)
	if err != nil {
		t.Fatalf("LoadOrGet model-a: %v", err)
	}
	_, releaseB, err := m.LoadOrGet(bgCtx, "org/model-b", params)
	if err != nil {
		t.Fatalf("LoadOrGet model-b: %v", err)
	}

	// Third model with both in-flight: must fail.
	_, _, err = m.LoadOrGet(bgCtx, "org/model-c", config.ModelParams{Slots: 1})
	if err == nil {
		t.Fatal("expected capacity error for third model with all models busy, got nil")
	}

	releaseA()
	releaseB()
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
