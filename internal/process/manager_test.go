package process

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/signal"
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

	// Parse --port from remaining flags.
	port := "8080"
	for i, a := range args[1:] {
		if a == "--port" && i+1 < len(args[1:]) {
			port = args[1:][i+1]
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	srv := &http.Server{Addr: ":" + port, Handler: mux}
	go srv.ListenAndServe() //nolint:errcheck

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
func testCmdBuilder(ctx context.Context, b config.Backend, port int) *exec.Cmd {
	args := append(
		[]string{"-test.run=TestHelperProcess", "--", "llama-server"},
		buildArgs(b, port)...,
	)
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
	b := config.Backend{Model: "org/model"}
	got := buildArgs(b, 19100)
	want := []string{"-hf", "org/model", "--port", "19100", "--no-webui"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs minimal: got %v, want %v", got, want)
	}
}

func TestBuildArgs_full(t *testing.T) {
	b := config.Backend{
		Model:     "org/model",
		GPULayers: 32,
		Context:   4096,
		Pooling:   "mean",
		Extra:     []string{"--flash-attn"},
	}
	got := buildArgs(b, 19101)
	want := []string{
		"-hf", "org/model",
		"--port", "19101",
		"--no-webui",
		"-ngl", "32",
		"-c", "4096",
		"--embedding", "--pooling", "mean",
		"--flash-attn",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("buildArgs full: got %v, want %v", got, want)
	}
}

func TestBuildArgs_zeroGPULayersOmitted(t *testing.T) {
	b := config.Backend{Model: "org/model", GPULayers: 0, Context: 0}
	args := buildArgs(b, 19102)
	for _, a := range args {
		if a == "-ngl" || a == "-c" {
			t.Errorf("unexpected flag %q in args: %v", a, args)
		}
	}
}

func TestFreePort(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Errorf("freePort returned out-of-range port: %d", port)
	}
}

// ---------------------------------------------------------------------------
// External backend tests
// ---------------------------------------------------------------------------

func TestManager_AddrExternal(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: "http://api.example.com", Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	addr, ok := m.Addr("ext")
	if !ok {
		t.Fatal("Addr: expected ok=true for external backend")
	}
	if addr != "http://api.example.com" {
		t.Errorf("Addr: got %q, want %q", addr, "http://api.example.com")
	}
}

func TestManager_AddrMissing(t *testing.T) {
	m := New(&config.Config{Backends: map[string]config.Backend{}}, slog.Default())
	_, ok := m.Addr("nonexistent")
	if ok {
		t.Error("Addr: expected ok=false for missing backend")
	}
}

func TestManager_HealthyExternal_200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer srv.Close()

	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: srv.URL, Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	if !m.Healthy("ext") {
		t.Error("Healthy: expected true for external backend returning 200")
	}
}

func TestManager_HealthyExternal_503(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: srv.URL, Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	if m.Healthy("ext") {
		t.Error("Healthy: expected false for external backend returning 503")
	}
}

func TestManager_HealthyExternal_noHealthEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: srv.URL, Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	// A 404 is not 200, so this should return false.
	if m.Healthy("ext") {
		t.Error("Healthy: expected false for external backend returning 404")
	}
}

func TestManager_HealthyExternal_unreachable(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: "http://127.0.0.1:1", Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	// Unreachable host → best-effort true.
	if !m.Healthy("ext") {
		t.Error("Healthy: expected true (best-effort) for unreachable external backend")
	}
}

// ---------------------------------------------------------------------------
// Managed backend lifecycle tests
// ---------------------------------------------------------------------------

func TestManager_StartStop(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"local": {Model: "org/test-model"},
		},
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
		if m.Healthy("local") {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if !m.Healthy("local") {
		t.Fatal("backend did not become healthy within 10s")
	}

	addr, ok := m.Addr("local")
	if !ok {
		t.Fatal("Addr: expected ok=true after healthy")
	}
	if addr == "" {
		t.Error("Addr: returned empty string")
	}

	if err := m.Stop(); err != nil {
		t.Errorf("Stop: %v", err)
	}

	if m.Healthy("local") {
		t.Error("Healthy: expected false after Stop")
	}
}

func TestManager_SwapNotManaged(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"ext": {URL: "http://api.example.com", Provider: "openai"},
		},
	}
	m := New(cfg, slog.Default())

	ctx := context.Background()
	if err := m.Swap(ctx, "ext", "new/model"); err == nil {
		t.Error("Swap: expected error for non-managed backend, got nil")
	}
}

func TestManager_Swap(t *testing.T) {
	cfg := &config.Config{
		Backends: map[string]config.Backend{
			"local": {Model: "org/model-v1"},
		},
	}
	m := newTestManager(cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := m.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Wait for initial healthy.
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if m.Healthy("local") {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !m.Healthy("local") {
		t.Fatal("backend did not become healthy before swap")
	}

	addrBefore, _ := m.Addr("local")

	if err := m.Swap(ctx, "local", "org/model-v2"); err != nil {
		t.Fatalf("Swap: %v", err)
	}

	if !m.Healthy("local") {
		t.Error("backend not healthy after swap")
	}

	addrAfter, _ := m.Addr("local")
	// The port may differ; both should be non-empty.
	if addrAfter == "" {
		t.Error("Addr returned empty after swap")
	}
	t.Logf("addr before swap: %s, after swap: %s", addrBefore, addrAfter)

	m.Stop() //nolint:errcheck
}
