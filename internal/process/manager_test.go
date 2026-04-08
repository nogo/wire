package process

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
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
func testCmdBuilder(ctx context.Context, modelName string, params config.ModelParams, port int) *exec.Cmd {
	args := append(
		[]string{"-test.run=TestHelperProcess", "--", "llama-server"},
		buildArgs(modelName, params, port)...,
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
	params := config.ModelParams{}
	got := buildArgs("org/model", params, 19100)
	want := []string{"-hf", "org/model", "--port", "19100", "--no-webui", "--parallel", "1"}
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
	got := buildArgs("org/model", params, 19101)
	want := []string{
		"-hf", "org/model",
		"--port", "19101",
		"--no-webui",
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

func TestBuildArgs_zeroGPULayersOmitted(t *testing.T) {
	params := config.ModelParams{GPULayers: 0, Context: 0}
	args := buildArgs("org/model", params, 19102)
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
// Manager tests
// ---------------------------------------------------------------------------

func TestManager_AddrMissing(t *testing.T) {
	m := New(&config.Config{}, slog.Default())
	_, ok := m.Addr("nonexistent")
	if ok {
		t.Error("Addr: expected ok=false for missing model")
	}
}

func TestManager_HealthyMissing(t *testing.T) {
	m := New(&config.Config{}, slog.Default())
	if m.Healthy("nonexistent") {
		t.Error("Healthy: expected false for unknown model")
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
		if m.Healthy("org/test-model") {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if !m.Healthy("org/test-model") {
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

	if m.Healthy("org/test-model") {
		t.Error("Healthy: expected false after Stop")
	}
}
