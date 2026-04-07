package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"flag"

	"github.com/nogo/wire/internal/build"
	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/gateway"
	"github.com/nogo/wire/internal/pidfile"
	"github.com/nogo/wire/internal/process"
)

const usageText = `Usage: wire <command> [arguments]

Commands:
  start               Start the gateway (foreground)
  stop                Stop the running gateway
  status              Show backend status
  swap <backend> <model>  Hot-swap a backend's model
  build               Build and install llama-server from source (ROCm)
  remove              Remove a previous llama-server installation

Environment:
  WIRE_CONFIG         Config file path (default: ~/.config/wire/config.yaml)
  WIRE_LLAMA_SRC      llama.cpp source directory (default: ~/Development/llama.cpp)
  WIRE_GPU_TARGET     AMD GPU target (default: gfx1151; find yours: rocminfo | grep gfx)
  WIRE_PREFIX         Install prefix (default: ~/.local)
`

// Run dispatches CLI commands from the given argument slice (os.Args[1:]).
// It returns an exit code suitable for os.Exit.
func Run(args []string) int {
	if err := dispatch(args); err != nil {
		fmt.Fprintln(os.Stderr, "wire:", err)
		return 1
	}
	return 0
}

func dispatch(args []string) error {
	if len(args) == 0 {
		fmt.Print(usageText)
		return nil
	}
	switch args[0] {
	case "help", "--help", "-h":
		fmt.Print(usageText)
		return nil
	case "start":
		return runStart()
	case "stop":
		return runStop()
	case "status":
		return runStatus()
	case "swap":
		if len(args) < 3 {
			return fmt.Errorf("swap requires <backend> and <model>\nUsage: wire swap <backend> <model>")
		}
		return runSwap(args[1], args[2])
	case "build":
		return runBuild(args[1:])
	case "remove":
		return runRemove(args[1:])
	default:
		fmt.Print(usageText)
		return fmt.Errorf("unknown command: %q", args[0])
	}
}

// listenURL converts the listen address from config to an http:// URL.
func listenURL(listen string) string {
	if strings.HasPrefix(listen, ":") {
		return "http://localhost" + listen
	}
	return "http://" + listen
}

func runStart() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	pidPath := pidfile.Path()
	if err := pidfile.Write(pidPath); err != nil {
		return err
	}
	defer os.Remove(pidPath)

	// bgCtx drives the process manager; cancelled only after gateway drains.
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()

	// shutCtx is cancelled by SIGTERM/SIGINT. Its Cause is the received signal.
	shutCtx, stop := signal.NotifyContext(bgCtx, syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	// sighupCh receives SIGHUP for config reload.
	sighupCh := make(chan os.Signal, 1)
	signal.Notify(sighupCh, syscall.SIGHUP)
	defer signal.Stop(sighupCh)

	mgr := process.New(cfg, slog.Default())
	if err := mgr.Start(bgCtx); err != nil {
		return fmt.Errorf("starting process manager: %w", err)
	}

	gw := gateway.New(cfg, mgr, slog.Default())
	gwErrCh := make(chan error, 1)
	go func() { gwErrCh <- gw.ListenAndServe(shutCtx) }()

	slog.Info("wire ready", "listen", cfg.Listen)

	// swapMu serialises SIGHUP handling so overlapping reloads are dropped.
	var swapMu sync.Mutex

	for {
		select {
		case err := <-gwErrCh:
			bgCancel()
			_ = mgr.Stop()
			if err != nil {
				return fmt.Errorf("gateway error: %w", err)
			}
			return nil

		case <-shutCtx.Done():
			if cause := context.Cause(shutCtx); cause != nil {
				slog.Info("shutting down", "cause", cause)
			}
			stop() // release signal resources early
			// Wait for gateway to drain connections, then stop process manager.
			if err := <-gwErrCh; err != nil {
				slog.Error("gateway shutdown error", "error", err)
			}
			bgCancel()
			return mgr.Stop()

		case <-sighupCh:
			if !swapMu.TryLock() {
				slog.Warn("SIGHUP received while swap in progress, ignoring")
				continue
			}
			go func() {
				defer swapMu.Unlock()
				newCfg, loadErr := config.Load()
				if loadErr != nil {
					slog.Error("failed to reload config on SIGHUP", "error", loadErr)
					return
				}
				for name, b := range newCfg.Backends {
					oldB, ok := cfg.Backends[name]
					if !ok || !b.IsManaged() {
						continue
					}
					if b.Model != oldB.Model {
						slog.Info("swapping backend model", "backend", name, "old", oldB.Model, "new", b.Model)
						if swapErr := mgr.Swap(bgCtx, name, b.Model); swapErr != nil {
							slog.Error("model swap failed", "backend", name, "error", swapErr)
						}
					}
				}
				cfg = newCfg
			}()
		}
	}
}

func runStop() error {
	pidPath := pidfile.Path()
	pid, err := pidfile.Read(pidPath)
	if err != nil {
		return err
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("finding process %d: %w", pid, err)
	}
	if err := proc.Signal(syscall.SIGTERM); err != nil {
		return fmt.Errorf("sending SIGTERM to pid %d: %w", pid, err)
	}

	// Poll /proc/<pid> via kill -0 for up to 10 seconds.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if err := proc.Signal(syscall.Signal(0)); err != nil {
			fmt.Printf("wire stopped (pid %d)\n", pid)
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}

	// Still alive: force kill.
	_ = proc.Signal(syscall.SIGKILL)
	fmt.Printf("wire force-killed (pid %d)\n", pid)
	return nil
}

func runStatus() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	baseURL := listenURL(cfg.Listen)
	client := &http.Client{Timeout: 5 * time.Second}

	healthResp, err := client.Get(baseURL + "/health")
	if err != nil {
		fmt.Println("wire is not running")
		os.Exit(1)
	}
	defer healthResp.Body.Close()

	var healthBody struct {
		Status string `json:"status"`
	}
	_ = json.NewDecoder(healthResp.Body).Decode(&healthBody)
	overall := healthBody.Status
	if overall == "" {
		if healthResp.StatusCode == http.StatusOK {
			overall = "healthy"
		} else {
			overall = "degraded"
		}
	}
	fmt.Printf("wire: %s (%s)\n", overall, baseURL)

	modelsResp, err := client.Get(baseURL + "/v1/models")
	if err != nil {
		return nil
	}
	defer modelsResp.Body.Close()

	var modelsBody struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(modelsResp.Body).Decode(&modelsBody); err != nil {
		return nil
	}

	for _, m := range modelsBody.Data {
		b, ok := cfg.Backends[m.ID]
		model := m.ID
		if ok {
			if b.IsManaged() {
				model = b.Model
			} else {
				model = b.URL
			}
		}
		fmt.Printf("  %-14s %-40s %s\n", m.ID+":", model, overall)
	}
	return nil
}

func runSwap(backend, model string) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	b, ok := cfg.Backends[backend]
	if !ok {
		return fmt.Errorf("backend %q not found in config", backend)
	}
	if b.IsExternal() {
		return fmt.Errorf("backend %q is external and cannot be swapped", backend)
	}

	b.Model = model
	cfg.Backends[backend] = b
	if err := config.WriteFile(cfg); err != nil {
		return err
	}

	pidPath := pidfile.Path()
	pid, err := pidfile.Read(pidPath)
	if err != nil {
		return err
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("finding process: %w", err)
	}
	if err := proc.Signal(syscall.SIGHUP); err != nil {
		return fmt.Errorf("sending SIGHUP to pid %d: %w", pid, err)
	}

	fmt.Printf("sent reload signal to wire (pid %d); run 'wire status' to check progress\n", pid)
	return nil
}

func runBuild(args []string) error {
	opts, err := build.DefaultOptions()
	if err != nil {
		return err
	}

	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.StringVar(&opts.Source, "source", opts.Source, "path to llama.cpp git checkout")
	fs.StringVar(&opts.GPUTarget, "gpu-target", opts.GPUTarget, "AMD GPU target (find yours: rocminfo | grep gfx)")
	fs.StringVar(&opts.Prefix, "prefix", opts.Prefix, "install prefix")
	if err := fs.Parse(args); err != nil {
		return err
	}

	return build.Build(opts)
}

func runRemove(args []string) error {
	opts, err := build.DefaultOptions()
	if err != nil {
		return err
	}

	fs := flag.NewFlagSet("remove", flag.ContinueOnError)
	fs.StringVar(&opts.Source, "source", opts.Source, "path to llama.cpp git checkout")
	if err := fs.Parse(args); err != nil {
		return err
	}

	return build.Remove(opts)
}
