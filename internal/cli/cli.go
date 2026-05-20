package cli

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/nogo/wire/internal"
	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/gateway"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/process"
	"github.com/nogo/wire/internal/vram"
)

const usageText = `Usage: wire <command> [arguments]

Commands:
  up                  Start the gateway (foreground)
  version             Print version and exit

Environment:
  WIRE_CONFIG         Config file path (default: ~/.config/wire/config.yaml)
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
	case "up":
		return runUp()
	case "version", "--version", "-v":
		fmt.Printf("wire %s (%s) built %s\n", internal.Version, internal.Commit, internal.BuildDate)
		return nil
	default:
		fmt.Print(usageText)
		return fmt.Errorf("unknown command: %q", args[0])
	}
}

func runUp() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	// bgCtx drives the process manager; cancelled only after gateway drains.
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()

	// shutCtx is cancelled by SIGTERM/SIGINT.
	shutCtx, stop := signal.NotifyContext(bgCtx, syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	// 2. VRAM tracker init (GPU auto-detect, HF cache scan).
	backend := detectGPUBackend(cfg)
	tracker := vram.New(backend, vram.NewHFCache(cfg.HFCacheDir))
	gpus := tracker.Discover()
	tracker.HFCache().Scan()
	tracker.Start()
	defer tracker.Stop()

	// 3. Process manager init.
	mgr := process.New(cfg, slog.Default())
	if err := mgr.Start(bgCtx); err != nil {
		return fmt.Errorf("starting process manager: %w", err)
	}

	// Create remote tracker if any model references a provider.
	var remote *monitor.RemoteTracker
	for _, p := range cfg.Models {
		if p.IsRemote() {
			remote = monitor.NewRemoteTracker()
			break
		}
	}

	// Create model discovery for providers that have list_models enabled.
	var discovery *monitor.ModelDiscovery
	if len(cfg.Providers) > 0 {
		var endpoints []monitor.ProviderEndpoint
		for name, p := range cfg.Providers {
			if p.ShouldListModels() {
				endpoints = append(endpoints, monitor.ProviderEndpoint{
					Name:    name,
					BaseURL: p.BaseURL,
					APIKey:  p.APIKey,
				})
			}
		}
		if len(endpoints) > 0 {
			discovery = monitor.NewModelDiscovery(endpoints, slog.Default())
			discovery.Start()
			defer discovery.Stop()
		}
	}

	// 4. Hub init (subscribes to VRAM tracker for GPU state).
	var hub *monitor.Hub
	if cfg.Monitor {
		hub = monitor.NewHub(slog.Default(), mgr.Events(), mgr.Snapshot, remote, discovery, tracker.GPUStateCh())
	}

	// 5. Gateway init (uses VRAM tracker + queue).
	gw := gateway.New(cfg, mgr, hub, remote, discovery, slog.Default())

	if hub != nil {
		hub.SetModelProvider(func() []monitor.ModelInfo {
			return gw.Models()
		})
		hub.Start()
		defer hub.Stop()
	}

	// 6. Warmup models submitted through gateway's admission path (config order,
	// bounded retry). Runs in background so the gateway serves requests immediately.
	go gw.Warmup(bgCtx)

	gwErrCh := make(chan error, 1)
	go func() { gwErrCh <- gw.ListenAndServe(shutCtx) }()

	var reloadCh <-chan *config.Config
	if ch, err := config.Watch(bgCtx, slog.Default()); err != nil {
		slog.Warn("config watcher unavailable", "err", err)
	} else {
		reloadCh = ch
	}

	slog.Info("wire ready", "listen", cfg.Listen, "version", internal.Version, "commit", internal.Commit, "gpus", len(gpus))

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
			stop()
			if err := <-gwErrCh; err != nil {
				slog.Error("gateway shutdown error", "error", err)
			}
			bgCancel()
			return mgr.Stop()

		case newCfg, ok := <-reloadCh:
			if !ok || newCfg == nil {
				reloadCh = nil
				continue
			}
			if newCfg.Listen != cfg.Listen {
				slog.Warn("listen address change requires restart")
			}
			if newCfg.Monitor != cfg.Monitor {
				slog.Warn("monitor setting change requires restart")
			}
			slog.Info("config reloaded")
			mgr.Reload(newCfg)
			gw.UpdateConfig(newCfg)
			cfg = newCfg
		}
	}
}

// detectGPUBackend returns the appropriate vram.Backend following the
// auto-detection chain: AMD sysfs → noop (CPU-only).
func detectGPUBackend(_ *config.Config) vram.Backend {
	sysfs := vram.NewSysfsBackend()
	if gpus := sysfs.Discover(); len(gpus) > 0 {
		return sysfs
	}
	return vram.NoopBackend{}
}
