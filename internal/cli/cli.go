package cli

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/nogo/wire/internal"
	"github.com/nogo/wire/internal/build"
	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/gateway"
	"github.com/nogo/wire/internal/monitor"
	"github.com/nogo/wire/internal/process"
)

const usageText = `Usage: wire <command> [arguments]

Commands:
  start               Start the gateway (foreground)
  build               Build and install llama-server from source (ROCm)
  remove              Remove a previous llama-server installation
  version             Print version and exit

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
	case "version", "--version", "-v":
		fmt.Printf("wire %s (%s) built %s\n", internal.Version, internal.Commit, internal.BuildDate)
		return nil
	case "build":
		return runBuild(args[1:])
	case "remove":
		return runRemove(args[1:])
	default:
		fmt.Print(usageText)
		return fmt.Errorf("unknown command: %q", args[0])
	}
}

func runStart() error {
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

	mgr := process.New(cfg, slog.Default())
	if err := mgr.Start(bgCtx); err != nil {
		return fmt.Errorf("starting process manager: %w", err)
	}

	var hub *monitor.Hub
	if cfg.Monitor {
		hub = monitor.NewHub(slog.Default(), mgr.Events(), mgr.Snapshot)
		hub.Start()
		defer hub.Stop()
	}

	gw := gateway.New(cfg, mgr, hub, slog.Default())
	gwErrCh := make(chan error, 1)
	go func() { gwErrCh <- gw.ListenAndServe(shutCtx) }()

	slog.Info("wire ready", "listen", cfg.Listen, "version", internal.Version, "commit", internal.Commit)

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
		}
	}
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
