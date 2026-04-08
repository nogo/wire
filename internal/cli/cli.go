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
