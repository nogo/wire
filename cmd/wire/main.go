package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/nogo/wire/internal/config"
	"github.com/nogo/wire/internal/process"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	slog.Info("config loaded",
		"listen", cfg.Listen,
		"backends", len(cfg.Backends),
		"peers", len(cfg.Peers),
	)

	for name, b := range cfg.Backends {
		if b.IsManaged() {
			slog.Info("backend",
				"name", name,
				"type", "managed",
				"model", b.Model,
				"context", b.Context,
				"gpu_layers", b.GPULayers,
			)
		} else {
			slog.Info("backend",
				"name", name,
				"type", "external",
				"url", b.URL,
				"provider", b.Provider,
			)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	mgr := process.New(cfg, slog.Default())
	if err := mgr.Start(ctx); err != nil {
		slog.Error("failed to start process manager", "error", err)
		os.Exit(1)
	}

	<-ctx.Done()

	if cause := context.Cause(ctx); cause != nil {
		slog.Info("shutting down", "cause", cause)
	}
	stop() // release signal resources

	if err := mgr.Stop(); err != nil {
		slog.Error("error stopping process manager", "error", err)
		os.Exit(1)
	}
}
