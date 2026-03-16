package main

import (
	"log/slog"
	"os"

	"github.com/nogo/wire/internal/config"
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
}
