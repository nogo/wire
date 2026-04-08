package config

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Path returns the resolved config file path.
// Resolution order: WIRE_CONFIG env var → XDG_CONFIG_HOME/wire/config.yaml →
// ~/.config/wire/config.yaml.
func Path() string {
	if v := os.Getenv("WIRE_CONFIG"); v != "" {
		return v
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "wire", "config.yaml")
}

type Config struct {
	Listen    string                 `yaml:"listen"`
	Monitor   bool                   `yaml:"monitor"`
	MaxModels int                    `yaml:"max_models"`
	Models    map[string]ModelParams `yaml:"models"`
	Warmup    []string               `yaml:"warmup"`
}

type ModelParams struct {
	Context   int      `yaml:"context"`
	GPULayers int      `yaml:"gpu_layers"`
	Slots     int      `yaml:"slots"`
	Pooling   string   `yaml:"pooling"`
	Extra     []string `yaml:"extra"`
}

// MatchModel returns the ModelParams for the first glob pattern in Models that
// matches name. Returns zero ModelParams and false if no pattern matches.
func (c *Config) MatchModel(name string) (ModelParams, bool) {
	for pattern, params := range c.Models {
		matched, err := filepath.Match(pattern, name)
		if err == nil && matched {
			return params, true
		}
	}
	return ModelParams{}, false
}

// Load reads the config from disk, applies defaults, and validates it.
func Load() (*Config, error) {
	path := Path()

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("config file not found at %s: %w", path, err)
		}
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}

	cfg := new(Config)
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("parsing config %s: %w", path, err)
	}

	applyDefaults(cfg)

	if err := validate(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

func applyDefaults(cfg *Config) {
	if cfg.Listen == "" {
		cfg.Listen = ":9480"
	}
	if cfg.MaxModels == 0 {
		cfg.MaxModels = 1
	}
	for pattern, p := range cfg.Models {
		if p.Slots == 0 {
			p.Slots = 1
			cfg.Models[pattern] = p
		}
	}
}

func validate(cfg *Config) error {
	for _, entry := range cfg.Warmup {
		var matches int
		for pattern := range cfg.Models {
			matched, err := filepath.Match(pattern, entry)
			if err == nil && matched {
				matches++
			}
		}
		if matches == 0 {
			return fmt.Errorf("warmup entry %q does not match any model pattern", entry)
		}
		if matches > 1 {
			return fmt.Errorf("warmup entry %q matches %d model patterns (expected 1)", entry, matches)
		}
	}
	if len(cfg.Warmup) > cfg.MaxModels {
		return fmt.Errorf("warmup list length %d exceeds max_models %d", len(cfg.Warmup), cfg.MaxModels)
	}
	return nil
}
