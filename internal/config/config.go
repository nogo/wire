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

// WriteFile marshals cfg to YAML and atomically replaces the config file.
func WriteFile(cfg *Config) error {
	path := Path()
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return fmt.Errorf("writing config temp file: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("renaming config file: %w", err)
	}
	return nil
}

type Config struct {
	Listen   string             `yaml:"listen"`
	Backends map[string]Backend `yaml:"backends"`
	Peers    []string           `yaml:"peers"`
}

type Backend struct {
	// Managed backend (llama-server)
	Model     string   `yaml:"model"`      // HuggingFace ref, e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
	Context   int      `yaml:"context"`    // Context window size
	GPULayers int      `yaml:"gpu_layers"` // -ngl flag
	Pooling   string   `yaml:"pooling"`    // "mean", "cls", "rank" — triggers --embedding flag
	Extra     []string `yaml:"extra"`      // Additional llama-server flags

	// External backend (Ollama, OpenAI, Anthropic, etc.)
	URL      string `yaml:"url"`      // If set, backend is external — no process management
	APIKey   string `yaml:"api_key"`  // Bearer token or x-api-key
	Provider string `yaml:"provider"` // "openai" (default) or "anthropic"
}

// IsManaged reports whether wire manages the llama-server process for this backend.
func (b Backend) IsManaged() bool { return b.URL == "" }

// IsExternal reports whether this backend is an external service.
func (b Backend) IsExternal() bool { return !b.IsManaged() }

// ValidationError describes a config validation failure.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("config validation: %s: %s", e.Field, e.Message)
}

// Load reads the config from disk, applies defaults, and validates it.
// Path resolution order: WIRE_CONFIG env var → XDG_CONFIG_HOME/wire/config.yaml →
// ~/.config/wire/config.yaml.
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
		cfg.Listen = ":8080"
	}
	for name, b := range cfg.Backends {
		if b.Provider == "" {
			b.Provider = "openai"
		}
		cfg.Backends[name] = b
	}
}

func validate(cfg *Config) error {
	for name, b := range cfg.Backends {
		switch {
		case b.Model != "" && b.URL != "":
			return &ValidationError{
				Field:   "backends." + name,
				Message: "cannot set both model and url",
			}
		case b.Model == "" && b.URL == "":
			return &ValidationError{
				Field:   "backends." + name,
				Message: "must set either model or url",
			}
		}
	}
	return nil
}
