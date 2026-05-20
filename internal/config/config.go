package config

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration is a time.Duration that unmarshals from a YAML string like "30s".
type Duration time.Duration

func (d *Duration) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}
	dur, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	*d = Duration(dur)
	return nil
}

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

// Provider defines a remote OpenAI-compatible API provider.
type Provider struct {
	BaseURL    string `yaml:"base_url"`
	APIKey     string `yaml:"api_key"`
	ListModels *bool  `yaml:"list_models"`
}

// ShouldListModels reports whether Wire should fetch the provider's model list.
// Defaults to true when not explicitly set.
func (p Provider) ShouldListModels() bool {
	return p.ListModels == nil || *p.ListModels
}

type Config struct {
	Listen       string                 `yaml:"listen"`
	Monitor      bool                   `yaml:"monitor"`
	MaxModels    int                    `yaml:"max_models"`
	HFCacheDir   string                 `yaml:"hf_cache_dir"`
	DrainTimeout Duration               `yaml:"drain_timeout"`
	QueueDepth   int                    `yaml:"queue_depth"`
	Providers    map[string]Provider    `yaml:"providers"`
	Models       map[string]ModelParams `yaml:"models"`
	Warmup       []string               `yaml:"warmup"`
}

type ModelParams struct {
	Context    int      `yaml:"context"`
	GPULayers  int      `yaml:"gpu_layers"`
	Slots      int      `yaml:"slots"`
	Pooling    string   `yaml:"pooling"`
	Extra      []string `yaml:"extra"`
	SpecType   string   `yaml:"spec_type"`
	DraftMax   int      `yaml:"draft_max"`
	DraftMin   int      `yaml:"draft_min"`
	SpecNgramN int      `yaml:"spec_ngram_n"`
	SpecNgramM int      `yaml:"spec_ngram_m"`

	// Remote provider fields. When Provider is set, this model proxies to a
	// remote API instead of a local llama-server process.
	Provider      string   `yaml:"provider"`
	UpstreamModel string   `yaml:"upstream_model"`
	Capabilities  []string `yaml:"capabilities"`
}

// IsRemote reports whether this model is backed by a remote provider.
func (p ModelParams) IsRemote() bool { return p.Provider != "" }

// Equal reports whether p and other have identical parameters.
// []string (Extra) is compared element-by-element.
func (p ModelParams) Equal(other ModelParams) bool {
	if p.Context != other.Context ||
		p.GPULayers != other.GPULayers ||
		p.Slots != other.Slots ||
		p.Pooling != other.Pooling ||
		p.SpecType != other.SpecType ||
		p.DraftMax != other.DraftMax ||
		p.DraftMin != other.DraftMin ||
		p.SpecNgramN != other.SpecNgramN ||
		p.SpecNgramM != other.SpecNgramM ||
		p.Provider != other.Provider ||
		p.UpstreamModel != other.UpstreamModel ||
		len(p.Extra) != len(other.Extra) ||
		len(p.Capabilities) != len(other.Capabilities) {
		return false
	}
	for i, v := range p.Extra {
		if v != other.Extra[i] {
			return false
		}
	}
	for i, v := range p.Capabilities {
		if v != other.Capabilities[i] {
			return false
		}
	}
	return true
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

	expandProviderEnvVars(cfg)
	applyDefaults(cfg)

	if err := validate(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// knownProviderBaseURLs maps well-known provider names to their default API base URLs.
var knownProviderBaseURLs = map[string]string{
	"openai":     "https://api.openai.com",
	"openrouter": "https://openrouter.ai/api",
	"groq":       "https://api.groq.com/openai",
	"together":   "https://api.together.xyz",
	"mistral":    "https://api.mistral.ai",
	"deepseek":   "https://api.deepseek.com",
}

// expandProviderEnvVars replaces ${VAR} references in provider API keys with
// the corresponding environment variable value.
func expandProviderEnvVars(cfg *Config) {
	for name, p := range cfg.Providers {
		if strings.HasPrefix(p.APIKey, "${") && strings.HasSuffix(p.APIKey, "}") {
			envVar := p.APIKey[2 : len(p.APIKey)-1]
			p.APIKey = os.Getenv(envVar)
			cfg.Providers[name] = p
		}
	}
}

func applyDefaults(cfg *Config) {
	if cfg.Listen == "" {
		cfg.Listen = ":9480"
	}
	if cfg.MaxModels == 0 {
		cfg.MaxModels = 1
	}
	if cfg.HFCacheDir == "" {
		home, _ := os.UserHomeDir()
		cfg.HFCacheDir = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	if cfg.DrainTimeout == Duration(0) {
		cfg.DrainTimeout = Duration(30 * time.Second)
	} else {
		_ = time.Duration(cfg.DrainTimeout) // ensure valid
	}
	if cfg.QueueDepth == 0 {
		cfg.QueueDepth = 200
	}
	for pattern, p := range cfg.Models {
		if p.Slots == 0 && !p.IsRemote() {
			p.Slots = 1
			cfg.Models[pattern] = p
		}
	}
	for name, p := range cfg.Providers {
		if p.BaseURL == "" {
			if defaultURL, ok := knownProviderBaseURLs[name]; ok {
				p.BaseURL = defaultURL
				cfg.Providers[name] = p
			}
		}
	}
}

func validate(cfg *Config) error {
	// Validate providers.
	for name, p := range cfg.Providers {
		if p.APIKey == "" {
			return fmt.Errorf("provider %q: api_key is empty (set the environment variable or provide a literal value)", name)
		}
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q: base_url is required (no known default for this provider name)", name)
		}
	}

	// Validate warmup entries. Unmatched entries are dropped with a warning so
	// the daemon still starts when a model has been removed from the config.
	kept := cfg.Warmup[:0]
	for _, entry := range cfg.Warmup {
		var matches int
		for pattern, p := range cfg.Models {
			matched, err := filepath.Match(pattern, entry)
			if err == nil && matched {
				if p.IsRemote() {
					return fmt.Errorf("warmup entry %q references a remote model (remote models cannot be warmed up)", entry)
				}
				matches++
			}
		}
		if matches == 0 {
			slog.Warn("warmup entry does not match any model pattern; skipping", "entry", entry)
			continue
		}
		if matches > 1 {
			return fmt.Errorf("warmup entry %q matches %d model patterns (expected 1)", entry, matches)
		}
		kept = append(kept, entry)
	}
	cfg.Warmup = kept
	if len(cfg.Warmup) > cfg.MaxModels {
		return fmt.Errorf("warmup list length %d exceeds max_models %d", len(cfg.Warmup), cfg.MaxModels)
	}

	validSpecTypes := map[string]bool{
		"ngram-simple":  true,
		"ngram-mod":     true,
		"ngram-cache":   true,
		"ngram-map-k":   true,
		"ngram-map-k4v": true,
	}
	for pattern, p := range cfg.Models {
		if p.SpecType != "" && !validSpecTypes[p.SpecType] {
			return fmt.Errorf("model pattern %q: invalid spec_type %q (valid: ngram-simple, ngram-mod, ngram-cache, ngram-map-k, ngram-map-k4v)", pattern, p.SpecType)
		}

		// Remote model validation.
		if p.IsRemote() {
			if _, ok := cfg.Providers[p.Provider]; !ok {
				return fmt.Errorf("model pattern %q: references unknown provider %q", pattern, p.Provider)
			}
			if p.Context != 0 || p.GPULayers != 0 || p.Slots != 0 || p.Pooling != "" || len(p.Extra) != 0 || p.SpecType != "" {
				return fmt.Errorf("model pattern %q: remote models cannot set local-only fields (context, gpu_layers, slots, pooling, extra, spec_type)", pattern)
			}
		}
	}
	return nil
}
