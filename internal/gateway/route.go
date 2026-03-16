package gateway

import (
	"fmt"
	"path"
	"strings"

	"github.com/nogo/wire/internal/config"
)

// routeTable maps model aliases to backend names.
type routeTable struct {
	aliases        map[string]string   // alias → backend name
	backendAliases map[string][]string // backend name → all its aliases
}

// buildRouteTable constructs the route table from config.
func buildRouteTable(cfg *config.Config) *routeTable {
	rt := &routeTable{
		aliases:        make(map[string]string),
		backendAliases: make(map[string][]string),
	}
	for name, b := range cfg.Backends {
		aliases := []string{name}
		if b.IsManaged() && b.Model != "" {
			aliases = append(aliases, modelAliases(b.Model)...)
		}
		rt.backendAliases[name] = aliases
		for _, alias := range aliases {
			if _, exists := rt.aliases[alias]; !exists {
				rt.aliases[alias] = name
			}
		}
	}
	return rt
}

// resolve returns the backend name for the given model string.
func (rt *routeTable) resolve(model string) (string, error) {
	if name, ok := rt.aliases[model]; ok {
		return name, nil
	}
	return "", fmt.Errorf("unknown model %q", model)
}

// modelAliases returns all routing aliases for a HuggingFace model ref.
//
// For "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M" this produces:
//   - "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"  (full ref)
//   - "Qwen3-Coder-30B-A3B-Instruct-GGUF"              (repo name)
//   - "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"            (filename stem — what llama-server reports)
//   - "Qwen3-Coder-30B-A3B-Instruct"                   (short form without GGUF/quantization)
func modelAliases(hfRef string) []string {
	aliases := []string{hfRef}

	repoPath, quant, _ := strings.Cut(hfRef, ":")
	repoName := path.Base(repoPath)
	if repoName != "" && repoName != "." {
		aliases = append(aliases, repoName)
	}

	// Strip the -GGUF suffix to get the model base name.
	baseName := strings.TrimSuffix(repoName, "-GGUF")
	baseName = strings.TrimSuffix(baseName, "-gguf")

	if quant != "" {
		// This is the filename stem llama-server sets as the model name.
		aliases = append(aliases, baseName+"-"+quant)
	}

	if baseName != "" && baseName != repoName {
		aliases = append(aliases, baseName)
	}

	return aliases
}
