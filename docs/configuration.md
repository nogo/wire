# Configuration

Wire reads its configuration from a single YAML file.

**Resolution order:**
1. `WIRE_CONFIG` environment variable
2. `$XDG_CONFIG_HOME/wire/config.yaml`
3. `~/.config/wire/config.yaml`

Changes to the config file are picked up automatically (no restart required), except for `listen` and `monitor` which require a restart.

Unknown fields cause a parse error.

---

## Full example

```yaml
listen: ":9480"
monitor: true
max_models: 3

providers:
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
  openai:
    api_key: ${OPENAI_API_KEY}
    list_models: false
  custom:
    base_url: https://my-proxy.example.com/v1
    api_key: ${CUSTOM_API_KEY}
    list_models: false

models:
  "org/Qwen3-30B-A3B":
    context: 32768
    gpu_layers: 99
    slots: 4
    capabilities: [chat, code]

  "org/nomic-embed-*":
    context: 8192
    gpu_layers: 99
    pooling: mean
    capabilities: [embeddings]

  "org/Qwen3-8B":
    context: 16384
    gpu_layers: 99
    spec_type: ngram-mod
    capabilities: [chat]

  "claude-sonnet":
    provider: openrouter
    upstream_model: anthropic/claude-3.5-sonnet
    capabilities: [chat, code, tools, vision]

  "gpt-4o":
    provider: openai
    upstream_model: gpt-4o
    capabilities: [chat, code, tools, vision, long-context]

  "deepseek-r1":
    provider: openrouter
    upstream_model: deepseek/deepseek-r1
    capabilities: [chat, code, reasoning]

warmup:
  - "org/Qwen3-30B-A3B"
```

---

## Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `listen` | string | `":9480"` | Address the HTTP server binds to. |
| `monitor` | bool | `false` | Enable the real-time monitor dashboard at `/monitor`. |
| `max_models` | int | `1` | Maximum number of concurrent local llama-server processes. Does not count remote models. |
| `providers` | map | — | Remote API provider definitions. See [Providers](#providers). |
| `models` | map | — | Model patterns and their parameters. Keys are glob patterns. See [Models](#models). |
| `warmup` | list | `[]` | Model names to pre-load at startup. Each entry must match exactly one model pattern. Only local models can be warmed up. |

---

## Providers

The `providers` section defines remote OpenAI-compatible API providers. Define a provider once, reference it from any number of models.

```yaml
providers:
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_url` | string | Depends | — | Base URL of the provider's API. Required for unknown provider names. See [known providers](#known-providers) for defaults. |
| `api_key` | string | Yes | — | API key for authentication. Supports `${ENV_VAR}` syntax for environment variable expansion. |
| `list_models` | bool | No | `true` | Fetch the provider's model list via `GET /v1/models`. Set `false` to disable. See [Model discovery](#model-discovery). |

### Known providers

These provider names have default `base_url` values. You can omit `base_url` when using one of these names:

| Name | Default base URL |
|---|---|
| `openai` | `https://api.openai.com` |
| `openrouter` | `https://openrouter.ai/api` |
| `groq` | `https://api.groq.com/openai` |
| `together` | `https://api.together.xyz` |
| `mistral` | `https://api.mistral.ai` |
| `deepseek` | `https://api.deepseek.com` |

Any other provider name requires an explicit `base_url`.

### API key environment variables

Use `${VAR_NAME}` to read the API key from an environment variable at startup:

```yaml
providers:
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
```

The variable is expanded when the config is loaded. If the variable is empty or unset, Wire exits with an error.

---

## Models

The `models` section maps glob patterns to model parameters. When a request arrives with a `model` field, Wire matches it against these patterns (first match wins).

A model is either **local** (backed by a llama-server process) or **remote** (proxied to a provider). The `provider` field determines which: absent means local, present means remote.

### Local model fields

```yaml
models:
  "org/Qwen3-*":
    context: 32768
    gpu_layers: 99
    slots: 4
    pooling: mean
    extra: ["--flash-attn"]
    capabilities: [chat, code]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `context` | int | 0 (llama-server default) | Context window size (`-c` flag). |
| `gpu_layers` | int | 0 | Number of layers to offload to GPU (`-ngl` flag). |
| `slots` | int | 1 | Number of concurrent request slots (`--parallel` flag). |
| `pooling` | string | — | Embedding pooling mode. When set, enables `--embedding --pooling <value>`. Values: `mean`, `cls`, `last`. |
| `extra` | list | `[]` | Extra CLI flags passed directly to llama-server. |
| `capabilities` | list | `[]` | Informational tags exposed via `/v1/models`. Not used for routing. |

### Speculative decoding fields (local only)

| Field | Type | Default | Description |
|---|---|---|---|
| `spec_type` | string | — | N-gram speculative decoding mode. Valid: `ngram-simple`, `ngram-mod`, `ngram-cache`, `ngram-map-k`, `ngram-map-k4v`. |
| `draft_max` | int | 0 | Maximum draft tokens for speculative decoding. |
| `draft_min` | int | 0 | Minimum draft tokens for speculative decoding. |
| `spec_ngram_n` | int | 0 | N-gram size for n-gram speculation. |
| `spec_ngram_m` | int | 0 | N-gram context window size. |

### Remote model fields

```yaml
models:
  "claude-sonnet":
    provider: openrouter
    upstream_model: anthropic/claude-3.5-sonnet
    capabilities: [chat, code, tools, vision]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `provider` | string | Yes | Name of a provider defined in the `providers` section. |
| `upstream_model` | string | — | Model name sent to the upstream API. If omitted, the Wire model name (the pattern key) is used as-is. |
| `capabilities` | list | — | Informational tags exposed via `/v1/models`. |

Remote models cannot set local-only fields (`context`, `gpu_layers`, `slots`, `pooling`, `extra`, `spec_type`). Doing so is a validation error.

Remote models cannot appear in the `warmup` list.

---

## Glob patterns

Model keys support Go's `filepath.Match` glob syntax:

| Pattern | Matches |
|---|---|
| `"org/Qwen3-*"` | `org/Qwen3-8B`, `org/Qwen3-30B-A3B`, etc. |
| `"org/nomic-embed-*"` | `org/nomic-embed-text-v1.5`, etc. |
| `"gpt-4o"` | Exactly `gpt-4o`. |
| `"*"` | Any model name without a `/`. |

The first matching pattern wins. Pattern evaluation order is non-deterministic (map iteration), so avoid overlapping patterns that could match the same model name.

---

## Warmup

Models listed in `warmup` are loaded at startup and kept alive. They are restarted automatically on crash (up to 5 retries per 60-second window).

```yaml
warmup:
  - "org/Qwen3-30B-A3B"
```

Constraints:
- Each entry must match exactly one model pattern.
- The number of warmup entries cannot exceed `max_models`.
- Only local models can be warmed up.

Models not in the warmup list are loaded on demand when the first request arrives, and evicted (LRU) when capacity is reached.

---

## Model discovery

When `list_models` is `true` (the default), Wire fetches each provider's available models by calling `GET <base_url>/v1/models` on startup and every 5 minutes.

Discovered models are:

- **Listed in `/v1/models`** alongside explicitly configured models, with `owned_by` set to the provider name and `local: false`.
- **Routable.** A request for a discovered model is proxied to its provider, even if the model is not explicitly in the `models` config section. No model name rewriting occurs — the name passes through as-is.
- **Visible in the monitor dashboard** under a "Discovered Models" section showing each provider's model count and list.

### Precedence

Explicitly configured models (in the `models` section) always take precedence over discovered models. If a model name matches a config pattern, the config pattern's settings are used.

### Conflicts

If two providers list the same model ID, both appear in `/v1/models` (tagged with their respective provider). For routing, Wire picks the provider whose name is alphabetically first. To control routing precisely, add an explicit entry in the `models` section.

### Disabling discovery

Set `list_models: false` on a provider to skip fetching its model list:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    list_models: false  # only explicitly configured models
```

---

## Routing behavior

### Local models

Request arrives with `model: "org/Qwen3-8B"` -> Wire matches the pattern -> starts a llama-server process (or reuses an existing one) -> reverse-proxies the request to `localhost:<port>`. Slot affinity pins requests with matching system prompts to the same llama-server slot for KV cache reuse.

### Remote models

Request arrives with `model: "claude-sonnet"` -> Wire matches the pattern -> looks up the provider -> proxies to `https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer <key>`. If `upstream_model` is set, the `model` field in the request body is rewritten before forwarding.

Remote models bypass the process manager, slot affinity, context window guard, and `max_models` limit. They are always available.

The response includes an `X-Wire-Model` header with the actual upstream model name.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (local or remote). |
| POST | `/v1/completions` | Text completions (local or remote). |
| POST | `/v1/embeddings` | Embeddings (local or remote). |
| POST | `/v1/rerank` | Reranking (local only). |
| GET | `/v1/models` | List all models with status and capabilities. |
| GET | `/health` | Health check (200 if any warmup model is loaded, 503 otherwise). |
| GET | `/monitor` | Real-time dashboard (when `monitor: true`). |
| GET | `/monitor/ws` | WebSocket for dashboard updates. |

### `/v1/models` response

```json
{
  "object": "list",
  "data": [
    {
      "id": "org/Qwen3-30B-A3B",
      "object": "model",
      "owned_by": "wire",
      "status": "loaded",
      "capabilities": ["chat", "code"],
      "local": true
    },
    {
      "id": "claude-sonnet",
      "object": "model",
      "owned_by": "openrouter",
      "status": "available",
      "capabilities": ["chat", "code", "tools", "vision"],
      "local": false
    }
  ]
}
```

Clients can use the `capabilities` and `local` fields to decide which model to use. Wire does not perform automatic capability-based routing.

---

## Monitor

When `monitor: true`, the dashboard at `/monitor` shows:

- **GPU cards** -- utilization, VRAM, temperature, power (AMD sysfs).
- **Backend cards** -- local llama-server processes with context usage, token rates, queue depth, slot state.
- **Remote provider cards** -- in-flight requests, average latency, total requests, error count.
- **Discovered models** -- per-provider cards showing model count and a scrollable list of all available model IDs. Refreshed every 5 minutes.
- **Request log** -- prompt eval and generation eval events from local backends.

---

## Environment variables

| Variable | Description |
|---|---|
| `WIRE_CONFIG` | Override the config file path. |
| `XDG_CONFIG_HOME` | Base directory for config (default: `~/.config`). |
| Any variable referenced by `${...}` in `api_key` | Provider API keys. |
