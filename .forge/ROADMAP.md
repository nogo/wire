# Wire — Inference Gateway

Single-port OpenAI-compatible gateway that manages llama-server processes and routes requests to the correct backend by model name. Supports managed backends (llama-server children) and external backends (Ollama, OpenAI, Anthropic).

## Architecture

```
app → wire (:8080) → /v1/chat/completions → llama-server (chat, internal port)
                   → /v1/embeddings       → llama-server (embedding, internal port)
                   → /v1/chat/completions  → api.anthropic.com (external, translated)
```

## Tech Stack
- Go 1.26, standard library + yaml.v3
- llama-server via `-hf` for managed backends
- ROCm build (AMD GPU, target configurable)

## Design Decisions
- **Two llama-server instances, not router mode**: embedding requires `--pooling mean` which conflicts with chat. Router mode applies flags globally.
- **No admin API**: CLI manages processes via signals (SIGHUP for reload, SIGTERM for stop). Status queries the public API.
- **Config as source of truth**: `~/.config/wire/config.yaml` stores current state. `wire swap` updates config + signals reload.
- **Stream without buffering**: reverse proxy streams SSE responses directly. No full-body buffering.
- **Anthropic translation in-process**: not via reverse proxy (needs request/response rewriting). Uses forge's proven translation logic.

## Status

### Pending
- wu-01: Scaffold project and load config
- wu-02: Manage llama-server child processes
- wu-03: Route requests through HTTP gateway
- wu-04: Add CLI commands
- wu-05: Translate Anthropic API format
- wu-06: Add build script and systemd service

### Future (not planned)
- Federation: peer discovery and cross-node routing
- Metrics: Prometheus endpoint for usage tracking
- Reranking: `/v1/rerank` backend support
- Auth: per-app API keys
