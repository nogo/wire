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
- wu-07: Parse llama-server events and collect metrics
- wu-08: Serve live monitor dashboard via websocket
- wu-09: Route requests by prompt prefix for slot affinity

### Future (not planned)
- Response caching: LRU cache keyed by request body hash for deterministic requests (temperature=0). Return cached responses without GPU. Especially valuable for embeddings (same text re-embedded during RAG development) and agent retries. Invalidate on model swap.
- Overflow routing: dynamic fallback from saturated local backend to external backend (OpenAI, Anthropic). Trigger on slot exhaustion + queue depth threshold. Client-transparent — same endpoint, same format. Extend route table with fallback chain instead of static model→backend mapping.
- Request prioritization: assign priority by endpoint type (`/v1/chat/completions` = high, `/v1/embeddings` = low), streaming flag, or custom `X-Wire-Priority` header. Gateway holds low-priority requests while high-priority ones get immediate slot access.
- Context window guard: estimate token count from request, reject with clear error before llama-server truncates silently. Optional auto-truncate (sliding window) at proxy layer.
- Request coalescing: deduplicate concurrent identical embedding requests — one GPU call, fan out response to all waiting clients.
- Thermal throttle management: detect throttling via sysfs (clock below max DPM level while GPU saturated), proactive admission control at gateway (queue requests, reduce effective concurrency, Retry-After headers), optional clock capping via `pp_od_clk_voltage` to trade peak speed for steady throughput. Depends on wu-07/08 GPU monitoring foundation.
- Federation: peer discovery and cross-node routing
- Metrics: Prometheus endpoint for usage tracking
- Reranking: `/v1/rerank` backend support
- Auth: per-app API keys
