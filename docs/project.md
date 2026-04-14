# Wire

Single-port OpenAI-compatible inference gateway that owns the full lifecycle of llama-server processes. One binary, one config file, one port. Applications talk OpenAI; Wire handles everything behind it — process supervision, model loading, slot routing, GPU awareness, live telemetry.

Wire is not a model runner. It is the layer between your application and llama-server that makes local inference feel like a managed service.

## What it does today

**Process management.** Supervises llama-server children with a four-state lifecycle (idle → loading → healthy → draining). Auto-restarts on crash (5 retries / 60s window). Warmup models load at startup; all others load on demand when the first request arrives. Concurrent requests for the same unloaded model coalesce into a single load. LRU eviction when at capacity — models with zero in-flight requests are drained first.

**Request routing.** Extracts the `model` field from every request and proxies to the correct llama-server. SSE responses stream without buffering. Returns OpenAI-compatible JSON errors. Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models`, `/health`.

**Slot affinity.** Hashes the system message (or first user message) and pins requests with matching prefixes to the same llama-server slot. Maximizes KV cache reuse across conversations with stable system prompts. LRU eviction when all slots are assigned.

**Monitoring.** Parses llama-server stderr into structured events (prompt eval, generation eval, slot assign/release, model load, VRAM allocation). Polls each backend's `/metrics` (Prometheus) and `/slots` (JSON) every second. Reads AMD GPU telemetry from sysfs (utilization, VRAM, temperature, power). Streams everything over WebSocket to an embedded real-time dashboard.

**Configuration.** Single YAML file (`~/.config/wire/config.yaml`). Glob patterns map model names to parameters (context size, GPU layers, slots, pooling mode, extra flags). `max_models` caps concurrent instances. `warmup` list pre-loads models at startup.

## Tech stack

- Go 1.26, standard library + yaml.v3 + nhooyr.io/websocket
- llama-server via `-hf` for model management
- AMD GPU telemetry via sysfs (no vendor SDK dependency)
- mise for build/test/install tasks
- systemd user service for deployment

## Design decisions

- **Process-per-model, not router mode.** Wire owns the full lifecycle. Embedding requires `--pooling mean` which conflicts with chat — router mode can't mix flags. Process isolation also gives clean crash boundaries and independent eviction.
- **No admin API.** `wire up` starts everything. Config is the source of truth. Restart to reconfigure.
- **Stream without buffering.** Reverse proxy streams SSE directly. No full-body buffering.
- **On-demand over pre-allocation.** Models load when first requested. Warmup is opt-in for latency-sensitive models. This keeps VRAM free for the model you actually need right now.
- **Count-based capacity, not VRAM-based.** `max_models` is simple and predictable. VRAM-aware scheduling is a roadmap goal.
- **Best-effort monitoring.** Event channel drops on full. Metric polls fail silently. The gateway never blocks on telemetry.
- **Pass-through for llama-server features.** Function calling, JSON schema, grammar, reasoning extraction, sampling parameters — all handled server-side. Wire forwards them unchanged. No gateway-side parsing or validation of model-specific features.

---

## Roadmap

Ordered by dependency and impact. Each item states what changes, why it matters, and what must hold true.

### VRAM-aware scheduling

Replace count-based `max_models` with VRAM budget tracking.

**Outcome.** Wire tracks VRAM consumed per loaded model (from stderr alloc events) and available VRAM (from sysfs). Load requests are admitted only when the model fits. Eviction targets the model whose removal frees enough VRAM for the incoming one, preferring LRU among candidates. `max_models` becomes a secondary ceiling, not the primary constraint.

**Value.** A 1B embedding model and a 70B chat model should not count the same against capacity. VRAM is the real bottleneck — scheduling against it prevents OOM kills and avoids leaving GPU memory unused when small models could still fit.

**Constraints.** Falls back to count-based eviction when VRAM data is unavailable (no AMD GPU, container without sysfs). Must not block the request path on sysfs reads — use the last polled value. Eviction still respects in-flight requests (never evict a model serving traffic).

---

### Slot KV cache persistence

Save slot state before eviction, restore on reload — eliminating cold-start penalty for rotated models.

**Outcome.** Before draining a model, Wire calls `POST /slots/{id}?action=save` on each active slot (requires `--slot-save-path` in llama-server config). When the same model is loaded again on demand, Wire restores the saved slots via `POST /slots/{id}?action=restore`. Requests that hit a restored slot skip prompt evaluation entirely.

**Value.** On-demand loading with LRU eviction means popular models get evicted and reloaded repeatedly. Without cache persistence, every reload is a full cold start — prompt eval for every slot from scratch. Save/restore turns the eviction cost from "lose all KV state" to "lose nothing, pay disk I/O."

**Constraints.** Save path must be on fast storage (NVMe, not spinning disk). Cache files are model-specific and slot-specific — stale files from a different model version must not be restored. Wire must handle save/restore failures gracefully (fall back to cold start). Save must complete before SIGTERM — add a drain timeout budget for slot saves.

---

### Speculative decoding orchestration

Manage draft model pairing for llama-server's speculative decoding.

**Outcome.** Config gains a `draft` field per model pattern specifying a smaller draft model. When Wire starts a llama-server for the main model, it passes `--model-draft`, `--draft-max`, and `--draft-min` flags. Wire tracks the draft model's VRAM cost alongside the main model for scheduling. N-gram modes (`ngram-simple`, `ngram-mod`) are exposed as config options without extra process management — they require no draft model.

**Value.** Speculative decoding delivers 2-5x generation speedup when draft predictions are accurate. N-gram modes are free (no extra VRAM, no extra process) and benefit from slot reuse — Wire's affinity routing directly improves n-gram hit rates. Draft model mode trades VRAM for throughput — Wire's VRAM-aware scheduling (prerequisite) ensures the draft model fits.

**Constraints.** Draft model mode requires VRAM-aware scheduling to land first (draft model adds to VRAM budget). N-gram modes have no prerequisites and can ship independently. Per-request override of speculative parameters (`speculative.n_max`, `speculative.p_min`) is pass-through — Wire forwards them unchanged. Wire does not validate draft/main model compatibility — llama-server rejects mismatches at load time.

---

### Context window guard

Reject requests that exceed the model's context window before llama-server silently truncates.

**Outcome.** Wire reads `n_ctx` from the `/slots` endpoint for each loaded model. On each request, it estimates token count from the input (character-based heuristic, not a tokenizer). If the estimate exceeds `n_ctx`, Wire returns a clear error with the limit and estimated count. No silent truncation.

**Value.** Silent truncation is the worst failure mode for agents and RAG pipelines — the model answers confidently with missing context. A clear rejection lets the caller retry with a shorter prompt or switch models.

**Constraints.** Estimation is approximate (no tokenizer in the gateway). Err on the side of permissive — reject only when the estimate clearly exceeds the limit (e.g., >90% of `n_ctx`). Must not add latency to requests that fit comfortably. Configurable per-model override for users who want to disable the guard.

---

### Request coalescing for embeddings

Deduplicate concurrent identical embedding requests — one GPU call, fan out response to all waiters.

**Outcome.** When multiple goroutines request the same embedding (same model + same input text), only one request reaches llama-server. All waiters receive the same response. Hit rate is visible in the monitor dashboard.

**Value.** RAG development loops re-embed the same text constantly. Agent retries hit the same prompt. Coalescing turns N identical GPU calls into 1 without caching state or invalidation complexity.

**Constraints.** Keyed by model name + SHA-256 of input. Only applies to `/v1/embeddings`. Time-bounded wait (if the first request takes too long, later arrivals fall through). Must not hold responses if the original request fails — propagate the error to all waiters.

---

### Request prioritization

Interactive chat gets immediate slot access; batch workloads (embeddings, rerank) queue behind.

**Outcome.** Requests are classified into priority tiers: chat completions (high), completions (medium), embeddings/rerank (low). When slots are saturated, low-priority requests wait in a bounded queue while high-priority requests proceed. Optional `X-Wire-Priority` header overrides the default tier.

**Value.** A user typing in a chat should never wait behind a batch embedding job. Without prioritization, first-come-first-served makes interactive latency unpredictable under load.

**Constraints.** Low-priority requests must eventually complete — no starvation. Queue depth is bounded; excess requests get 503 with `Retry-After`. Priority only affects queueing order, not slot affinity (a low-priority request with a cache-hot slot is still better than a high-priority cache miss).

---

### Thermal admission control

React to GPU thermal state by throttling admission when the hardware is throttling itself.

**Outcome.** Wire monitors GPU temperature and power draw (already read from sysfs). When temperature exceeds a configurable threshold or sustained power draw indicates clock throttling, Wire reduces effective concurrency: new requests receive `503 Retry-After` until the GPU recovers. Recovery is automatic — no operator intervention needed.

**Value.** GPUs under sustained load thermal-throttle silently — throughput drops but the queue keeps growing. Proactive admission control trades immediate throughput for steady sustained throughput. Especially important for always-on deployments without active monitoring.

**Constraints.** Thresholds must be configurable (GPUs vary). Default thresholds should be conservative (act only when throttling is clearly happening, not on normal temperature rise). AMD sysfs only for now. Must not trigger on brief temperature spikes during model loading.

---

## Future

Not committed. Possible directions if Wire's scope expands.

### llama-server deep integration

**LoRA adapter routing.** Expose llama-server's per-request LoRA scaling (`"lora": [{"id": 0, "scale": 0.5}]`). Wire manages which adapters are loaded per model instance and routes requests to instances with the right adapters. No extra VRAM per swap — it's a weight overlay. Useful for fine-tuned variants of the same base model without loading separate instances.

**Multimodal awareness.** Track which models have vision/audio projectors loaded (`--mmproj`). Route multimodal requests only to capable instances. Account for projector VRAM in scheduling. Detection via `/props` endpoint or config annotation.

**Dynamic model management via API.** llama-server exposes `/models/load` and `/models/unload`. Alternative to process-per-model: run fewer llama-server instances in router mode, manage models via HTTP API. Trades Wire's process-level isolation for less overhead. Revisit if llama-server's router mode matures — today it can't mix incompatible flags (embedding pooling vs. chat).

### Infrastructure

**Response caching.** LRU cache for deterministic requests (`temperature=0`). Keyed by model + request hash. Valuable for embeddings during RAG development and agent retry loops. Invalidate on model eviction/reload. Trade-off: memory usage vs. GPU savings.

**Config reload via SIGHUP.** Hot-reconfigure without restart. Add/remove model patterns, change warmup list, adjust thresholds. Drain removed models gracefully. Complexity: partial config changes that leave the system in an inconsistent intermediate state.

**Multi-GPU tensor split.** Expose llama-server's `--tensor-split` and `--split-mode` flags in config. Auto-detect multi-GPU topology from sysfs. Useful for models that exceed single-GPU VRAM.

**NVIDIA GPU support.** Read telemetry from nvidia-smi or NVML instead of AMD sysfs. Wire's GPU abstraction is already interface-shaped (discover + read metrics). Adding a second backend is straightforward but untested.

### Beyond local inference

**Token reduction.** When Wire routes to external providers (OpenAI, Anthropic), it sits in a unique position: it sees every request before it leaves the machine. A token reduction layer could compress prompts before forwarding — stripping redundant whitespace, deduplicating repeated system instructions across multi-turn conversations, summarizing long conversation histories via a local model before sending to an expensive API. The local model is free (already running); the external API bills per token. Even modest compression (20-30% on bloated agent prompts) pays for itself immediately. This only applies when Wire gains external provider routing — it's a natural complement to that feature.

**External provider routing.** Route requests to OpenAI, Anthropic, or other OpenAI-compatible APIs when no local model matches or when local capacity is exhausted. Transparent to the client — same endpoint, same format. Wire translates request/response formats where needed (Anthropic Messages API). Combined with token reduction, this makes Wire a cost-aware routing layer: use local GPU when available, fall back to cloud with minimized token spend.

**Federation.** Peer discovery and cross-node request routing. A Wire instance that can't serve a model forwards to one that can. Requires service discovery, health propagation, and split-brain handling. Large scope increase.
