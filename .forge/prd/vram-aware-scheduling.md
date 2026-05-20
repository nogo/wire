# VRAM-Aware Scheduling and Request Queue

## Problem Statement

Loading multiple llama-server models concurrently on a GPU with limited VRAM causes system-wide OOM kills that crash the entire desktop session (Wayland compositor, etc.). The current count-based `max_models` capacity treats a 1B model and a 70B model identically, making it impossible to safely run mixed workloads. On CPU-only servers, there is no memory guard at all — the OOM killer can terminate critical system processes instead of the inference child.

Additionally, Wire has no request queueing. When all slots are saturated, requests fail immediately with no buffering, and there is no mechanism to batch or coalesce concurrent requests for unloaded models.

## Solution

Replace count-based capacity with a VRAM-aware admission control system that estimates memory consumption before loading models, evicts idle models proactively when capacity is tight, and drains in-flight requests gracefully. Add a per-model priority queue with time-bounded waits so requests are buffered rather than rejected when slots are saturated. Protect the host system with OOMScoreAdjust on every llama-server child process.

## User Stories

1. As a user with a 24GB GPU, I want to run a 7B chat model and a 3B reranker simultaneously without crashing my desktop session, so that I can use both models in a single RAG pipeline.

2. As a user, I want Wire to automatically detect my GPU vendor (AMD or NVIDIA) and track VRAM usage without manual configuration, so that it works out of the box on any machine.

3. As a user with a CPU-only remote server, I want inference to work without VRAM tracking but still protected from OOM kills that could destabilize the system, so that I can run Wire on cheap cloud instances.

4. As a user, I want my warmup models to go through the same admission checks as regular requests, so that my startup config doesn't bypass safety guards.

5. As a user, when a new model request arrives and VRAM is full, I want the least-recently-used idle models to be evicted automatically to make room, so that I don't have to manually manage which models are loaded.

6. As a user, when a model is evicted, I want in-flight requests to be given a grace period to finish (default 30s) before the process is killed, so that I don't lose responses that are almost complete.

7. As a user, when a queued request's model gets evicted before it can be served, I want to get a 503 response immediately with a Retry-After header, so that my client can decide whether to retry or fail gracefully.

8. As a user with multiple concurrent requests for a model that isn't loaded, I want them all to share a single model load operation, so that I don't trigger redundant downloads or duplicate VRAM allocation.

9. As a user, I want interactive chat requests to take priority over batch embedding workloads when queueing, so that a user typing in a chat doesn't wait behind a long embedding job.

10. As a user, I want queued requests to have a time budget (30s for chat, 15s for completions, 5s for embeddings/rerank) after which they get rejected, so that no request waits indefinitely.

11. As a user, I want the VRAM estimate for a model to be derived automatically from the GGUF file size plus a KV cache heuristic, so that I don't need to manually configure memory budgets per model.

12. As a user with models in the HuggingFace cache, I want Wire to parse their headers at startup so that VRAM estimates are available before any request arrives.

13. As a user, when a model is downloaded on-demand via `-hf` (not in local cache), I want Wire to use a conservative placeholder VRAM estimate until the actual allocation is reported, so that admission decisions are safe even for unknown models.

14. As a user, I want the real-time monitor dashboard to continue showing GPU metrics (VRAM, temperature, power), so that I can see what's happening on my GPU alongside the new scheduling behavior.

15. As a user with a config that lists warmup models whose combined VRAM exceeds available memory, I want Wire to attempt loading them in config order and log a warning if some can't fit, so that I'm informed but the system still starts.

16. As a user, I want the eviction algorithm to prefer evicting the minimum number of models that frees enough VRAM for the incoming model, so that I minimize disruption to running workloads.

17. As a user, when I send a request for a model that's already loaded but all slots are busy, I want my request to queue and wait for a slot to free up, so that I don't get an immediate rejection.

18. As a developer deploying Wire in a container, I want the GPU auto-detection to gracefully fall back to a no-op backend when GPU sysfs is inaccessible, so that Wire still runs (without VRAM scheduling) in restricted environments.

19. As a user, when the system recovers from a GPU OOM kill, I want the VRAM tracker to reconcile with actual sysfs readings rather than relying on potentially missed events, so that the admission budget reflects reality.

20. As a user, I want the OOM killer to always target llama-server children before system-critical processes, so that even if VRAM estimation is wrong, the worst outcome is a model crash, not a session crash.

## Implementation Decisions

### New Module: `internal/vram/` (VRAM Tracker)

- **Deep module** — encapsulates GPU discovery, backend abstraction, VRAM budget tracking, and admission decisions behind a simple interface.
- Interface: `Discover()`, `CanFit(estimate uint64) bool`, `Reserve(model, estimate)`, `Release(model)`, `PushGPUState()` (broadcasts to Hub).
- GPU backend auto-detection chain: AMD sysfs → NVIDIA NVML → noop (CPU-only). Config override available but not required.
- VRAM estimation: `GGUF file size × 1.3` heuristic (weights + 30% KV cache overhead). Single `os.Stat` call per model. For cache-miss models (downloaded on-demand), use conservative placeholder estimate based on model name pattern, reconciled with actual sysfs on next poll.
- VRAM tracker subscribes to Hub's `MsgGPUTick` for GPU metrics (Hub remains the sole GPU poller). VRAM tracker reconciles reserved vs. actual VRAM from sysfs readings.
- On model drain, reservations are marked "releasing" until the next sysfs poll confirms actual VRAM drop. Admission uses `total - (actual_sysfs_used + pending_releases)` as the available budget.
- On CPU-only machines, the tracker is a no-op — `CanFit` always returns true. `OOMScoreAdjust` is the only guard.

### New Module: `internal/gateway/queue/` (Request Queue)

- **Deep module** — per-model priority queue with time-bounded waits and coalesced loading.
- Interface: `Enqueue(model, priority, ctx) -> Ticket`, `Ticket.Ready() <-chan string`, `Ticket.Release()`, `Ticket.Err() error`.
- Ticket type returned immediately (async). Caller waits on `Ready` channel. On model load success, channel receives the backend address. On failure or timeout, channel is closed and `Err` is set.
- Priority tiers: chat completions (high, 30s budget), completions (medium, 15s budget), embeddings/rerank (low, 5s budget).
- Global queue depth cap: 200 requests total across all models.
- Coalesced loading: concurrent requests for the same unloaded model share one `LoadTicket`. First request triggers eviction + load, subsequent requests attach to the same ticket.
- On model eviction, all queued requests for that model are cancelled with `503 Retry-After`.

### Modified Module: `internal/process/` (Process Manager)

- `LoadOrGet` returns a `LoadTicket` instead of `(addr, release, err)` synchronously.
- Eviction logic changes from single-model count-based to cascading LRU: find minimum set of idle models whose combined VRAM frees enough for the incoming model, then drain all simultaneously.
- Drain timeout: configurable, default 30s. New requests to a draining model are rejected immediately. In-flight requests get 30s to finish, then SIGTERM + 10s `WaitDelay`.
- `OOMScoreAdjust=1000` set on every llama-server child process via `exec.Cmd.SysProcAttr`.
- Warmup models go through the same `LoadOrGet` admission path as regular requests (no special privileges). Config order, bounded retry (3 attempts over 5 minutes) if VRAM insufficient.

### Modified Module: `internal/gateway/` (Gateway)

- `handleRoute` changes to use ticket-based async model loading. The HTTP handler creates a ticket via the queue, waits on `Ready` (or context cancellation), then proxies.
- Context guard (token estimation vs. n_ctx) remains — applied after the model is loaded but before proxying.
- Remote provider handling is unchanged (bypasses process manager entirely).

### Modified Module: `internal/monitor/` (Hub)

- Hub's GPU polling code moves to the VRAM tracker. Hub receives GPU state via `PushGPUState()` from the VRAM tracker.
- `MsgGPUTick` messages continue to be broadcast to subscribers — sourced from VRAM tracker instead of polled directly.
- Hub's other polling (backend `/metrics`, `/slots`, `/props`, remote stats, model discovery) is unchanged.

### Modified Module: `internal/config/` (Config)

- New config fields: `drain_timeout` (duration, default 30s), `queue_depth` (int, default 200).
- `warmup` list validated to not exceed VRAM capacity (warning if exceeded, not error — system still starts).

### Lifecycle Ordering in `cli.go`

1. Config load
2. VRAM tracker init (GPU auto-detect, HF cache scan)
3. Process manager init
4. Hub init (subscribes to VRAM tracker for GPU state)
5. Gateway init (uses VRAM tracker + queue)
6. Warmup models submitted through gateway's admission path
7. Gateway listen + config watch loop

### OpenAI API Contract

- No changes. All queuing, ticketing, and admission control is internal. Clients see the same OpenAI-compatible endpoints.
- New behavior: `503 Retry-After: <seconds>` responses when queue is full or model is draining.
- Streaming responses (SSE) are unaffected — the handler holds the connection open while waiting in queue, then streams normally.

## Testing Decisions

- **Test VRAM tracker in isolation** — mock GPU backends, verify admission decisions, test cascading eviction logic, test reservation reconciliation with sysfs deltas.
- **Test request queue in isolation** — verify priority ordering, time budget expiration, coalesced loading (multiple waiters on one ticket), drain cancellation, queue depth cap.
- **Test OOMScoreAdjust integration** — verify `SysProcAttr` is set on the command builder.
- **Test GGUF estimation** — verify file size + heuristic produces reasonable estimates for known model sizes.
- **Integration test** — submit concurrent requests for unloaded models, verify only one process starts, verify queue ordering, verify drain + re-queue flow.
- **Prior art**: `process/manager_test.go` already tests concurrent `LoadOrGet` coalescing, eviction, and crash restart. `gateway/gateway_test.go` already tests context guard and remote proxying. `monitor/hub_test.go` tests subscriber fan-out. The new modules follow the same pattern: faked backends, concurrent goroutine tests with `sync.WaitGroup`, context cancellation tests.
- **Tests must not disable safety checks** — OOMScoreAdjust must be set in all tests, even with faked process builders. Fakes must mirror real error contracts.

## Out of Scope

- **NVIDIA NVML backend implementation** — the interface is defined, but the NVIDIA backend will be implemented in a subsequent work unit. AMD sysfs and noop backends are in scope.
- **GGUF header metadata parsing** — the file size + heuristic approach is sufficient for admission control. Deep header parsing (architecture-specific layer counts, embedding dimensions) is deferred.
- **Slot KV cache persistence** — save/restore of llama-server slot state to disk is a separate roadmap item.
- **Thermal admission control** — GPU temperature/power throttling decisions are deferred.
- **Request coalescing for embeddings** — deduplication of identical embedding requests is a separate roadmap item.
- **Config hot-reload for VRAM settings** — `drain_timeout` and `queue_depth` can be changed at config reload, but the VRAM tracker does not support changing GPU backend at runtime.
- **Multi-GPU tensor splitting** — models split across multiple GPUs are not explicitly handled. The VRAM tracker uses total VRAM across all detected GPUs.
- **cgroup memory limits** — out of scope. `OOMScoreAdjust` is the sole safety net.

## Further Notes

- The VRAM tracker is the keystone module. Once it's implemented, speculative decoding orchestration (draft model VRAM budgeting), thermal admission control, and slot KV cache persistence can all build on it.
- The ticket-based async `LoadOrGet` is the biggest behavioral change in the process manager. All existing tests that call `LoadOrGet` directly will need to be updated to use the ticket API.
- The 1.3 heuristic multiplier (file size + 30% KV cache) was chosen based on typical llama.cpp memory profiles. It's intentionally conservative — better to reject a load that could have fit than to admit one that OOMs. The actual VRAM consumption is reconciled with sysfs on every poll tick.
- The queue depth cap (200) and time budgets (30s/15s/5s) are defaults that can be tuned per deployment. They are exposed as config fields for operator control.
- On CPU-only machines, the system behaves identically to the current implementation except for `OOMScoreAdjust`. No queuing admission gate is applied — requests queue only based on slot availability, not memory.
