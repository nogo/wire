# llama.cpp b9247 Integration

## Problem Statement

llama.cpp b9247 (May 2025) ships two features Wire should expose but currently cannot:

**Host-memory prompt caching** (`--cache-ram`, PR #16391): llama-server can store precomputed prompt KV state in RAM and hot-swap it back to the GPU when the same context is loaded again. Wire has no config field for this. Users who want it must put `--cache-ram` in `extra` with no guidance, and Wire's VRAM estimates do not account for the RAM headroom this requires.

**Slot KV cache persistence** (save/restore via `/slots/{id}?action=save|restore`): llama-server can dump the full KV state of a slot to disk and restore it on demand. Before b9247, the internal cell tracking (`llama_kv_cells_unified` refactor) was less reliable across multi-sequence scenarios, making save/restore fragile. With the new structure-of-arrays layout and corrected defragmentation logic in b9247, this is now stable enough to build Wire orchestration on top of. Wire has no orchestration for this today — every model eviction discards all slot KV state, and every reload pays full prompt evaluation from scratch.

For a multi-agent local inference setup (multiple small models, on-demand loading, LRU eviction), cold-start prompt evaluation on reload is the primary latency cost. A coder agent whose model gets evicted while idle and is then reloaded for the next task pays full prompt eval again. Slot persistence eliminates that cost at the price of disk I/O.

## Solution

Add `cache_ram: true` as a per-model config field that passes `--cache-ram` to llama-server at launch.

Add slot KV cache persistence orchestration to the process manager: before draining a model, Wire calls `POST /slots/{id}?action=save` for each loaded slot, writing state to a configured save path. When the same model is loaded again, Wire calls `POST /slots/{id}?action=restore` after the health check passes. Requests that land on a restored slot skip prompt evaluation entirely. Save/restore failures fall back to cold start without crashing.

## User Stories

1. As a user with a model that has a long stable system prompt, I want to set `cache_ram: true` so that the precomputed prompt KV is kept in RAM across context switches, reducing GPU work on repeated loads.

2. As a user, I want `cache_ram` to default to false so that existing configs are unaffected and RAM is not consumed unless explicitly requested.

3. As a user, I want Wire to pass `--cache-ram` to llama-server only when `cache_ram: true` is set, so that the flag is not applied globally to all models.

4. As a user with multiple agents each using a different model, I want slot KV state saved before a model is evicted so that when the model is reloaded for the next agent request, prompt evaluation is skipped.

5. As a user, I want slot saves to happen within the existing drain timeout budget so that saves do not delay shutdown beyond what I have configured.

6. As a user, I want Wire to fall back to a cold start if a slot save or restore fails, rather than refusing to load the model, so that a disk error does not block inference.

7. As a user, I want stale slot cache files from a previous model version to be detected and ignored, so that a model update does not cause incorrect KV state to be restored.

8. As a user, I want to configure `slot_save_path` in my Wire config to point to fast NVMe storage, so that disk I/O during save/restore does not become the bottleneck.

9. As a user who has not set `slot_save_path`, I want slot persistence to be disabled automatically rather than falling back to a slow default path, so that I do not get unexpected disk writes.

10. As a user, I want Wire to log which slots were saved before eviction and which were restored after load, so that I can verify persistence is working without inspecting files manually.

11. As a user, I want the restore step to complete before Wire marks a model as healthy and begins routing requests to it, so that the first request after reload always hits a warm slot.

12. As a user with a multi-slot model, I want Wire to save and restore all slots concurrently (not sequentially) so that the save/restore overhead scales with slot count as little as possible.

13. As a user, I want slot cache files to be scoped per model name so that two models with different names never share or overwrite each other's saved state.

14. As a user, I want Wire to clean up slot cache files when a model is removed from config or explicitly drained without intent to reload, so that stale files do not accumulate on disk.

15. As a user, I want the real-time monitor dashboard to show whether each loaded model's slots have saved state available, so that I can see at a glance which models benefit from persistence.

## Implementation Decisions

### Modified module: Config

- Add `CacheRAM bool` to `ModelParams` (`cache_ram` in YAML). Passed to `buildArgs` as `--cache-ram` when true. No VRAM estimate impact — RAM is a separate budget.
- Add `SlotSavePath string` to `Config` (global, not per-model). Empty = slot persistence disabled. Wire passes `--slot-save-path <path>` to every llama-server launch when set. Path is validated to be writable at startup.

### Modified module: Process Manager

**Save on drain**: before cancelling a model's context, Wire iterates its slots (obtained from the last `/slots` poll via the Hub), calls `POST /slots/{id}?action=save` on each concurrently, waits for all saves to complete or for the drain timeout to elapse (saves share the same timeout budget as in-flight request draining). Slots that fail to save are logged and skipped.

**Restore after load**: after `waitHealthy` succeeds, if `SlotSavePath` is set and saved files exist for this model, Wire calls `POST /slots/{id}?action=restore` for each slot concurrently. The model is not marked `stateLoaded` until all restores complete (or fail). Failed restores are logged; the slot proceeds cold.

**Staleness detection**: each save file is named `<model-name>-slot-<id>.bin` and accompanied by a sidecar `<model-name>-slot-<id>.meta` containing the model name and a hash of the model params at save time. On restore, Wire reads the sidecar and skips the restore if params have changed (context size, GPU layers, or any field that changes the KV shape).

**Cleanup**: when a model is removed from config or drained via `Reload`, Wire deletes the corresponding save files after the process exits.

### Modified module: Hub / Monitor

Adds a `HasSavedSlots bool` field to `BackendMetrics` (or the slot struct). Populated by checking for the existence of save files for each loaded model. Surfaced in the dashboard.

### Modified module: Gateway / buildArgs

`buildArgs` adds `--cache-ram` when `CacheRAM` is true. Adds `--slot-save-path <path>` when `SlotSavePath` is set in global config. No other changes.

### llama-server API used

- `GET /slots` — already polled by Hub; used to enumerate slot IDs before save.
- `POST /slots/{id}?action=save` — triggers save to `--slot-save-path`.
- `POST /slots/{id}?action=restore` — triggers restore from `--slot-save-path`.

No new llama-server versions or flags beyond what shipped in the b9247 range are required.

### OpenAI API Contract

No changes. Save/restore is entirely internal to the load/drain lifecycle. Clients observe only that the first request after a model reload is faster (no prompt eval latency).

## Testing Decisions

- **Test save-on-drain**: fake process builder; fake HTTP server mimicking `/slots` and `POST /slots/{id}?action=save`; verify Wire calls save for each slot before SIGTERM; verify saves within timeout; verify timeout expiry proceeds to SIGTERM without blocking.
- **Test restore-after-load**: fake save files on disk; fake `/slots/{id}?action=restore` endpoint; verify Wire calls restore after health check; verify model reaches `stateLoaded` only after restore completes; verify failed restore still reaches `stateLoaded` (cold start fallback).
- **Test staleness detection**: write a sidecar with mismatched params hash; verify Wire skips restore and logs a warning.
- **Test cleanup**: verify save files are deleted after model removal via `Reload`.
- **Test `cache_ram` flag**: verify `--cache-ram` appears in args when set; verify absent when not set.
- **Prior art**: `process/manager_test.go` drain and eviction tests for the save-on-drain flow; `monitor/hub_test.go` for BackendMetrics assertions.
- Tests use temp directories for `SlotSavePath`; no writes to real home directory paths.

## Out of Scope

- **Cross-process KV sharing** (RDMA, CXL, shared memory): not supported by llama.cpp; not in scope.
- **Automatic `slot_save_path` selection**: Wire does not auto-detect NVMe vs spinning disk. The user configures the path explicitly.
- **KV quantization** (`kv_type`): covered in the model-discovery PRD.
- **MLA KV cache fix for DeepSeek models**: internal to llama.cpp; Wire has no action item.
- **Partial slot restore** (restoring a subset of slots): all-or-nothing per model.

## Further Notes

- The `llama_kv_cells_unified` refactor in b9247 corrects defragmentation and shift logic for multi-sequence KV cells. This is the reason slot save/restore is now reliable enough to build on — the binary format written by `action=save` reflects correct cell state. This PRD depends on llama-server being at b9247 or later.
- Save files can be large: a 7B model with 8k context and 4 slots at f16 KV is roughly `7B params * 2 layers * 2 (K+V) * 2 bytes * 8192 tokens / num_layers` — order of hundreds of MB per slot. NVMe is strongly recommended. This should be documented in the config reference.
- The sidecar metadata approach for staleness detection is intentionally simple. A more robust approach (embedding a model hash in the GGUF) is out of scope.
