# Model Discovery and Configuration Simplification

## Problem Statement

Wire currently loads all models exclusively via `-hf <name>`, which requires models to be fetched from HuggingFace Hub. Users who have downloaded GGUF files to a local directory (e.g. `~/models`) have no way to point Wire at those files — every load goes through HF, even when the file is already on disk in a non-HF layout.

Additionally, HF cache scanning is duplicated in two separate places (gateway and VRAM tracker), scanning the same filesystem tree independently with different TTLs and different data shapes. Neither scanner feeds the other.

Finally, `ModelParams` carries five speculative decoding fields (`spec_type`, `draft_max`, `draft_min`, `spec_ngram_n`, `spec_ngram_m`) that Wire has no scheduling logic for — they are pure passthrough to llama-server. Keeping them as first-class fields creates false API surface and bloats `Equal` comparisons. Meanwhile, one parameter Wire genuinely needs to understand — KV cache type (for VRAM estimation) — is missing entirely.

## Solution

Introduce a `model_dirs` config field: an ordered list of local directories Wire scans for `*.gguf` files. Wire builds a shared model index from those directories plus the existing HF cache, derives a model name from each discovered file (filename without extension), and uses that index to resolve `-m <path>` vs `-hf <name>` at load time. Glob patterns in the `models` config remain the parameter-matching layer — they are unaffected by discovery.

Unify the two HF cache scanners into one module that feeds both the gateway's `/v1/models` listing and the VRAM tracker's file-size estimates.

Simplify `ModelParams` by removing passthrough-only spec fields (they move to `extra`) and adding `kv_type` (Wire reads it for VRAM estimation and passes it to llama-server).

## User Stories

1. As a user with models in `~/models`, I want to set `model_dirs: [~/models]` in my config so that Wire discovers and serves those files without HF network access.

2. As a user, I want model names to be derived automatically from filenames (strip `.gguf`) so that I do not need to manually define a name-to-path mapping for every file.

3. As a user, I want Wire to scan multiple directories so that I can organise models across several locations (e.g. `~/models`, `~/models/coding`, a secondary drive) and have all of them available through a single endpoint.

4. As a user, I want glob patterns in the `models` config to match discovered model names so that I can apply parameter overrides by pattern without knowing the exact filename at config write time.

5. As a user, I want `GET /v1/models` to list all locally discovered models alongside HF cache and loaded models, so that I can see everything available from a single API call.

6. As a user, I want HF cache auto-discovery to continue working without any config change, so that models previously loaded via `-hf` remain discoverable by default.

7. As a user, I want Wire to fall back to `-hf <name>` when a requested model name is not found in any local directory or HF cache, so that on-demand HF downloads still work as before.

8. As a user, I want `tilde` (`~`) expansion in `model_dirs` paths so that I can write `~/models` without needing to know the absolute home directory.

9. As a user, I want the model index to refresh periodically (short TTL) so that newly downloaded files are picked up without restarting Wire.

10. As a user, I want to configure `kv_type: q8_0` (or `q4_0`) on a model pattern so that Wire passes the correct `--cache-type-k` and `--cache-type-v` flags to llama-server and uses the reduced footprint in VRAM estimates.

11. As a user, I want `kv_type` to default to the llama-server default (f16) when unset, so that existing configs are unaffected.

12. As a user migrating from speculative decoding config fields (`spec_type`, `draft_max`, etc.), I want Wire to accept those parameters via `extra` so that I can continue using them without losing functionality.

15. As a user, I want `extra` to remain a catch-all for any llama-server flag not explicitly modelled in `ModelParams`, so that new llama-server flags are always accessible without waiting for a Wire release.

16. As a user, I want the VRAM tracker's file-size estimation to use the same discovered path as model loading, so that estimates are based on the actual file Wire will load rather than a separately-resolved path.

17. As a user, I want Wire to log at startup which models were discovered from each `model_dirs` entry, so that I can verify my directory configuration without inspecting files manually.

18. As a user, I want Wire to warn at startup if a `model_dirs` entry does not exist or is not readable, rather than silently ignoring it, so that typos in paths surface immediately.

19. As a user, I want the model name shown in `/v1/models` for a locally discovered file to exactly match the name I must use in API requests, so that there is no guessing or case-normalisation surprises.

20. As a user with duplicate model names across directories (same filename in two dirs), I want Wire to use the first match in `model_dirs` order and log a warning about the duplicate, so that precedence is predictable.

## Implementation Decisions

### New module: Model Index

A single module responsible for building and refreshing the complete model-name-to-source map. It merges two sources:

- **Local scan**: walks each directory in `model_dirs`, collects `*.gguf` files, derives name = filename without extension, stores absolute path.
- **HF cache scan**: existing logic (currently duplicated in gateway and VRAM tracker) — walks `~/.cache/huggingface/hub` and derives model IDs from the `models--org--name` directory naming convention.

Interface: `Lookup(name string) (path string, source Source, ok bool)` where `Source` is `Local` or `HFCache`. Returns path for local files; returns empty path for HF cache entries (caller uses `-hf name`). `List() []ModelEntry` for `/v1/models`.

The index is TTL-cached (e.g. 30 seconds). It is shared by the gateway (for listing and routing), the process manager (for resolving load flags), and the VRAM tracker (for file-size estimation). This eliminates the current duplication.

### Modified module: Config

- Add `ModelDirs []string` to `Config`. Empty by default; HF cache is always included in discovery regardless.
- Remove `HFCacheDir` as a top-level field — it becomes an internal default resolved from `$HOME`. The user no longer needs to set it. If a power user needs to override it, it can remain as an undocumented field.
- Remove `SpecType`, `DraftMax`, `DraftMin`, `SpecNgramN`, `SpecNgramM` from `ModelParams`.
- Add `KVType string` to `ModelParams` (values: `""`, `"q8_0"`, `"q4_0"`, `"q5_0"`, `"q5_1"`). Wire validates the value against known types at config load.
- `Equal` simplifies accordingly.
- Validation: warn (not error) on unreadable `model_dirs` entries; error on unknown `kv_type` value.

### Modified module: Process Manager (`buildArgs`)

`buildArgs` receives a resolved path alongside model name and params. Resolution order: local index → HF cache entry → raw name (HF download). If path is non-empty: use `-m <path>`. Otherwise: use `-hf <name>`. Adds `--cache-type-k` and `--cache-type-v` flags when `KVType` is set. Removes five spec argument branches (they arrive via `Extra` now).

### Modified module: Gateway

Replaces the local `hfCache` struct with the shared model index for `/v1/models` listing. Resolves model name to source before enqueuing. Passes resolved path through the queue to the process manager.

### Modified module: VRAM Tracker

Replaces its own HF cache scan with the shared model index for file-size estimation. When a model is local, file size is from the discovered path. When HF cache, existing logic applies.

### Config example

```
model_dirs:
  - ~/models
  - ~/models/coding

models:
  "*7b*instruct*":
    gpu_layers: 35
    context: 8192
    kv_type: q8_0

  "*embed*":
    pooling: mean
    context: 2048
    kv_type: q8_0

  "*coder*":
    context: 16384
    extra: ["--spec-type", "ngram-simple"]
```

### OpenAI API Contract

No changes. `/v1/models` continues to return the same shape. Locally discovered models appear with their derived names. Clients use those exact strings in `model` fields.

## Testing Decisions

- **Test the model index in isolation**: mock filesystem (temp dirs with `.gguf` files), verify name derivation, duplicate handling, TTL refresh, missing directory warning, `Lookup` resolution order.
- **Test `buildArgs` for path resolution**: verify `-m path` when local, `-hf name` when HF, correct `--cache-type-k`/`--cache-type-v` flags for each `kv_type` value, spec flags via `Extra` passthrough.
- **Test config validation**: unknown `kv_type` values error; unreadable `model_dirs` entries warn; removed spec fields in config produce unknown-field error (yaml decoder uses `KnownFields(true)`).
- **Prior art**: `vram/hf_cache_test.go` for filesystem scan patterns, `process/manager_test.go` for eviction candidate logic, `config/config_test.go` for validation error cases.
- Tests must not perform real filesystem discovery against the user's home directory — all paths injected via config in tests.

## Out of Scope

- **Name aliases**: mapping a short user-defined name to a discovered filename. Clients use the derived filename as-is.
- **Subdirectory recursion**: `model_dirs` entries are non-recursive. Files must be directly in the listed directory.
- **Multi-file models** (split GGUFs): only single-file `.gguf` models are discovered.
- **NVIDIA NVML backend**: unrelated to discovery.
- **Slot KV cache persistence**: separate roadmap item.
- **Remote provider config changes**: `Provider`, `UpstreamModel`, `Capabilities` are unchanged.
- **Hot-reload of `model_dirs`**: the index refreshes on TTL but changing `model_dirs` requires restart.

## Further Notes

- The name derivation rule (strip `.gguf`, keep original casing) is intentionally simple. If downstream tools lowercase model names, users should name their files accordingly. Wire does not normalise.
- The shared model index is the keystone of this work. Once it exists, the VRAM tracker's file-size estimates become more accurate (real path, real file size) and the gateway listing becomes consistent with what can actually be loaded.
- The removal of five `ModelParams` fields is a breaking config change. Users with existing configs using those fields will get an unknown-field error at startup (yaml decoder is strict). The migration path is moving those values into `extra`. This should be clearly communicated in the release notes.
