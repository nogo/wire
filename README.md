# Wire

Single-port OpenAI-compatible inference gateway that owns the full lifecycle of llama-server processes. One binary, one config file, one port.

Applications talk OpenAI; Wire handles everything behind it — process supervision, model loading, slot routing, GPU awareness, live telemetry.

## Install

Requires [llama-server](https://github.com/ggml-org/llama.cpp) on `PATH`.

### From release

```sh
curl -fsSL https://raw.githubusercontent.com/nogo/wire/main/scripts/install.sh | bash
```

Installs the binary to `~/.local/bin/wire` and a systemd user service.

### From source

Requires [mise](https://mise.jdx.dev).

```sh
mise trust
mise run build
mise run install
```

## Configuration

Wire reads `~/.config/wire/config.yaml` (override with `WIRE_CONFIG`).

```yaml
listen: ":9480"
monitor: true
max_models: 3

models:
  "org/chat-model":
    context: 8192
    gpu_layers: 99
    slots: 4
  "org/embed-model":
    context: 2048
    gpu_layers: 99
    pooling: mean
  "org/*":
    context: 4096

warmup:
  - "org/chat-model"
```

### Fields

| Field | Default | Description |
|-------|---------|-------------|
| `listen` | `:9480` | Address to bind |
| `monitor` | `false` | Enable live dashboard via WebSocket |
| `max_models` | `1` | Maximum concurrent llama-server instances |
| `models` | — | Glob patterns mapping model names to parameters |
| `warmup` | — | Models to pre-load at startup |

### Model parameters

| Field | Default | Description |
|-------|---------|-------------|
| `context` | — | Context window size (`-c`) |
| `gpu_layers` | — | Layers to offload to GPU (`-ngl`) |
| `slots` | `1` | Parallel request slots (`--parallel`) |
| `pooling` | — | Pooling mode for embeddings (`mean`, `cls`, `last`) |
| `extra` | — | Additional llama-server flags |
| `spec_type` | — | Speculative decoding type (`ngram-simple`, `ngram-mod`) |
| `draft_max` | — | Max draft tokens (`--draft-max`) |
| `draft_min` | — | Min draft tokens (`--draft-min`) |

## Usage

```sh
wire up
```

### API endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/rerank`
- `GET  /v1/models`
- `GET  /health`
- `GET  /monitor` — live dashboard (requires `monitor: true`)
- `GET  /monitor/ws` — WebSocket stream of telemetry data

All endpoints accept and return OpenAI-compatible JSON. SSE streaming is supported for completions.

### Monitor

Enable the real-time dashboard by setting `monitor: true` in config. Open `http://localhost:9480/monitor` in a browser.

The dashboard streams live telemetry over WebSocket:

- Per-model status (loading, healthy, draining), request counts, token throughput
- Slot assignment and KV cache utilization
- GPU metrics (utilization, VRAM, temperature, power) via AMD sysfs
- llama-server events (prompt eval, generation eval, model load)

### Example

```sh
curl http://localhost:9480/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "org/chat-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Models load on first request if not in the warmup list. Concurrent requests for an unloaded model coalesce into a single load.

## systemd

The install script sets up a user service. Manage it with:

```sh
systemctl --user start wire
systemctl --user stop wire
systemctl --user status wire
journalctl --user -u wire -f
```

To start on boot without a login session:

```sh
loginctl enable-linger $USER
systemctl --user enable wire
```

## Docker

```sh
# CPU
docker build -f docker/Dockerfile -t wire:cpu .

# AMD (ROCm 7.1)
docker build -f docker/Dockerfile.rocm -t wire:rocm .

# NVIDIA (CUDA)
docker build -f docker/Dockerfile.cuda -t wire:cuda .
```

### Running

```sh
# CPU
docker run -v ~/.config/wire:/home/nonroot/.config/wire \
  -p 9480:9480 wire:cpu

# AMD GPU
docker run --device /dev/kfd --device /dev/dri \
  -v ~/.config/wire:/home/wire/.config/wire \
  -p 9480:9480 wire:rocm

# NVIDIA GPU
docker run --gpus all \
  -v ~/.config/wire:/home/wire/.config/wire \
  -p 9480:9480 wire:cuda
```

## Development

```sh
mise run test       # Run tests with race detector
mise run check      # Full verification: vet, fmt, tidy, test, lint
mise run dev        # Rebuild on source changes
```

## License

MIT
