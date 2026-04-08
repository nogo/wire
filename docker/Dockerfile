# --- llama-server (official CPU build) ---
FROM ghcr.io/ggml-org/llama.cpp:server AS llama-server

# --- wire build ---
FROM docker.io/library/debian:trixie-slim AS wire-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN curl -fsSL https://mise.jdx.dev/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /src/wire
COPY mise.toml go.mod go.sum ./
RUN mise install && mise run -- go mod download
COPY . .
RUN CGO_ENABLED=0 mise run build && strip bin/wire

# --- runtime ---
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=llama-server /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-server /usr/lib/ /usr/lib/
COPY --from=wire-build /src/wire/bin/wire /usr/local/bin/wire
EXPOSE 9480

ENTRYPOINT ["wire"]
CMD ["up"]
