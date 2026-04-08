# --- llama.cpp build stage ---
FROM docker.io/library/ubuntu:24.04 AS llama-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_VERSION=master
RUN git clone --depth 1 --branch "${LLAMA_CPP_VERSION}" \
        https://github.com/ggml-org/llama.cpp.git /src/llama.cpp

WORKDIR /src/llama.cpp
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DGGML_CPU=ON \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && strip build/bin/llama-server

# --- wire build stage ---
FROM docker.io/library/ubuntu:24.04 AS wire-build

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
FROM docker.io/library/ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libcurl4 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /usr/sbin/nologin -d /nonexistent wire

COPY --from=llama-build /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=wire-build /src/wire/bin/wire /usr/local/bin/wire

USER wire
EXPOSE 8080

ENTRYPOINT ["wire"]
CMD ["up"]
