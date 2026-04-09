# ---------------------------------------------------------------
# Stage 1: Build llama.cpp with CUDA support
# ---------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ARG LLAMA_CPP_REF=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
        https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && cmake --install build --prefix /opt/llama

# ---------------------------------------------------------------
# Stage 2: Runtime (no model baked in, downloads at boot)
# ---------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary and libraries
COPY --from=builder /opt/llama /opt/llama
ENV PATH="/opt/llama/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/llama/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN pip install --no-cache-dir --break-system-packages \
        runpod requests huggingface-hub

COPY handler.py /app/handler.py

# Model config (override via RunPod environment variables)
# MODEL_DIR should point to a RunPod network volume for caching
ENV MODEL_REPO=ggml-org/gemma-4-26B-A4B-it-GGUF
ENV MODEL_FILE=gemma-4-26B-A4B-it-Q4_K_M.gguf
ENV MODEL_DIR=/runpod-volume/models
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV PARALLEL=1
ENV LLAMA_PORT=8080

CMD ["python3", "/app/handler.py"]
