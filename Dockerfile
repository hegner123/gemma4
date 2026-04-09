FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Reset the entrypoint so we control startup from handler.py
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

RUN pip install --no-cache-dir --break-system-packages \
        runpod requests huggingface-hub

COPY handler.py /app/handler.py

# Model config (override via RunPod environment variables)
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV MODEL_REPO=ggml-org/gemma-4-26B-A4B-it-GGUF
ENV MODEL_FILE=gemma-4-26B-A4B-it-Q4_K_M.gguf
ENV MODEL_DIR=/runpod-volume/models
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV PARALLEL=1
ENV LLAMA_PORT=8080

CMD ["python3", "/app/handler.py"]
