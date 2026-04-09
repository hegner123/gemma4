# gemma4

RunPod serverless harness for serving Google's Gemma 4 26B-A4B-it (MoE, 25B params, ~4B active) via llama.cpp. Builds a lightweight Docker image with llama-server, downloads the GGUF model at first boot to a RunPod network volume, and exposes an OpenAI-compatible inference API.

## Features

- Multi-stage Docker build compiles llama.cpp with CUDA, producing a ~2-3 GB image (no model baked in)
- Automatic model download from HuggingFace on first cold start, cached on network volume for subsequent boots
- OpenAI-compatible `/v1/chat/completions` API via llama-server
- Streaming and synchronous response modes
- Configurable quantization (Q4_K_M default, Q8_0, F16), context size, GPU layer offload, and parallelism via environment variables
- GitHub Actions workflow for automated image builds pushed to GHCR
- Python client with sync, async, and streaming modes

## Prerequisites

- A RunPod account with serverless access
- A RunPod network volume (same region as your endpoint) for model caching
- A GPU with 24 GB+ VRAM for Q4_K_M (48 GB for Q8_0, 80 GB for F16)

## Installation

GitHub Actions builds and pushes the image automatically on push to `main`. The image lands at `ghcr.io/<your-user>/gemma4:latest`.

To deploy on RunPod:

1. Push this repo to GitHub. The action builds the Docker image.
2. Make the GHCR package public (GitHub Settings > Packages), or add Docker credentials in RunPod.
3. Create a **Network Volume** in RunPod (same region as your endpoint).
4. Create a **Serverless Endpoint**:
   - Container Image: `ghcr.io/<your-user>/gemma4:latest`
   - Network Volume: attach at `/runpod-volume`
   - GPU: 24 GB+ (L4, RTX 4090, A5000)
   - Environment variables: defaults work out of the box

## Usage

### Send requests via the RunPod API

```bash
curl -s https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Explain quantum entanglement briefly."}],
      "max_tokens": 256,
      "temperature": 0.7
    }
  }'
```

### Python client

```bash
export RUNPOD_API_KEY=your_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id

python client.py "What causes the northern lights?"
python client.py --stream "Write a haiku about programming."
```

## Configuration

All settings are configured via environment variables on the RunPod endpoint.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `ggml-org/gemma-4-26B-A4B-it-GGUF` | HuggingFace repo ID |
| `MODEL_FILE` | `gemma-4-26B-A4B-it-Q4_K_M.gguf` | GGUF filename to download |
| `MODEL_DIR` | `/runpod-volume/models` | Local directory for model cache |
| `N_GPU_LAYERS` | `-1` | GPU layers to offload (-1 = all) |
| `CTX_SIZE` | `8192` | Context window size |
| `PARALLEL` | `1` | Concurrent request slots |

## License

MIT. See [LICENSE](LICENSE).
