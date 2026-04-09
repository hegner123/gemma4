"""
RunPod Serverless Handler for Gemma 4 26B-A4B-it via llama.cpp

Downloads the GGUF model on first boot (cached on network volume),
starts llama-server, and proxies OpenAI-compatible requests.
"""

import json
import os
import subprocess
import time

import requests
import runpod
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "ggml-org/gemma-4-26B-A4B-it-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "gemma-4-26B-A4B-it-Q4_K_M.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
PARALLEL = int(os.environ.get("PARALLEL", "1"))

BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"


def ensure_model() -> str:
    """Download the GGUF if not already present. Returns the local path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, MODEL_FILE)

    if os.path.isfile(local_path):
        size_gb = os.path.getsize(local_path) / (1024 ** 3)
        print(f"Model already cached: {local_path} ({size_gb:.1f} GB)")
        return local_path

    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} to {MODEL_DIR}...")
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
    )
    size_gb = os.path.getsize(downloaded) / (1024 ** 3)
    print(f"Download complete: {downloaded} ({size_gb:.1f} GB)")
    return downloaded


def start_llama_server(model_path: str):
    """Launch llama-server and block until it reports healthy."""
    cmd = [
        "llama-server",
        "-m", model_path,
        "--port", str(LLAMA_PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(CTX_SIZE),
        "--parallel", str(PARALLEL),
        "--host", "127.0.0.1",
    ]

    print(f"Starting llama-server: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode}:\n{stdout}"
            )

        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                body = r.json()
                if body.get("status") == "ok":
                    print("llama-server is ready")
                    return proc
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(1)

    proc.kill()
    raise RuntimeError("llama-server failed to become healthy within 300s")


print("Ensuring model is available...")
model_path = ensure_model()

print("Initializing llama-server...")
server_proc = start_llama_server(model_path)


def handler(job):
    """
    Process a single inference request.

    Expected input format (OpenAI-compatible):
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": false,
        "endpoint": "/v1/chat/completions"  // optional override
    }

    The "endpoint" field is stripped before forwarding.
    Supported endpoints:
        /v1/chat/completions  (default)
        /v1/completions
        /v1/embeddings
    """
    job_input = job["input"]

    endpoint = job_input.pop("endpoint", "/v1/chat/completions")
    stream = job_input.get("stream", False)

    try:
        resp = requests.post(
            f"{BASE_URL}{endpoint}",
            json=job_input,
            stream=stream,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}

    if not stream:
        return resp.json()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            yield chunk
        except json.JSONDecodeError:
            continue


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
