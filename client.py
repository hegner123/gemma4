"""
Client for calling the Gemma 4 RunPod serverless endpoint.

Usage:
    export RUNPOD_API_KEY=your_key
    export RUNPOD_ENDPOINT_ID=your_endpoint_id
    python client.py "What is the meaning of life?"
    python client.py --stream "Tell me a story."
"""

import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def run_sync(prompt, max_tokens=512, temperature=0.7):
    """Blocking call. Returns the full response when done."""
    payload = {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    }

    resp = requests.post(f"{BASE}/runsync", json=payload, headers=headers(), timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "FAILED":
        print(f"Job failed: {data}", file=sys.stderr)
        sys.exit(1)

    return data.get("output", data)


def run_async(prompt, max_tokens=512, temperature=0.7):
    """Submit job, poll for result."""
    payload = {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    }

    resp = requests.post(f"{BASE}/run", json=payload, headers=headers(), timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Job submitted: {job_id}", file=sys.stderr)

    while True:
        status_resp = requests.get(f"{BASE}/status/{job_id}", headers=headers(), timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()

        if status_data["status"] == "COMPLETED":
            return status_data.get("output", status_data)
        if status_data["status"] == "FAILED":
            print(f"Job failed: {status_data}", file=sys.stderr)
            sys.exit(1)

        time.sleep(1)


def run_stream(prompt, max_tokens=512, temperature=0.7):
    """Submit a streaming job, poll the stream endpoint."""
    payload = {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    }

    resp = requests.post(f"{BASE}/run", json=payload, headers=headers(), timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Streaming job: {job_id}", file=sys.stderr)

    while True:
        stream_resp = requests.get(f"{BASE}/stream/{job_id}", headers=headers(), timeout=30)
        stream_resp.raise_for_status()
        stream_data = stream_resp.json()

        for chunk in stream_data.get("stream", []):
            output = chunk.get("output", chunk)
            if isinstance(output, dict):
                choices = output.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
            else:
                print(output, end="", flush=True)

        if stream_data.get("status") == "COMPLETED":
            print()
            return

        time.sleep(0.5)


def extract_content(output):
    """Pull the assistant message text from an OpenAI-compatible response."""
    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", json.dumps(output, indent=2))
    return json.dumps(output, indent=2)


def main():
    if not API_KEY or not ENDPOINT_ID:
        print(
            "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    streaming = "--stream" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--stream"]

    if not args:
        print(f"Usage: {sys.argv[0]} [--stream] \"your prompt\"", file=sys.stderr)
        sys.exit(1)

    prompt = " ".join(args)

    if streaming:
        run_stream(prompt)
    else:
        output = run_sync(prompt)
        print(extract_content(output))


if __name__ == "__main__":
    main()
