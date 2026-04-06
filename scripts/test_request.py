"""
Step-by-step diagnostic script.

Usage:
    # Test mock server directly (port 8003)
    python scripts/test_request.py --target mock --port 8003

    # Test through proxy (port 9000)
    python scripts/test_request.py --target proxy --port 9000

    # Test streaming through proxy
    python scripts/test_request.py --target proxy --port 9000 --stream
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["mock", "proxy"], default="proxy")
parser.add_argument("--port", type=int, default=9000)
parser.add_argument("--stream", action="store_true")
parser.add_argument("--model", default="llm-proxy/router")
args = parser.parse_args()

BASE = f"http://127.0.0.1:{args.port}/v1"

SEP = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def post_json(url: str, payload: dict, stream: bool = False) -> None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  HTTP {resp.status}")
            if stream:
                print("  --- SSE chunks ---")
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data:") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[5:].strip())
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                print(f"  chunk: {delta!r}")
                        except Exception:
                            print(f"  raw: {line}")
                    elif line == "data: [DONE]":
                        print("  [DONE]")
            else:
                raw = resp.read().decode()
                try:
                    data = json.loads(raw)
                    content = data["choices"][0]["message"]["content"]
                    print(f"  content: {content!r}")
                    served = data.get("x-served-by") or data.get("model", "?")
                    print(f"  served-by / model: {served}")
                except Exception:
                    print(f"  raw response: {raw[:300]}")
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        print(f"  HTTP {e.code} ERROR")
        print(f"  body: {body_text[:300]}")
    except urllib.error.URLError as e:
        print(f"  CONNECTION ERROR: {e.reason}")
        print("  -> Is the server running?")


def get_models(url: str) -> None:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            print(f"  {len(models)} model(s): {models[:8]}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        print(f"  CONNECTION ERROR: {e.reason}")


# ── 1. Health check ────────────────────────────────────────────────────────
section("1. Health check")
health_url = f"http://127.0.0.1:{args.port}/health"
req = urllib.request.Request(health_url)
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        print(f"  OK - {resp.read().decode()}")
except Exception as e:
    print(f"  FAILED: {e}")
    print("  -> Server is not running on this port. Start it first.")
    sys.exit(1)

# ── 2. Model list ──────────────────────────────────────────────────────────
section("2. GET /v1/models")
get_models(f"{BASE}/models")

# ── 3. Chat completion ────────────────────────────────────────────────────
section(f"3. POST /v1/chat/completions  (stream={args.stream})")
payload = {
    "model": args.model,
    "stream": args.stream,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
}
print(f"  Sending: {json.dumps(payload)}")
post_json(f"{BASE}/chat/completions", payload, stream=args.stream)

print(f"\n{SEP}\nDone.\n")
