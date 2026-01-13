#!/usr/bin/env python3
from argparse import ArgumentParser
import time
import json
import sys
import requests
import argparse

# ANSI codes
BOLD   = "\033[1m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

# maintain rolling history of last 5 turns
history: list[tuple[str, str]] = []  # list of (user, assistant)

def format_context(keep_last_n=20):
    """
    Formats the last X turns as a messages list following Qwen3 chat_template style.
    We include only final outputs (no internal <think> block), per best practice.
    """
    msgs = []
    for u, a in history[-keep_last_n:]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs

def stream_chat(api_url, model, prompt: str):
    input_tokens = len(prompt.split())
    # prepare messages with history
    messages = format_context()
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "stream": True,
    }

    start = time.time()
    try:
        resp = requests.post(api_url, json=payload, stream=True)
        resp.raise_for_status()

        first_ts = None
        token_count = 0
        output = ""

        # print AI prefix
        print(f"{CYAN}{BOLD}AI:{RESET} ", end="", flush=True)

        for raw in resp.iter_lines(decode_unicode=False):
            if not raw: continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "): continue
            data = line[len("data: "):].strip()
            if data == "[DONE]": break

            chunk = json.loads(data)
            token = chunk["choices"][0]["delta"].get("content")
            if not token: continue

            now = time.time()
            if first_ts is None: first_ts = now
            token_count += 1
            output += token
            print(token, end="", flush=True)

        end = time.time()
        ttft = (first_ts - start) if first_ts else 0.0
        e2el = end - start
        tps  = token_count / (end - first_ts) if first_ts else 0.0

        print("\n")
        print(
            f"{YELLOW}{BOLD}Metrics{RESET}: "
            f"{BOLD}inp_tok{RESET}={input_tokens}  "
            f"{BOLD}out_tok{RESET}={token_count}  "
            f"{BOLD}ttft{RESET}={ttft:.3f}s  "
            f"{BOLD}e2el{RESET}={e2el:.3f}s  "
            f"{BOLD}tps{RESET}={tps:.1f}"
        )

        # store in history
        history.append((prompt, output))

    except requests.RequestException as e:
        print(f"\n{BOLD}{YELLOW}Error:{RESET} {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser: ArgumentParser = argparse.ArgumentParser(description="Streaming vLLM chat CLI with history (Qwen3-style)")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8066, help="vLLM server port")
    parser.add_argument("--model", default="Qwen3-14B", help="Model name")
    args = parser.parse_args()

    api_url: str = f"http://{args.host}:{args.port}/v1/chat/completions"
    model = args.model

    try:
        while True:
            prompt: str = input(f"{GREEN}{BOLD}You:{RESET} ").strip()
            if not prompt: continue
            stream_chat(api_url, model, prompt)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
