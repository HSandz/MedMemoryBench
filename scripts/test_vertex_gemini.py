#!/usr/bin/env python3
"""Run repeated live Gemini calls through the repository's Vertex AI client."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Call Gemini through Vertex AI repeatedly and log every result."
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: Vertex Gemini API test successful.",
    )
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between calls")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / f"vertex_gemini_smoke_{timestamp}.jsonl",
    )
    return parser.parse_args()


def write_log(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative")

    load_dotenv(PROJECT_ROOT / ".env")
    # One shared-client attempt per loop keeps --count equal to the requested API calls.
    os.environ["GEMINI_MAX_RETRIES"] = "1"

    from utils.llm_client import create_llm_client

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0

    with output_path.open("w", encoding="utf-8") as log_handle:
        try:
            client = create_llm_client(
                provider="gemini",
                model=args.model,
                temperature=0.0,
                max_tokens=64,
            )
        except Exception as exc:
            write_log(
                log_handle,
                {
                    "timestamp": utc_now(),
                    "event": "client_initialization",
                    "status": "error",
                    "model": args.model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"Client initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"Log: {output_path}")
            return 1

        print(
            f"Calling {args.model} {args.count} times via Vertex AI "
            f"(project={client.project}, location={client.location})"
        )
        print(f"Log: {output_path}")

        for request_index in range(1, args.count + 1):
            started = time.perf_counter()
            try:
                response = client.chat([{"role": "user", "content": args.prompt}])
                record = {
                    "timestamp": utc_now(),
                    "request_index": request_index,
                    "status": "success",
                    "model": args.model,
                    "project": client.project,
                    "location": client.location,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "response": response.content,
                }
                print(f"[{request_index:02d}/{args.count}] success: {response.content!r}")
            except Exception as exc:
                failures += 1
                record = {
                    "timestamp": utc_now(),
                    "request_index": request_index,
                    "status": "error",
                    "model": args.model,
                    "project": client.project,
                    "location": client.location,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(
                    f"[{request_index:02d}/{args.count}] error: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            write_log(log_handle, record)
            if args.delay and request_index < args.count:
                time.sleep(args.delay)

        write_log(
            log_handle,
            {
                "timestamp": utc_now(),
                "event": "summary",
                "total": args.count,
                "successes": args.count - failures,
                "failures": failures,
            },
        )

    print(f"Completed: {args.count - failures} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
