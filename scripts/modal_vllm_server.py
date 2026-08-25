"""Deploy the benchmark's OpenAI-compatible vLLM server on Modal."""

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal


def _load_local_env() -> None:
    """Load deployment settings locally without copying secrets remotely."""
    if not modal.is_local():
        return
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)


_load_local_env()


APP_NAME = os.getenv("MODAL_APP_NAME", "medmemorybench-vllm")
MODEL_ID = os.getenv(
    "MODAL_VLLM_MODEL",
    "ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ",
)
SERVED_MODEL_NAME = os.getenv("MODAL_VLLM_SERVED_MODEL", MODEL_ID)
PORT = 8000
GPU = os.getenv("MODAL_GPU", "L40S")
MAX_MODEL_LEN = os.getenv("MODAL_VLLM_MAX_MODEL_LEN", "32768")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
SCALEDOWN_WINDOW_SECONDS = int(
    os.getenv("MODAL_VLLM_SCALEDOWN_WINDOW_SECONDS", "300")
)
WARMUP_ENABLED = os.getenv("MODAL_VLLM_WARMUP", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
WARMUP_PROMPT_LENGTHS = os.getenv(
    "MODAL_VLLM_WARMUP_PROMPT_LENGTHS",
    "128,512,2048,8192",
)
WARMUP_MAX_TOKENS = os.getenv("MODAL_VLLM_WARMUP_MAX_TOKENS", "32")
GPU_MEMORY_UTILIZATION = os.getenv("MODAL_VLLM_GPU_MEMORY_UTILIZATION", "0.9")

app = modal.App(APP_NAME)
server_secrets = (
    [modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})]
    if HF_TOKEN
    else []
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Modal imports the source again inside the remote container. Bake
            # deployment-time values into the image so that remote imports do
            # not fall back to the local defaults.
            "MODAL_VLLM_MODEL": MODEL_ID,
            "MODAL_VLLM_SERVED_MODEL": SERVED_MODEL_NAME,
            "MODAL_VLLM_MAX_MODEL_LEN": MAX_MODEL_LEN,
            "MODAL_VLLM_SCALEDOWN_WINDOW_SECONDS": str(
                SCALEDOWN_WINDOW_SECONDS
            ),
            "MODAL_VLLM_WARMUP": "1" if WARMUP_ENABLED else "0",
            "MODAL_VLLM_WARMUP_PROMPT_LENGTHS": WARMUP_PROMPT_LENGTHS,
            "MODAL_VLLM_WARMUP_MAX_TOKENS": WARMUP_MAX_TOKENS,
            "MODAL_VLLM_GPU_MEMORY_UTILIZATION": GPU_MEMORY_UTILIZATION,
        }
    )
)

hf_cache = modal.Volume.from_name("medmemorybench-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("medmemorybench-vllm-cache", create_if_missing=True)


def _release_deploy_start_pin() -> None:
    """Restore scale-to-zero after a full ready-state idle window."""
    print(
        "vLLM ready; deploy startup pin releases in "
        f"{SCALEDOWN_WINDOW_SECONDS} seconds"
    )
    time.sleep(SCALEDOWN_WINDOW_SECONDS)
    while True:
        try:
            Server.update_autoscaler(min_containers=0)
            print("Released deploy startup pin; server can now scale to zero")
            return
        except Exception as exc:
            print(f"Failed to release deploy startup pin; retrying: {exc}")
            time.sleep(30)


@app.server(
    image=image,
    gpu=GPU,
    secrets=server_secrets,
    port=PORT,
    startup_timeout=20 * 60,
    # Deployment temporarily starts one container. The ready container changes
    # this back to zero after the idle grace period below.
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    target_concurrency=8,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class Server:
    @staticmethod
    def _is_qwen35_model(model_id: str) -> bool:
        return "qwen3.8" in model_id.lower()

    @staticmethod
    def _parse_prompt_lengths(value: str) -> list[int]:
        lengths = []
        for item in value.split(","):
            try:
                length = int(item.strip())
            except ValueError:
                continue
            if length > 0:
                lengths.append(length)
        return lengths or [128, 512, 2048]

    @staticmethod
    def _request_json(path: str, payload: dict | None = None) -> int:
        data = None
        headers = {}
        method = "GET"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            return response.status

    def _wait_until_ready(self, timeout: int = 20 * 60) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._request_json("/health") == 200:
                    return
            except (OSError, urllib.error.HTTPError):
                pass
            time.sleep(2)
        raise TimeoutError("vLLM did not become ready before the startup timeout")

    def _warmup(self, model_id: str) -> None:
        if os.environ.get("MODAL_VLLM_WARMUP", "1").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            print("vLLM warmup disabled")
            return

        prompt_lengths = self._parse_prompt_lengths(
            os.environ.get(
                "MODAL_VLLM_WARMUP_PROMPT_LENGTHS", "128,512,2048,8192"
            )
        )
        max_tokens = int(os.environ.get("MODAL_VLLM_WARMUP_MAX_TOKENS", "32"))
        print(
            f"Warming up vLLM model={model_id} prompt_lengths={prompt_lengths} "
            f"max_tokens={max_tokens}"
        )
        for prompt_length in prompt_lengths:
            prompt = "warmup " * prompt_length
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            deadline = time.monotonic() + 5 * 60
            while True:
                try:
                    status = self._request_json("/v1/chat/completions", payload)
                    if status == 200:
                        break
                except (OSError, urllib.error.HTTPError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"vLLM warmup failed for prompt length {prompt_length}"
                    )
                time.sleep(2)
            print(f"Completed vLLM warmup prompt_length={prompt_length}")

    @modal.enter()
    def start(self):
        model_id = os.environ["MODAL_VLLM_MODEL"]
        served_model_name = os.environ["MODAL_VLLM_SERVED_MODEL"]
        max_model_len = os.environ["MODAL_VLLM_MAX_MODEL_LEN"]
        print(
            "Starting vLLM model="
            f"{model_id} served_model={served_model_name} "
            f"max_model_len={max_model_len}"
        )
        command = [
            "vllm",
            "serve",
            model_id,
            "--served-model-name",
            served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--max-model-len",
            max_model_len,
            "--trust-remote-code",
        ]
        if self._is_qwen35_model(model_id):
            command.extend(
                [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "qwen3_coder",
                    "--reasoning-parser",
                    "qwen3",
                    "--mm-encoder-tp-mode",
                    "data",
                    "--gpu-memory-utilization",
                    os.environ.get("MODAL_VLLM_GPU_MEMORY_UTILIZATION", "0.9"),
                ]
            )
        self.process = subprocess.Popen(command)
        self._wait_until_ready()
        self._warmup(model_id)
        threading.Thread(
            target=_release_deploy_start_pin,
            name="modal-deploy-start-pin",
            daemon=True,
        ).start()

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
