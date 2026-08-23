"""Deploy the benchmark's OpenAI-compatible vLLM server on Modal."""

import os
import signal
import subprocess

import modal


APP_NAME = os.getenv("MODAL_APP_NAME", "medmemorybench-vllm")
MODEL_ID = os.getenv(
    "MODAL_VLLM_MODEL",
    "ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ",
)
SERVED_MODEL_NAME = os.getenv("MODAL_VLLM_SERVED_MODEL", MODEL_ID)
PORT = 8000
GPU = os.getenv("MODAL_GPU", "L40S")
MAX_MODEL_LEN = os.getenv("MODAL_VLLM_MAX_MODEL_LEN", "32768")

app = modal.App(APP_NAME)

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
        }
    )
)

hf_cache = modal.Volume.from_name("medmemorybench-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("medmemorybench-vllm-cache", create_if_missing=True)


@app.server(
    image=image,
    gpu=GPU,
    port=PORT,
    startup_timeout=20 * 60,
    scaledown_window=5 * 60,
    target_concurrency=8,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class Server:
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
        self.process = subprocess.Popen(command)

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()