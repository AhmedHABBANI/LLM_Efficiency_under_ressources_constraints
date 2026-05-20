import json
import time
import subprocess
from pathlib import Path

import psutil
import requests


MODEL_NAME = "hf.co/QuantFactory/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
MODEL_LABEL = "Qwen2.5-0.5B-Instruct-GGUF-Q4_K_M"

OUTPUT_PATH = Path("results/gguf_q4_generation.json")


def get_ram_usage_gb():
    return psutil.virtual_memory().used / 1024**3


def get_gpu_memory_gb():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return round(float(result.stdout.strip()) / 1024, 2)
    except Exception:
        return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompt = "Artificial intelligence is"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100,
            "temperature": 0
        }
    }

    start = time.perf_counter()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=300
    )

    end = time.perf_counter()

    response.raise_for_status()
    data = response.json()

    latency_s = end - start

    generated_text = data.get("response", "")

    # Ollama returns nanoseconds for timing fields
    eval_count = data.get("eval_count", None)
    eval_duration = data.get("eval_duration", None)

    if eval_count and eval_duration:
        tokens_per_sec = eval_count / (eval_duration / 1e9)
    else:
        tokens_per_sec = None

    result = {
        "model": MODEL_LABEL,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "precision": "GGUF-Q4_K_M",
        "compression": "static_gguf_q4_k_m",
        "generation_eval": {
            "prompt": prompt,
            "max_new_tokens": 100,
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
            "latency_ms": round(latency_s * 1000, 2),
            "ollama_eval_count": eval_count,
            "ollama_eval_duration_ns": eval_duration
        },
        "efficiency": {
            "ram_system_gb": round(get_ram_usage_gb(), 2),
            "vram_gb_nvidia_smi": get_gpu_memory_gb()
        },
        "backend": {
            "runtime": "ollama",
            "format": "GGUF",
            "quantization": "Q4_K_M"
        },
        "generated_text_preview": generated_text[:500]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()