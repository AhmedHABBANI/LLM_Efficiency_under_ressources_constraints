import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_LABEL = "Qwen2.5-0.5B-Instruct-AWQ-Int4"

OUTPUT_PATH = Path("results/awq_generation.json")


def get_ram_usage_gb():
    return psutil.Process().memory_info().rss / 1024**3


def get_vram_usage_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def main():

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading AWQ model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    prompt = "Artificial intelligence is"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    max_new_tokens = 100

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency_s = end - start

    generated_tokens = (
        output.shape[-1]
        - inputs["input_ids"].shape[-1]
    )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    result = {
        "model": MODEL_LABEL,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "precision": "AWQ-Int4",
        "compression": "static_awq_int4",
        "generation_eval": {
            "tokens_per_sec": round(
                generated_tokens / latency_s,
                2
            ),
            "latency_ms": round(
                latency_s * 1000,
                2
            )
        },
        "efficiency": {
            "ram_system_gb": round(
                get_ram_usage_gb(),
                2
            ),
            "vram_gb": round(
                get_vram_usage_gb(),
                2
            )
        },
        "hardware": {
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            )
        },
        "generated_text_preview": generated_text[:500]
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()