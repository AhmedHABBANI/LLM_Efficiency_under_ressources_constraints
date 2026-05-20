import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_LABEL = "Qwen2.5-0.5B-FP16-baseline"

OUTPUT_PATH = Path("results/baseline_generation.json")


def get_ram_usage_gb():
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def get_vram_usage_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    prompt = "Artificial intelligence is"
    max_new_tokens = 100

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Running generation benchmark...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    latency_s = end_time - start_time
    latency_ms = latency_s * 1000

    input_tokens = inputs["input_ids"].shape[-1]
    total_tokens = output.shape[-1]
    generated_tokens = total_tokens - input_tokens

    tokens_per_sec = generated_tokens / latency_s

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    result = {
        "model": MODEL_LABEL,
        "base_model": MODEL_NAME,
        "precision": "FP16",
        "compression": "none",
        "generation_eval": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_tokens": int(generated_tokens),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "latency_ms": round(latency_ms, 2)
        },
        "efficiency": {
            "ram_system_gb": round(get_ram_usage_gb(), 2),
            "vram_gb": round(get_vram_usage_gb(), 2)
        },
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "generated_text_preview": generated_text[:500]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nBenchmark completed.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()