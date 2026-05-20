import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_LABEL = "Qwen2.5-0.5B-NF4-double-quant"

OUTPUT_PATH = Path("results/nf4_generation.json")


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

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    model.eval()

    prompt = "Artificial intelligence is"
    max_new_tokens = 100

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
    input_tokens = inputs["input_ids"].shape[-1]
    total_tokens = output.shape[-1]
    generated_tokens = total_tokens - input_tokens

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    result = {
        "model": MODEL_LABEL,
        "base_model": MODEL_NAME,
        "precision": "NF4",
        "compression": "bitsandbytes_4bit_nf4_double_quant",
        "generation_eval": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_tokens": int(generated_tokens),
            "tokens_per_sec": round(generated_tokens / latency_s, 2),
            "latency_ms": round(latency_s * 1000, 2)
        },
        "efficiency": {
            "ram_system_gb": round(get_ram_usage_gb(), 2),
            "vram_gb": round(get_vram_usage_gb(), 2)
        },
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16"
        },
        "generated_text_preview": generated_text[:500]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()