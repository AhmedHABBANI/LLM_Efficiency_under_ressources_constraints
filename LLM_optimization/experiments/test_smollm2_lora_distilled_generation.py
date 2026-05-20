import torch
import time
import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
LORA_ADAPTER_PATH = "models/smollm2_lora_distilled_from_qwen"

PROMPT = "Artificial intelligence is"
MAX_NEW_TOKENS = 100


def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base SmolLM2...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH
    )

    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nGenerating text...\n")

    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    latency = end_time - start_time
    tokens_per_second = MAX_NEW_TOKENS / latency
    vram_usage = get_vram_usage()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    results = {
        "model": BASE_MODEL,
        "adapter": LORA_ADAPTER_PATH,
        "compression": "lora_distillation",
        "trainable_params": 460800,
        "trainable_percentage": 0.3414,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "latency_seconds": latency,
        "tokens_per_second": tokens_per_second,
        "vram_gb": vram_usage,
        "generated_text": generated_text
    }

    print("\n===== LORA DISTILLED SMOLLM2 GENERATION RESULTS =====")
    print(json.dumps(results, indent=4, ensure_ascii=False))

    with open("results/smollm2_lora_distilled_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nResults saved to results/smollm2_lora_distilled_generation_results.json")


if __name__ == "__main__":
    main()