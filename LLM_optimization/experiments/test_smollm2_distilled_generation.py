import torch
import time
import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "models/smollm2_distilled_from_qwen"
PROMPT = "Artificial intelligence is"
MAX_NEW_TOKENS = 100


def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def main():
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading distilled SmolLM2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading distilled SmolLM2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

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

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    results = {
        "model": MODEL_PATH,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "latency_seconds": latency,
        "tokens_per_second": tokens_per_second,
        "vram_gb": vram_usage,
        "generated_text": text
    }

    print("\n===== DISTILLED SMOLLM2 GENERATION RESULTS =====")
    print(json.dumps(results, indent=4, ensure_ascii=False))

    with open("results/smollm2_distilled_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nResults saved to results/smollm2_distilled_generation_results.json")


if __name__ == "__main__":
    main()