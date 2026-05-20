import torch
import time
import json
import os
import torch.nn.utils.prune as prune

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "Artificial intelligence is"
MAX_NEW_TOKENS = 100
PRUNING_AMOUNT = 0.20


def apply_structured_pruning(model, amount=0.20):
    """
    Apply structured pruning only on large Linear layers.
    We skip very small layers to avoid instability.
    """
    total_layers = 0
    skipped_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):

            # Skip small or special layers
            if module.weight.ndim != 2 or module.weight.shape[0] < 128:
                skipped_layers += 1
                continue

            try:
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=amount,
                    n=1,
                    dim=0
                )
                prune.remove(module, "weight")
                total_layers += 1

            except Exception as e:
                print(f"Skipped layer: {name} | reason: {e}")
                skipped_layers += 1

    print(f"Structured pruning applied on {total_layers} Linear layers.")
    print(f"Skipped layers: {skipped_layers}")
    print(f"Pruning amount: {amount * 100:.0f}%")
    print("Mode: L1 structured pruning, dim=0")

    return model


def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def main():
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Applying structured pruning...")
    model = apply_structured_pruning(model, amount=PRUNING_AMOUNT)

    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("Running structured pruning generation benchmark...")

    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    latency = end_time - start_time
    tokens_per_second = MAX_NEW_TOKENS / latency
    vram_usage = get_vram_usage()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    results = {
        "model": MODEL_NAME,
        "compression": f"structured_pruning_{int(PRUNING_AMOUNT * 100)}%",
        "pruning_type": "ln_structured_l1_dim0",
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "latency_seconds": latency,
        "tokens_per_second": tokens_per_second,
        "vram_gb": vram_usage,
        "generated_text": generated_text
    }

    print("\n===== STRUCTURED PRUNING GENERATION RESULTS =====")
    print(json.dumps(results, indent=4))

    output_path = f"results/structured_pruning_generation_results_{int(PRUNING_AMOUNT * 100)}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()