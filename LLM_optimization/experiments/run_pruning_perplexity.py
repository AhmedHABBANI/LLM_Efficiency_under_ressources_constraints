import torch
import math
import json
import os
import torch.nn.utils.prune as prune

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
STRIDE = 256
PRUNING_AMOUNT = 0.40


def apply_unstructured_pruning(model, amount=0.20):
    """
    Apply L1 unstructured pruning on all Linear layers.
    amount=0.20 means 20% of the weights are set to zero.
    """
    total_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
            total_layers += 1

    print(f"Pruning applied on {total_layers} Linear layers.")
    print(f"Pruning amount: {amount * 100:.0f}%")

    return model


def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids_slice = input_ids[:, begin_loc:end_loc]

        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)

            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl.item()


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

    print("Applying pruning...")
    model = apply_unstructured_pruning(model, amount=PRUNING_AMOUNT)

    model.eval()

    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    text = "\n\n".join(dataset["text"])

    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, text, device)

    results = {
        "model": MODEL_NAME,
        "compression": f"unstructured_pruning_{int(PRUNING_AMOUNT * 100)}%",
        "dataset": "wikitext-2-raw-v1 test",
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "perplexity": perplexity
    }

    print("\n===== PRUNING PERPLEXITY RESULTS =====")
    print(json.dumps(results, indent=4))

    with open("results/pruning_perplexity_results_40.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nResults saved to results/pruning_perplexity_results_40.json")


if __name__ == "__main__":
    main()