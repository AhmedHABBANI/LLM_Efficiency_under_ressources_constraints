import torch
import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


MODEL_NAME = "models/smollm2_distilled_from_qwen"
MAX_LENGTH = 512
STRIDE = 256


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
            outputs = model(
                input_ids_slice,
                labels=target_ids
            )
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    perplexity = torch.exp(torch.stack(nlls).sum() / end_loc)

    return perplexity.item()


def main():
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading distilled SmolLM2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading distilled SmolLM2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    text = "\n\n".join(dataset["text"])

    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, text, device)

    results = {
        "model": MODEL_NAME,
        "dataset": "wikitext-2-raw-v1 test",
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "perplexity": perplexity
    }

    print("\n===== DISTILLED SMOLLM2 PERPLEXITY RESULTS =====")
    print(json.dumps(results, indent=4))

    with open("results/smollm2_distilled_perplexity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nResults saved to results/smollm2_distilled_perplexity_results.json")


if __name__ == "__main__":
    main()