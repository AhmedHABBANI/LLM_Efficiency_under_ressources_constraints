import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_LABEL = "Qwen2.5-0.5B-FP16-baseline"

OUTPUT_PATH = Path("results/baseline_perplexity.json")

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"

CONTEXT_LENGTH = 512
STRIDE = 256
MAX_TOKENS = 8192


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    print("Loading WikiText dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)

    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids[:, :MAX_TOKENS].to(model.device)

    seq_len = input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    print("Computing perplexity with sliding window...")

    for begin_loc in tqdm(range(0, seq_len, STRIDE)):
        end_loc = min(begin_loc + CONTEXT_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_window.clone()

        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len).item()

    result = {
        "model": MODEL_LABEL,
        "base_model": MODEL_NAME,
        "precision": "FP16",
        "compression": "none",
        "perplexity_eval": {
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "split": SPLIT,
            "value": round(ppl, 4),
            "method": "hf_sliding_window",
            "context_length": CONTEXT_LENGTH,
            "stride": STRIDE,
            "max_tokens": MAX_TOKENS
        },
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nPerplexity evaluation completed.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()