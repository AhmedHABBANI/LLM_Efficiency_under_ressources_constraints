import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_LABEL = "Qwen2.5-0.5B-NF4-double-quant"

OUTPUT_PATH = Path("results/nf4_perplexity.json")

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"

CONTEXT_LENGTH = 512
STRIDE = 256
MAX_TOKENS = 8192


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)

    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :MAX_TOKENS].to(model.device)

    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

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
        "precision": "NF4",
        "compression": "bitsandbytes_4bit_nf4_double_quant",
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
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16"
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()