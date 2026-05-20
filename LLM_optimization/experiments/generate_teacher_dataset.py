import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


TEACHER_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_PATH = "data/teacher_synthetic_dataset.jsonl"

PROMPTS = [
    "Artificial intelligence is",
    "Machine learning can be used to",
    "Deep learning is useful because",
    "The future of technology depends on",
    "Data science helps companies",
    "Natural language processing is",
    "A neural network learns by",
    "Model compression is important because",
    "Quantization reduces",
    "Knowledge distillation allows",
    "Pruning a neural network means",
    "Efficient AI models are useful for",
    "A language model predicts",
    "The main challenge of AI is",
    "In computer science, optimization means",
]

NUM_GENERATIONS_PER_PROMPT = 5
MAX_NEW_TOKENS = 100


def main():
    os.makedirs("data", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading teacher tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    print("Loading teacher model...")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    examples = []

    print("Generating synthetic dataset...")

    for prompt in tqdm(PROMPTS):
        for i in range(NUM_GENERATIONS_PER_PROMPT):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )

            text = tokenizer.decode(output[0], skip_special_tokens=True)

            examples.append({
                "prompt": prompt,
                "teacher_output": text
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSynthetic dataset saved to: {OUTPUT_PATH}")
    print(f"Number of examples: {len(examples)}")


if __name__ == "__main__":
    main()