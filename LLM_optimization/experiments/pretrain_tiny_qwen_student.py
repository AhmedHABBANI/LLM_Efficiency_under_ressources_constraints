import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW


STUDENT_MODEL_PATH = "models/tiny_qwen_student"
OUTPUT_DIR = "models/tiny_qwen_student_pretrained_3000"
RESULTS_PATH = "results/tiny_qwen_pretraining_losses_3000.json"

MAX_LENGTH = 256
NUM_STEPS = 3000
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4
SAVE_EVERY = 500


def save_checkpoint(model, tokenizer, output_dir, step, losses):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    with open(os.path.join(checkpoint_dir, "losses.json"), "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=4)

    print(f"\nCheckpoint saved at step {step}: {checkpoint_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading TinyQwenStudent...")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH,
        torch_dtype=torch.float32
    ).to(device)

    model.train()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 100]

    print(f"Number of usable texts: {len(texts)}")
    print(f"Training steps: {NUM_STEPS}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")

    losses = []
    optimizer.zero_grad()

    for step in range(NUM_STEPS):
        text = texts[step % len(texts)]

        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        real_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS

        losses.append({
            "step": step + 1,
            "ce_loss": float(real_loss)
        })

        if (step + 1) % 50 == 0:
            recent = losses[-50:]
            avg_loss = sum(x["ce_loss"] for x in recent) / len(recent)
            print(
                f"Step {step + 1}/{NUM_STEPS} | "
                f"CE Loss: {real_loss:.4f} | "
                f"Avg last 50: {avg_loss:.4f}"
            )

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, tokenizer, OUTPUT_DIR, step + 1, losses)

    print("\nSaving final pretrained TinyQwenStudent...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=4)

    print(f"\nFinal pretrained TinyQwenStudent saved to: {OUTPUT_DIR}")
    print(f"Loss history saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()