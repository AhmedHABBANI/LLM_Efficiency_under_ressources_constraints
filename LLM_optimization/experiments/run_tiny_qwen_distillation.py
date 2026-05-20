import os
import math
import json
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW


TEACHER_MODEL = "Qwen/Qwen2.5-0.5B"
STUDENT_MODEL_PATH = "models/tiny_qwen_student"
OUTPUT_DIR = "models/tiny_qwen_student_distilled"

MAX_LENGTH = 128
NUM_STEPS = 200
LEARNING_RATE = 1e-5
TEMPERATURE = 2.0
ALPHA = 0.7  # distillation loss weight


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL divergence between teacher and student distributions.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    return loss


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print("\nLoading student model...")
    student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH,
    torch_dtype=torch.float32
).to(device)
    student.train()

    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)

    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 50]

    print(f"Number of usable texts: {len(texts)}")
    print(f"Training steps: {NUM_STEPS}")

    losses = []

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

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        student_logits = student_outputs.logits

        ce_loss = student_outputs.loss

        kd_loss = distillation_loss(
            student_logits,
            teacher_logits,
            temperature=TEMPERATURE
        )

        total_loss = ALPHA * kd_loss + (1 - ALPHA) * ce_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append({
            "step": step + 1,
            "total_loss": float(total_loss.item()),
            "kd_loss": float(kd_loss.item()),
            "ce_loss": float(ce_loss.item())
        })

        if (step + 1) % 10 == 0:
            print(
                f"Step {step + 1}/{NUM_STEPS} | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"KD Loss: {kd_loss.item():.4f} | "
                f"CE Loss: {ce_loss.item():.4f}"
            )

    print("\nSaving distilled student...")
    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open("results/tiny_qwen_distillation_losses.json", "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=4)

    print(f"\nDistilled TinyQwenStudent saved to: {OUTPUT_DIR}")
    print("Loss history saved to: results/tiny_qwen_distillation_losses.json")


if __name__ == "__main__":
    main()