import os
import json
import torch

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


BASE_STUDENT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = "data/teacher_synthetic_dataset.jsonl"

OUTPUT_DIR = "models/smollm2_lora_distilled_from_qwen"
RESULTS_PATH = "results/smollm2_lora_distillation_losses.json"

MAX_LENGTH = 256
EPOCHS = 5
LEARNING_RATE = 5e-5


def load_teacher_dataset(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["teacher_output"].strip()

            if len(text) > 20:
                examples.append(text)

    return examples


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    percent = 100 * trainable / total

    print("\n===== TRAINABLE PARAMETERS =====")
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable percentage: {percent:.4f}%")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_STUDENT_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading base SmolLM2 student...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_STUDENT_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("\nApplying LoRA adapters...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.train()

    print_trainable_parameters(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )

    print("\nLoading teacher synthetic dataset...")
    texts = load_teacher_dataset(DATA_PATH)

    print(f"Number of training examples: {len(texts)}")
    print(f"Epochs: {EPOCHS}")

    losses = []
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

        for text in texts:
            global_step += 1

            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length"
            )

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()

            losses.append({
                "step": global_step,
                "epoch": epoch + 1,
                "loss": float(loss.item())
            })

            if global_step % 10 == 0:
                recent = losses[-10:]
                avg_loss = sum(x["loss"] for x in recent) / len(recent)
                print(
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg last 10: {avg_loss:.4f}"
                )

    print("\nSaving LoRA distilled adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=4)

    print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")
    print(f"Loss history saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()