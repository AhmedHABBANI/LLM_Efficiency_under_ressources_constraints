import os
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


TEACHER_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "models/tiny_qwen_student"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading teacher config...")
    config = AutoConfig.from_pretrained(TEACHER_MODEL)

    print("\n===== ORIGINAL TEACHER CONFIG =====")
    print(f"vocab_size: {config.vocab_size}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")

    # TinyQwenStudent configuration
    config.hidden_size = 512
    config.intermediate_size = 1408
    config.num_hidden_layers = 8
    config.num_attention_heads = 8
    config.num_key_value_heads = 4

    # Important fix for Qwen configs
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = config.layer_types[:config.num_hidden_layers]

    print("\n===== TINY STUDENT CONFIG =====")
    print(f"vocab_size: {config.vocab_size}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")

    if hasattr(config, "layer_types") and config.layer_types is not None:
        print(f"layer_types length: {len(config.layer_types)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    print("Creating TinyQwenStudent from scratch...")
    model = AutoModelForCausalLM.from_config(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "teacher_model": TEACHER_MODEL,
        "student_model": "TinyQwenStudent",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    print("\n===== PARAMETER COUNT =====")
    print(json.dumps(info, indent=4))

    print("\nSaving TinyQwenStudent...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "student_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"\nTinyQwenStudent saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()