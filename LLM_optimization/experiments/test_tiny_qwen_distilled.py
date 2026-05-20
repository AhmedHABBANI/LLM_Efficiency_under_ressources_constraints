import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "models/tiny_qwen_student_pretrained"
PROMPT = "Artificial intelligence is"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading distilled TinyQwenStudent tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading distilled TinyQwenStudent model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    print("\nGenerating text...\n")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("===== DISTILLED GENERATED TEXT =====\n")
    print(text)


if __name__ == "__main__":
    main()