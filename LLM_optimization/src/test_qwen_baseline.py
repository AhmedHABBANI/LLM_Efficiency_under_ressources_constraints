import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully!")

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated text:")
print(text)

if torch.cuda.is_available():
    print("\nGPU:", torch.cuda.get_device_name(0))
    print("VRAM allocated:",
          round(torch.cuda.memory_allocated() / 1024**3, 2),
          "GB")
    print("VRAM reserved:",
          round(torch.cuda.memory_reserved() / 1024**3, 2),
          "GB")