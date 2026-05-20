from datasets import load_dataset
from pathlib import Path

OUTPUT_PATH = Path("data/wikitext_test.txt")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading WikiText...")

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="test"
)

text = "\n\n".join(dataset["text"])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved to: {OUTPUT_PATH}")