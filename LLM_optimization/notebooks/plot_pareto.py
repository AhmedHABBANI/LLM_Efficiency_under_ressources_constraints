import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = {
    "Model": ["FP16", "INT8", "NF4", "GGUF_Q4_K_M"],
    "Perplexity": [16.7764, 16.8820, 18.7531, 16.3136],
    "VRAM_GB": [0.94, 0.60, 0.45, 0.68],
    "Tokens_per_sec": [31.91, 9.60, 12.04, 180.08],
}

df = pd.DataFrame(data)

Path("plots").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

df.to_csv("results/results_summary.csv", index=False)

plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(
        row["VRAM_GB"],
        row["Perplexity"],
        s=row["Tokens_per_sec"] * 8
    )
    plt.text(
        row["VRAM_GB"] + 0.01,
        row["Perplexity"] + 0.03,
        row["Model"],
        fontsize=9
    )

plt.xlabel("VRAM Usage (GB)")
plt.ylabel("Perplexity (lower is better)")
plt.title("Pareto Trade-off: Quality vs Memory\nBubble size = Tokens/sec")
plt.grid(True, alpha=0.3)

plt.savefig("plots/pareto_frontier_llm_compression.png", bbox_inches="tight")
plt.show()