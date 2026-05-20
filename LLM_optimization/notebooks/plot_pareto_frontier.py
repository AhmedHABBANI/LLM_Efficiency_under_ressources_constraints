import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("plots", exist_ok=True)


data = [
    {
        "Technique": "FP16",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 16.7764,
        "VRAM_GB": 0.94,
        "Tokens_per_sec": 31.91,
    },
    {
        "Technique": "INT8",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 16.8820,
        "VRAM_GB": 0.60,
        "Tokens_per_sec": 9.60,
    },
    {
        "Technique": "NF4",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 18.7531,
        "VRAM_GB": 0.45,
        "Tokens_per_sec": 12.04,
    },
    {
        "Technique": "GGUF Q4_K_M",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 16.3136,
        "VRAM_GB": 0.68,
        "Tokens_per_sec": 180.08,
    },
    {
        "Technique": "Pruning 20%",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 16.2201,
        "VRAM_GB": 0.93,
        "Tokens_per_sec": 34.47,
    },
    {
        "Technique": "Pruning 40%",
        "Model": "Qwen2.5-0.5B",
        "Perplexity": 95.4381,
        "VRAM_GB": 0.93,
        "Tokens_per_sec": 35.74,
    },
    {
        "Technique": "SmolLM2-135M",
        "Model": "SmolLM2",
        "Perplexity": 16.5459,
        "VRAM_GB": 0.26,
        "Tokens_per_sec": 31.10,
    },
    {
        "Technique": "Distilled SmolLM2",
        "Model": "SmolLM2",
        "Perplexity": 20.3963,
        "VRAM_GB": 0.26,
        "Tokens_per_sec": 31.70,
    },

    {
    "Technique": "Distilled SmolLM2 INT8",
    "Model": "SmolLM2",
    "Perplexity": 20.4200,
    "VRAM_GB": 0.2185,
    "Tokens_per_sec": 10.50,
},
{
    "Technique": "Distilled SmolLM2 NF4",
    "Model": "SmolLM2",
    "Perplexity": 22.8757,
    "VRAM_GB": 0.1705,
    "Tokens_per_sec": 18.70,
},

{
    "Technique": "LoRA Distilled SmolLM2",
    "Model": "SmolLM2",
    "Perplexity": 16.6934,
    "VRAM_GB": 0.2636,
    "Tokens_per_sec": 24.77,
},
]


df = pd.DataFrame(data)


def is_pareto_efficient(df, x_col, y_col, minimize_x=True, minimize_y=True):
    """
    Returns Pareto-efficient points.

    For this project:
    - Perplexity should be minimized
    - VRAM should be minimized
    - Tokens/sec should be maximized
    """

    pareto = []

    for i, row in df.iterrows():
        dominated = False

        for j, other in df.iterrows():
            if i == j:
                continue

            x_better_or_equal = (
                other[x_col] <= row[x_col]
                if minimize_x
                else other[x_col] >= row[x_col]
            )

            y_better_or_equal = (
                other[y_col] <= row[y_col]
                if minimize_y
                else other[y_col] >= row[y_col]
            )

            x_strictly_better = (
                other[x_col] < row[x_col]
                if minimize_x
                else other[x_col] > row[x_col]
            )

            y_strictly_better = (
                other[y_col] < row[y_col]
                if minimize_y
                else other[y_col] > row[y_col]
            )

            if x_better_or_equal and y_better_or_equal and (
                x_strictly_better or y_strictly_better
            ):
                dominated = True
                break

        pareto.append(not dominated)

    return pareto


def plot_vram_vs_perplexity(df):
    df = df.copy()
    df["Pareto"] = is_pareto_efficient(
        df,
        x_col="VRAM_GB",
        y_col="Perplexity",
        minimize_x=True,
        minimize_y=True,
    )

    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        marker = "o" if row["Pareto"] else "x"
        size = 120 if row["Pareto"] else 80

        plt.scatter(
            row["VRAM_GB"],
            row["Perplexity"],
            s=size,
            marker=marker,
        )

        plt.annotate(
            row["Technique"],
            (row["VRAM_GB"], row["Perplexity"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    pareto_df = df[df["Pareto"]].sort_values("VRAM_GB")

    plt.plot(
        pareto_df["VRAM_GB"],
        pareto_df["Perplexity"],
        linestyle="--",
        linewidth=1,
        label="Pareto frontier",
    )

    plt.xlabel("VRAM usage (GB) ↓")
    plt.ylabel("Perplexity ↓")
    plt.title("Pareto Frontier: Memory vs Language Modeling Quality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = "plots/pareto_vram_vs_perplexity.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


def plot_speed_vs_perplexity(df):
    df = df.copy()
    df["Pareto"] = is_pareto_efficient(
        df,
        x_col="Tokens_per_sec",
        y_col="Perplexity",
        minimize_x=False,
        minimize_y=True,
    )

    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        marker = "o" if row["Pareto"] else "x"
        size = 120 if row["Pareto"] else 80

        plt.scatter(
            row["Tokens_per_sec"],
            row["Perplexity"],
            s=size,
            marker=marker,
        )

        plt.annotate(
            row["Technique"],
            (row["Tokens_per_sec"], row["Perplexity"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    pareto_df = df[df["Pareto"]].sort_values("Tokens_per_sec")

    plt.plot(
        pareto_df["Tokens_per_sec"],
        pareto_df["Perplexity"],
        linestyle="--",
        linewidth=1,
        label="Pareto frontier",
    )

    plt.xlabel("Throughput (tokens/sec) ↑")
    plt.ylabel("Perplexity ↓")
    plt.title("Pareto Frontier: Throughput vs Language Modeling Quality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = "plots/pareto_speed_vs_perplexity.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


def main():
    print("\n===== EXPERIMENTAL DATA =====")
    print(df)

    plot_vram_vs_perplexity(df)
    plot_speed_vs_perplexity(df)

    df.to_csv("results/final_compression_results.csv", index=False)
    print("Saved: results/final_compression_results.csv")


if __name__ == "__main__":
    main()