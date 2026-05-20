# Model Efficiency and Compression for Small LLMs

**Quantization · Pruning · Knowledge Distillation · Pareto Analysis**

A practical study of how different model compression techniques affect the
quality, speed, and memory footprint of a small Large Language Model running
under tight hardware constraints (an 6 GB laptop GPU).

---

## Overview

Large Language Models are powerful but expensive to run. They require
significant GPU memory, storage, and compute. This project investigates a
single question:

> **How much efficiency can we gain while keeping model quality acceptable?**

Starting from a small baseline model, we progressively apply and combine
compression techniques, measure their impact on a consistent set of metrics,
and visualize the best quality–efficiency trade-offs as a **Pareto frontier**.

---

## Techniques Studied

- **Quantization** — dynamic (INT8, NF4 via Transformers) and static (GGUF Q4_K_M via llama.cpp / Ollama)
- **Pruning** — unstructured and structured (linear layers, MLP-only)
- **Knowledge Distillation** — from-scratch student, pretrained-student (SmolLM2), and LoRA-based distillation
- **Compression composition** — stacking distillation with quantization
- **Pareto analysis** — memory vs quality and throughput vs quality

---

## Experimental Setup

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| OS | Windows |
| Python | 3.11 |

**Baseline model:** `Qwen/Qwen2.5-0.5B`
**Distillation student:** SmolLM2 (pretrained) and a custom ~101M transformer

### Evaluation

- **Quality metric:** Perplexity on `wikitext-2-raw-v1` (lower is better)
- **Method:** Hugging Face sliding-window perplexity (`context_length = 512`, `stride = 256`)
- **Efficiency metrics:** VRAM usage, throughput (tokens/sec), latency, model size

---

## Key Results

### Baseline vs Quantization

| Version | Backend | Quantization | Perplexity ↓ | VRAM ↓ | Tokens/sec ↑ |
|---|---|---|---|---|---|
| FP16 (baseline) | Transformers | none | 16.78 | 0.94 GB | 31.91 |
| INT8 | Transformers | runtime (dynamic) | 16.88 | 0.60 GB | 9.6 |
| NF4 | Transformers | runtime (dynamic) | 18.75 | 0.45 GB | 12.04 |
| **GGUF Q4_K_M** | **llama.cpp / Ollama** | **static** | **16.31** | **~0.68 GB** | **180.08** |

> **Takeaway:** Static quantization (GGUF Q4_K_M) is the standout — ~5.6× faster
> than the FP16 baseline *and* slightly better perplexity, at lower VRAM.
> Dynamic quantization saves memory but adds runtime overhead that slows inference.

### Pruning

| Metric | Baseline | 20% unstructured | 40% unstructured |
|---|---|---|---|
| Perplexity ↓ | 16.78 | 16.22 | 95.44 |
| VRAM ↓ | 0.94 GB | 0.93 GB | 0.93 GB |
| Tokens/sec ↑ | 31.91 | 34.47 | 35.74 |

> **Takeaway:** On a small model, pruning is risky and low-reward. VRAM barely
> drops (dense matrices keep storing the zeroed weights), and quality collapses
> past ~20%. Structured pruning produced corrupted/garbage generations.

### Knowledge Distillation

| Approach | Perplexity ↓ | VRAM ↓ | Tokens/sec ↑ |
|---|---|---|---|
| From-scratch student (~101M) | failed (repetitive loops) | — | — |
| SmolLM2 distilled (before) | 20.40 | 0.26 GB | 31.10 |
| **SmolLM2 distilled (after)** | **16.55** | **0.26 GB** | **31.70** |
| Distilled SmolLM2 + INT8 | 20.42 | 0.22 GB | 18.70 |
| Distilled SmolLM2 + NF4 | 22.88 | 0.17 GB | 10.50 |
| **LoRA distillation** (0.34% params trained) | **16.69** | **0.26 GB** | **24.77** |

> **Takeaway:** Distilling from a *pretrained* student (SmolLM2) reaches near-baseline
> quality (16.55) at ~3.6× less VRAM. From-scratch distillation was not viable under
> these hardware constraints. LoRA achieves near-baseline quality while training only
> 0.34% of parameters.

---

## Conclusions

1. **Static quantization (GGUF Q4_K_M)** offers the best overall trade-off:
   faster, lighter, and quality preserved — the most practical deployment option.
2. **Pruning** is disappointing on small models: near-zero memory gain and
   fragile quality without recovery fine-tuning.
3. **Distillation from a pretrained model**, especially with **LoRA**, delivers
   near-baseline quality at minimal memory — but from-scratch distillation is
   unrealistic under limited hardware.
4. **There is no single winner** — the right method depends on the binding
   constraint: minimal VRAM → distillation + quantization; maximal speed → GGUF Q4_K_M.

---

## Project Structure

```text
llm-efficiency-compression/
├── README.md
├── requirements.txt
├── config/
├── src/
├── experiments/
├── results/
├── notebooks/
├── plots/
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AhmedHABBANI/LLM_Efficiency_under_ressources_constraints.git
cd llm-efficiency-compression

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core dependencies

```text
torch
transformers
datasets
accelerate
bitsandbytes
peft            # LoRA
pandas
matplotlib
psutil
```

For static quantization you will also need
[llama.cpp](https://github.com/ggerganov/llama.cpp) or
[Ollama](https://ollama.com/) to run the GGUF Q4_K_M model.

---



---

## Reproducibility

- Fixed random seed (`seed = 42`) across all experiments
- Identical perplexity protocol for every model (same dataset, context length, stride, sample count)
- Hardware context (GPU, CUDA availability, VRAM) logged in every result file

---

## Author

**Ahmed HABBANI**
