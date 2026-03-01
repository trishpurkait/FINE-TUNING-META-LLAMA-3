# 🛒 LLaMA-3 E-Commerce Support Bot

![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-QLoRA-blueviolet?style=flat-square)
![Base Model](https://img.shields.io/badge/Base%20Model-Meta--Llama--3--8B--Instruct-orange?style=flat-square&logo=meta)
![Hardware](https://img.shields.io/badge/Hardware-Nvidia%20T4%2015GB-76b900?style=flat-square&logo=nvidia)
![Framework](https://img.shields.io/badge/Framework-HuggingFace%20Transformers-yellow?style=flat-square&logo=huggingface)
![UI](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A domain-specific customer support assistant fine-tuned from Meta's Llama-3-8B-Instruct using QLoRA — trained on a single consumer-grade T4 GPU in under 3 hours.

---

## 📖 Overview

This project fine-tunes **Meta-Llama-3-8B-Instruct** into a polite, factual e-commerce support agent capable of handling customer queries about orders, returns, refunds, and shipping policies. It is served through a **Gradio chat interface** with full multi-turn conversation memory.

The model was trained on **44,884 examples** from the `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`, intentionally stopped at **800 steps** (~15% of a full epoch), and uses 4-bit NF4 quantization to fit within a 15GB VRAM budget.

**Why I built it:** Most fine-tuning guides paper over the real engineering problems — VRAM ceilings, dtype incompatibilities, synthetic data artifacts, and the overfitting traps of highly structured datasets. This project confronts all of them, and every solution is documented below.

---

## 🏗️ Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio Chat UI                        │
│         Multi-turn · share=True public link              │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                   Inference Pipeline                     │
│      Manual prompt assembly → model.generate()          │
│      Regex post-processing (artifact scrubbing)          │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│            Fine-Tuned Model (checkpoint-800)             │
│   Meta-Llama-3-8B-Instruct + LoRA Adapters (float32)    │
│   Base weights: 4-bit NF4 quantized (float16 compute)   │
└─────────────────────────────────────────────────────────┘
```

| Component | Detail |
|---|---|
| **Base Model** | `meta-llama/Meta-Llama-3-8B-Instruct` |
| **Dataset** | `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` (44,884 examples) |
| **Fine-Tuning Method** | QLoRA via `peft` + `trl` SFTTrainer |
| **Quantization** | 4-bit NF4, double quant, `float16` compute dtype |
| **LoRA Rank / Alpha** | `r=16`, `lora_alpha=32` |
| **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Optimizer** | `paged_adamw_8bit` |
| **Hardware** | Single Nvidia T4 (15GB VRAM) — Kaggle |
| **Training Time** | ~2h 53m to step 800 |
| **Frontend** | `gradio` ChatInterface with `share=True` |

---

## 📉 Training Loss Curve

Training was stopped at step 800 with a final loss of **0.5248**, down from an opening loss of **2.0426** — rapid convergence that confirmed the early stopping decision.

| Phase | Steps | Loss Range |
|---|---|---|
| Rapid descent | 10 → 100 | 2.04 → 0.67 |
| Convergence plateau | 100 → 500 | 0.67 → 0.56 |
| Stable fine-tuning | 500 → 800 | 0.56 → 0.52 |

The loss plateau beginning around step 100 is the key signal: the model had absorbed the domain's vocabulary and tone. Continued training on synthetic, repetitive data past this point risks format memorization rather than further generalization.

---

## ⚙️ Engineering Challenges & Solutions

### 1. 🎯 Strategic Early Stopping — Fighting Overfitting on Synthetic Data

**The Problem:** The Bitext dataset is machine-generated and highly templated. Training for a full epoch (5,611 steps) risks the model locking onto rigid response schemas rather than internalizing genuine conversational fluency — a subtle form of format memorization that standard loss metrics won't surface.

**The Solution:** The training run was configured with `save_steps=100` and `save_total_limit=2` so checkpoints were continuously available for qualitative evaluation. After monitoring generation quality alongside the flattening loss curve, training was **deliberately interrupted via `KeyboardInterrupt` at step 844**. The `checkpoint-800` artifact was selected as the optimal stopping point and is the checkpoint used for all inference.

```python
training_args = SFTConfig(
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=100,        # Continuous checkpoints for manual evaluation
    save_total_limit=2,    # Disk-efficient: only keep the two latest
    logging_steps=10,
    ...
)
# Training deliberately interrupted after observing plateau
# checkpoint-800 selected as the final model artifact
```

---

### 2. 🔧 The BFloat16 / Float16 Dtype Problem on the T4 GPU

**The Problem:** Llama 3 is distributed natively in `bfloat16`. The Nvidia T4 does not support `bfloat16` computation. When left uncorrected, this produces two compounding failures: the `GradScaler` (used in `fp16=True` mixed-precision training) encounters NaN gradients and crashes, and — more insidiously — PEFT silently initializes LoRA adapter weights in `bfloat16` by inheriting from `model.config.torch_dtype`, making the root cause non-obvious to diagnose.

**The Solution:** A three-layer dtype strategy was applied:

1. Both `bnb_4bit_compute_dtype` and `torch_dtype` at load time are forced to `torch.float16`.
2. `model.config.torch_dtype` is **explicitly overwritten after loading** so PEFT cannot inherit `bfloat16` when attaching adapters.
3. A **parameter sweep failsafe** runs after `SFTTrainer` initialization: every `requires_grad=True` parameter (the LoRA adapters) is cast to `float32` for gradient stability, and any remaining stray `bfloat16` frozen weights are downcast to `float16`.

```python
# Layer 1: Force float16 at load time
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16   # Not bfloat16 — T4 incompatible
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.float16
)

# Layer 2: Prevent PEFT from spawning bfloat16 adapters
model.config.torch_dtype = torch.float16

# Layer 3: Post-trainer-init failsafe sweep
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)  # Adapters: float32 for gradient stability
    elif param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)  # Frozen weights: purge rogue bfloat16
```

---

### 3. 💾 VRAM Optimization: Training a ~15GB Model on a 15GB GPU

**The Problem:** At full `float16` precision, Llama-3-8B requires approximately 16GB of VRAM for weights alone — before activations, gradients, or optimizer states. The Kaggle T4 provides exactly 15GB.

**The Solution:** Four techniques were stacked to stay within budget:

- **4-bit NF4 + double quantization** compresses the frozen base model from ~16GB to ~4–5GB.
- **`paged_adamw_8bit`** offloads optimizer states to CPU-paged memory, dramatically cutting their VRAM footprint.
- **Gradient checkpointing** trades compute for memory by recomputing activations on the backward pass rather than storing them.
- **Gradient accumulation** (`batch_size=1`, `accumulation_steps=8`) simulates an effective batch size of 8 without ever materializing more than one sample's activations simultaneously.

```python
training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,    # Effective batch size = 8
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    max_length=512,
    ...
)
```

---

### 4. 🧹 Regex Post-Processing — Scrubbing Synthetic Dataset Artifacts

**The Problem:** The Bitext dataset is machine-generated and contains unfilled template placeholders — `{{Order Number}}`, `{{Customer Name}}`, `{{Product}}` — embedded directly in training responses. Without intervention, the fine-tuned model reproduces these tokens verbatim in generation, which is an immediate credibility failure in a demo.

**The Solution:** A `re.sub` call is integrated directly into the Gradio `respond` function and runs on every response before it reaches the user. The pattern `r'\{\{(.*?)\}\}'` matches any double-brace placeholder and replaces it with the inner label text — preserving semantic readability while cleanly removing the synthetic formatting artifact.

```python
def respond(message, history):
    # ... generation logic ...
    response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    # Replace {{Placeholder}} artifacts with their inner label text
    clean_response = re.sub(r'\{\{(.*?)\}\}', r'\1', response)
    return clean_response.strip()
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU (8GB+ VRAM; 4-bit quantization is applied automatically)
- A Hugging Face account with access granted to `meta-llama/Meta-Llama-3-8B-Instruct`

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llama3-ecommerce-bot.git
cd llama3-ecommerce-bot
```

### 2. Install Dependencies

```bash
pip install torch transformers peft trl bitsandbytes accelerate gradio datasets
```

### 3. Authenticate with Hugging Face

```bash
huggingface-cli login
```

### 4. Set Up the Checkpoint

Place the `checkpoint-800` folder inside the project directory, or update `ADAPTER_PATH` in `inference.py` to point to its location:

```
llama3-ecommerce-bot/
└── checkpoint-800/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

### 5. Launch

```bash
python inference.py
```

Navigate to `http://127.0.0.1:7860`. Set `share=True` in `demo.launch()` to generate a temporary public Gradio link.

---

## 💬 Usage

The complete inference script, taken directly from the training notebook:

```python
import re
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH  = "./checkpoint-800"

# ── Load base model in 4-bit ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.float16
)

# ── Apply fine-tuned LoRA adapter ──────────────────────────────────────────────
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# ── Multi-turn inference ───────────────────────────────────────────────────────
def respond(message, history):
    sys_prompt = "You are a polite and helpful e-commerce customer support AI."
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"

    for user_msg, bot_msg in history:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=150, temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    response_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response        = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return re.sub(r'\{\{(.*?)\}\}', r'\1', response).strip()

# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="🛒 E-Commerce Support Bot",
    description="Fine-tuned Llama 3 assistant. Ask me about orders, returns, or shipping!",
    theme="soft"
)
demo.launch(share=True)
```

**Example queries to try:**
- *"Where is my order?"*
- *"How do I return a damaged item?"*
- *"What is your refund policy?"*
- *"Can I change my shipping address after placing an order?"*

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **No live data access.** The model has no connection to a real order database. All responses are policy-level generalizations, not account-specific answers.
- **Synthetic training data.** Bitext is machine-generated. The model performs well on common query patterns but may behave unexpectedly on adversarial or highly colloquial inputs.
- **15% of one epoch.** The model was stopped at 800 of 5,611 total steps. While this was the right decision for this dataset, it means the model hasn't seen the full training distribution.
- **GPU required for inference.** 4-bit quantization is necessary for T4-class hardware. A GPU with 8GB+ VRAM is recommended.

### Future Work

- [ ] **RAG pipeline** — Ground responses in a real product/policy knowledge base using a retrieval layer (e.g., LangChain + FAISS).
- [ ] **DPO alignment** — Apply Direct Preference Optimization on human-rated response pairs to improve tone, accuracy, and refusal behavior.
- [ ] **Curated dataset extension** — Deduplicate and filter the Bitext dataset to enable safe training beyond 800 steps without format memorization.
- [ ] **GGUF export** — Convert to GGUF for CPU inference via `llama.cpp`, enabling deployment without a GPU.
- [ ] **Structured evaluation** — Implement ROUGE, BERTScore, and human evaluation benchmarks to measure response quality quantitatively.
- [ ] **Hugging Face Spaces** — Deploy the Gradio app as a permanent, publicly accessible Space.

---

## 📄 License

This project is licensed under the MIT License. The base model (`meta-llama/Meta-Llama-3-8B-Instruct`) is subject to Meta's [Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/).

---

<p align="center">
  Built on a Kaggle T4 · Stopped at the right moment · Shipped with intention
</p>
