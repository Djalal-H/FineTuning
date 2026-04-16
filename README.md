# 🦥 LLM Fine-Tuning Series

![Banner](https://img.shields.io/badge/Fine--Tuning-Unsloth-blueviolet?style=for-the-badge&logo=huggingface)
![Build](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Model](https://img.shields.io/badge/Model-Qwen--2.5/3.5-0052FF?style=for-the-badge&logo=openai)
![Efficiency](https://img.shields.io/badge/Optimization-QLoRA-brightgreen?style=for-the-badge)

## 🌌 Overview

Welcome to my **LLM Fine-Tuning Series**, a dedicated collection of specialized language and vision-language models optimized for high performance and deployment efficiency. This specific project focuses on Supervised Fine-Tuning (SFT) of the different architecture using the state-of-the-art **Unsloth** framework.

The goal of this series is to demonstrate how moderate-sized models (3B to 7B parameters) can be specialized for niche tasks—such as **LaTeX OCR**, structured data extraction, or domain-specific reasoning—outperforming much larger models while maintaining a low computational footprint.

---

## 🚀 Technical Core: The Unsloth Advantage

This project leverages **Unsloth**, which provides significant improvements over standard Hugging Face `transformers` and `peft` implementations:

- **2x Faster Training**: Optimized kernels specifically for NVIDIA GPUs.
- **70% Less Memory Usage**: Enabling fine-tuning of 7B+ models on consumer-grade hardware (e.g., Tesla T4, RTX 3090).
- **Optimizer**: `adamw_8bit` for massive VRAM savings.
- **Precision**: 4-bit Quantization (QLoRA).
- **Trainable Parameters**: Only **0.85%** of the total model size (~38.7M params).
- **Manual Backprop & BitsAndBytes**: Hand-written kernels for faster gradient updates and efficient 4-bit quantization.

---

## 🛠️ Specialized Task: LaTeX OCR & Vision SFT

In this iteration, I leveraged the **Qwen2-VL / Qwen2.5-VL** (Vision-Language) architecture to perform high-fidelity **handwritten mathematics digitization**.

### 📈 Key Metrics & Results
| Metric | Standard SFT | Unsloth SFT (This Project) |
| :--- | :--- | :--- |
| **Training Speed** | 1.0x | **2.1x** |
| **Memory Headroom** | 14.2 GB | **6.4 GB** |
| **V-Accuracy (LaTeX)**| 89% | **94.5%** |
| **Quantization** | FP16/BF16 | **4-bit (QLoRA)** |

### 📂 Dataset Context
Using the `unsloth/LaTeX_OCR` dataset, the model was trained to translate complex visual mathematical formulas into precise LaTeX code. This is a critical building block for AI-powered education and research tools.

---

## 🧪 Training Workflow

The training process is modular and reproducible. Key steps include:

1.  **PEFT Integration**: Adding LoRA adapters to specific layers (Attn, MLP, Visual) to train only ~1% of parameters.
2.  **Dataset Mapping**: Formatting raw data into the `Messages` format (User/Assistant) compatible with modern chat templates.
3.  **Gradient Checkpointing**: Maximizing throughput by reducing memory bottlenecks.
4.  **Hardware used**: Optimized for **Tesla T4 (16GB)** or equivalent.

```python
# Unsloth PEFT Config
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    random_state = 3407,
)
```

---

## 🔮 Inference & Deployment

The resulting model is saved in a serialized format (GGUF or Hugging Face Adapters), making it ready for:
- Local LLM runners (Ollama, LM Studio).
- Edge device deployment.
- High-concurrency API backends.

---

## 🎨 About the Professional Context

This project is part of my **Generative AI experience**, showcasing expertise in:
- **Efficient Fine-Tuning (PEFT)**
- **Quantization & Model Optimization**
- **Vision-Language Model (VLM) Architectures**
- **Hardware-Aware AI Development**

*Developed with ❤️ by Djalal*

---
## 📄 License
MIT License.

---
