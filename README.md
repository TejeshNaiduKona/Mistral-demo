# 🧠 MistralMind — Multi-Specialist AI Agent OS

> *4 fine-tuned Mistral specialists. One intelligent routing brain. Infinite cross-domain power.*

**MistralMind** is a system of domain expert models that **collaborate in real-time**, orchestrated by an intelligent routing agent. Ask one question — get a team of PhDs working on it.

---

## ⚡ What Makes This Win

| Feature | Typical Hackathon Project | MistralMind |
|---|---|---|
| Fine-tuning | One model, one dataset | **4 specialists, 4 curated datasets** |
| Task routing | Manual / none | **Automatic intelligent routing** |
| Multi-domain queries | Single perspective | **Parallel expert collaboration** |
| W&B tracking | Basic loss curves | **Before/after benchmarks per specialist** |
| Demo | Static output | **Live routing visualization** |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│     ROUTER (mistral-small)      │  ← Analyzes → decides mode
└─────────────┬───────────────────┘
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
  SINGLE  PARALLEL  SEQUENTIAL
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   🏥 Medical      📈 Finance      💻 Code
   Mistral-7B      Mistral-7B     Mistral-7B
  (PubMedQA FT)  (Finance-α FT) (CodeAlpaca FT)
                                       │
                   🎨 Creative ←───────┘
                   Mistral-7B
                  (GPTJ-pair FT)
         │
         ▼
┌─────────────────────────────────┐
│   SYNTHESIZER (mistral-large)   │  ← Merges into one cohesive answer
└─────────────────────────────────┘
```

### Three Routing Modes
- **SINGLE ⚡** — One expert handles it (fast path for clear single-domain queries)
- **PARALLEL 🔀** — Multiple experts answer simultaneously, synthesizer merges
- **SEQUENTIAL 🔗** — Expert A's output feeds Expert B (chained reasoning)

---

## 📦 Datasets Used

| Specialist | Dataset | Size | Why This Dataset |
|---|---|---|---|
| 🏥 Medical | `qiaojin/PubMedQA` | 1,000 labeled QA | Real biomedical questions with context from PubMed abstracts |
| 📈 Finance | `gbharti/finance-alpaca` | 68,912 samples | Alpaca-style finance instruction-following dataset |
| 💻 Code | `sahil2801/CodeAlpaca-20k` | 20,022 samples | Code generation/instruction following across languages |
| 🎨 Creative | `Dahoas/synthetic-instruct-gptj-pairwise` | 33,143 samples | Preference-aligned creative responses (uses `chosen` column) |

---

## 🔬 Fine-Tuning Details

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Framework:** Unsloth (2× faster, 50% less VRAM) + TRL SFTTrainer
- **Method:** QLoRA — 4-bit quantization + LoRA (r=64, α=128, RSLoRA)
- **Packing:** Sequence packing enabled (reduces wasted compute from padding)
- **Tracking:** Every run logged to W&B with before/after eval dashboards

---

## 🚀 Setup & Run

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/MistralMind
cd MistralMind
pip install -r requirements.txt

# Unsloth (GPU-specific — pick the right one)
# Colab / RunPod:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Local CUDA:
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Configure Keys

```bash
cp .env.example .env
# Edit .env and add:
#   MISTRAL_API_KEY=your_key
#   WANDB_API_KEY=your_key
```

### 3. Evaluate BEFORE Fine-Tuning (Baseline)

```bash
python finetune/evaluate_specialists.py --specialist medical  --phase before
python finetune/evaluate_specialists.py --specialist finance  --phase before
python finetune/evaluate_specialists.py --specialist code     --phase before
python finetune/evaluate_specialists.py --specialist creative --phase before
```

### 4. Fine-Tune All Specialists

```bash
# Train each (run in parallel across GPUs for speed)
python finetune/train_specialist.py --specialist medical  --epochs 3
python finetune/train_specialist.py --specialist finance  --epochs 3
python finetune/train_specialist.py --specialist code     --epochs 3
python finetune/train_specialist.py --specialist creative --epochs 3
```

### 5. Evaluate AFTER Fine-Tuning (Comparison)

```bash
python finetune/evaluate_specialists.py --specialist medical --phase after \
  --checkpoint ./checkpoints/medical/lora_adapter
# Repeat for each specialist
```

### 6. Launch the Demo

```bash
python demo/app.py
# → http://localhost:7860 (auto-opens shareable link)
```

### 7. Use the Agent in Code

```python
from agent.router import MistralMindAgent

agent = MistralMindAgent()   # reads MISTRAL_API_KEY from .env

result = agent.think(
    "I'm launching a health-tech startup. Analyze the market, "
    "model the financials, and write a compelling pitch narrative."
)

print(f"Mode: {result.routing.mode.value}")
print(f"Experts: {[s.value for s in result.routing.specialists]}")
print(result.synthesis)
```

---

## 🎯 Killer Demo Queries (for Judges)

### Cross-Domain Collaboration (PARALLEL mode)
> *"I'm a 40-year-old software engineer working 70+ hour weeks. What are the long-term health risks and how should I adjust my financial planning to account for future health costs?"*

→ 🏥 Medical + 📈 Finance → Synthesized

### Chained Expertise (SEQUENTIAL mode)
> *"Write a Python microservice that monitors my portfolio in real-time and alerts me when it drops 5% in a day. Include risk analysis of the strategy."*

→ 📈 Finance → 💻 Code (finance analysis informs the code architecture)

### Triple Collaboration (PARALLEL mode)
> *"I'm building a mental health app for Gen Z. Analyze the market opportunity, suggest the tech stack, and write the investor pitch narrative."*

→ 📈 Finance + 💻 Code + 🎨 Creative → Synthesized

---

## 📁 Project Structure

```
MistralMind/
├── agent/
│   └── router.py                ← Routing + dispatch + synthesis
├── finetune/
│   ├── train_specialist.py      ← Unsloth + TRL + W&B training
│   └── evaluate_specialists.py  ← Before/after benchmarking
├── demo/
│   └── app.py                   ← Gradio UI with routing visualization
├── .env.example                 ← Environment variable template
├── .gitignore                   ← Excludes .env, checkpoints, data
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Fast fine-tuning | **Unsloth** |
| Training framework | **TRL** (SFTTrainer) |
| Experiment tracking | **Weights & Biases** |
| Routing & synthesis | **Mistral API** |
| Demo UI | **Gradio** |
| Base model | **Mistral-7B-Instruct-v0.3** |

---

*Built for the Mistral Global Hackathon 2026 — because one expert is never enough.*
