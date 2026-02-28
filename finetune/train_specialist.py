"""
MistralMind - Specialist Fine-Tuning Script
============================================
Fine-tunes Llama-3.2-3B-Instruct into domain specialists using Unsloth + TRL.
Each specialist is tracked independently with W&B.

Specialists & Datasets:
  🏥 medical   : qiaojin/PubMedQA
  📈 finance   : gbharti/finance-alpaca
  💻 code      : sahil2801/CodeAlpaca-20k
  🎨 creative  : Dahoas/synthetic-instruct-gptj-pairwise

Usage:
  python finetune/train_specialist.py --specialist medical  --epochs 3
  python finetune/train_specialist.py --specialist finance  --epochs 3
  python finetune/train_specialist.py --specialist code     --epochs 3
  python finetune/train_specialist.py --specialist creative --epochs 3
"""

import os
import argparse
import torch
import wandb
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL   = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN  = 2048
LOAD_IN_4BIT = True
LORA_RANK    = 32
LORA_ALPHA   = 64
OUTPUT_DIR   = "./checkpoints"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ─────────────────────────────────────────────────────────────────────────────
# Specialist Configurations
# ─────────────────────────────────────────────────────────────────────────────

SPECIALIST_CONFIG = {
    "medical": {
        "dataset_id":    "qiaojin/PubMedQA",
        "dataset_name":  "pqa_labeled",   # PubMedQA labeled subset (1,000 QA pairs)
        "split":         "train",
        "system_prompt": (
            "You are a world-class medical expert and clinical reasoning specialist. "
            "You think step by step using evidence-based medicine. You cite relevant studies, "
            "explain pathophysiology clearly, and always prioritize patient safety. "
            "State your confidence level and flag when specialist consultation is needed."
        ),
        "learning_rate": 2e-4,
    },
    "finance": {
        "dataset_id":   "gbharti/finance-alpaca",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are a senior quantitative analyst and financial strategist with deep expertise "
            "in markets, valuation, risk management, and macroeconomics. "
            "Provide rigorous, data-driven analysis with clear reasoning chains. "
            "Always quantify uncertainty and explicitly flag key assumptions."
        ),
        "learning_rate": 2e-4,
    },
    "code": {
        "dataset_id":   "sahil2801/CodeAlpaca-20k",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are an expert software engineer with mastery across languages and paradigms. "
            "Write clean, efficient, well-documented code. Explain your reasoning thoroughly, "
            "identify edge cases proactively, and follow industry best practices. "
            "Think like a senior engineer conducting a careful code review."
        ),
        "learning_rate": 2e-4,
    },
    "creative": {
        "dataset_id":   "Dahoas/synthetic-instruct-gptj-pairwise",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are a brilliant creative writer with a distinctive voice and vivid imagination. "
            "Craft engaging narratives, evocative descriptions, and compelling characters. "
            "Use literary techniques masterfully — show don't tell, subtext, rhythm, metaphor. "
            "Adapt your style fluidly to match any genre, tone, or creative constraint."
        ),
        "learning_rate": 1e-4,   # lower LR preserves natural creativity
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Formatters
# Each formatter maps raw dataset rows → {"instruction": ..., "response": ...}
# ─────────────────────────────────────────────────────────────────────────────

def format_pubmedqa(example):
    """
    PubMedQA schema:
      pubid, question, context (dict: labels, contexts, meshes), long_answer, final_decision
    """
    question = example.get("question", "")
    answer   = example.get("long_answer", "")

    # Include abstract context for richer reasoning
    context_dict = example.get("context", {})
    contexts     = context_dict.get("contexts", []) if isinstance(context_dict, dict) else []
    context_str  = " ".join(contexts[:3])  # first 3 abstracts max

    if context_str:
        instruction = (
            f"Research Context:\n{context_str}\n\n"
            f"Clinical Question:\n{question}"
        )
    else:
        instruction = question

    # Skip examples with empty answers
    if not answer or len(answer) < 20:
        return None

    return {"instruction": instruction, "response": answer}


def format_finance_alpaca(example):
    """
    finance-alpaca schema (Alpaca-style):
      instruction, input, output
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if inp:
        instruction = f"{instruction}\n\nContext: {inp}"

    if not instruction or not output or len(output) < 10:
        return None

    return {"instruction": instruction, "response": output}


def format_code_alpaca(example):
    """
    CodeAlpaca-20k schema (Alpaca-style):
      instruction, input, output
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if inp:
        instruction = f"{instruction}\n\n```\n{inp}\n```"

    if not instruction or not output or len(output) < 10:
        return None

    return {"instruction": instruction, "response": output}


def format_creative_pairwise(example):
    """
    Dahoas/synthetic-instruct-gptj-pairwise schema:
      prompt, chosen, rejected
    We use `chosen` as the preferred response.
    """
    prompt  = example.get("prompt", "").strip()
    chosen  = example.get("chosen", "").strip()

    if not prompt or not chosen or len(chosen) < 20:
        return None

    # Clean up the prompt (remove leading "Human:" artifacts if present)
    prompt = prompt.replace("Human:", "").replace("Assistant:", "").strip()

    return {"instruction": prompt, "response": chosen}


FORMATTERS = {
    "medical":  format_pubmedqa,
    "finance":  format_finance_alpaca,
    "code":     format_code_alpaca,
    "creative": format_creative_pairwise,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_specialist_dataset(specialist: str, max_samples: int = 10_000) -> Dataset:
    cfg       = SPECIALIST_CONFIG[specialist]
    formatter = FORMATTERS[specialist]

    print(f"  📥 Loading {cfg['dataset_id']} ...")

    load_kwargs = {
        "path":              cfg["dataset_id"],
        "split":             cfg["split"],
        "trust_remote_code": True,
    }
    if cfg["dataset_name"]:
        load_kwargs["name"] = cfg["dataset_name"]

    ds = load_dataset(**load_kwargs)

    # Apply formatter
    formatted = []
    for row in ds:
        result = formatter(row)
        if result is not None:
            formatted.append(result)

    ds_clean = Dataset.from_list(formatted)

    # Shuffle and cap
    ds_clean = ds_clean.shuffle(seed=42)
    if len(ds_clean) > max_samples:
        ds_clean = ds_clean.select(range(max_samples))

    print(f"  ✅ {specialist}: {len(ds_clean):,} samples ready")
    return ds_clean

# ─────────────────────────────────────────────────────────────────────────────
# Chat Formatter (converts to Mistral chat template)
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_formatter(tokenizer, system_prompt: str):
    def format_for_sft(example):
        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    return format_for_sft

# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_specialist(specialist: str, epochs: int = 3, max_samples: int = 10_000):
    cfg = SPECIALIST_CONFIG[specialist]

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project = os.getenv("WANDB_PROJECT", "mistralmind"),
        name    = f"{specialist}-specialist-v1",
        tags    = ["mistralmind", specialist, "unsloth", "qlora"],
        config  = {
            "specialist":    specialist,
            "dataset":       cfg["dataset_id"],
            "base_model":    BASE_MODEL,
            "epochs":        epochs,
            "max_seq_len":   MAX_SEQ_LEN,
            "lora_rank":     LORA_RANK,
            "lora_alpha":    LORA_ALPHA,
            "learning_rate": cfg["learning_rate"],
            "max_samples":   max_samples,
            "load_in_4bit":  LOAD_IN_4BIT,
        }
    )

    # ── Model (Unsloth) ───────────────────────────────────────────────────────
    print(f"\n🔧 Loading base model for [{specialist}]...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,          # auto-detect bf16/fp16
        load_in_4bit   = LOAD_IN_4BIT,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # ── LoRA (Unsloth PEFT) ───────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = TARGET_MODULES,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0.05,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",  # saves VRAM
        random_state               = 42,
        use_rslora                 = True,        # Rank-Stabilized LoRA
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n📚 Preparing dataset for [{specialist}]...")
    dataset   = load_specialist_dataset(specialist, max_samples)
    formatter = build_chat_formatter(tokenizer, cfg["system_prompt"])
    dataset   = dataset.map(formatter, remove_columns=dataset.column_names)

    split   = dataset.train_test_split(test_size=0.05, seed=42)
    train_d = split["train"]
    eval_d  = split["test"]

    print(f"   Train: {len(train_d):,} | Eval: {len(eval_d):,}")

    # ── Log sample to W&B ─────────────────────────────────────────────────────
    sample_table = wandb.Table(
        columns = ["text"],
        data    = [[train_d[i]["text"]] for i in range(min(5, len(train_d)))]
    )
    wandb.log({"training_samples": sample_table})

    # ── Training Args ─────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, specialist)
    training_args = TrainingArguments(
        output_dir                  = output_path,
        num_train_epochs            = epochs,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 8,        # effective batch = 16
        warmup_ratio                = 0.05,
        learning_rate               = cfg["learning_rate"],
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 200,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "wandb",
        optim                       = "adamw_8bit",
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        dataloader_num_workers      = 4,
        group_by_length             = True,     # reduces padding waste
    )

    # ── SFT Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_d,
        eval_dataset       = eval_d,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        args               = training_args,
        packing            = True,  # pack short sequences for efficiency
    )

    print(f"\n🚀 Training [{specialist}] specialist — {epochs} epochs...")
    trainer_stats = trainer.train()

    # ── Final W&B Metrics ─────────────────────────────────────────────────────
    wandb.log({
        "final_train_loss":      trainer_stats.training_loss,
        "total_steps":           trainer_stats.global_step,
        "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
        "samples_per_second":    trainer_stats.metrics["train_samples_per_second"],
    })

    # ── Save Checkpoints ──────────────────────────────────────────────────────
    lora_path   = os.path.join(output_path, "lora_adapter")
    merged_path = os.path.join(output_path, "merged_16bit")

    # Save LoRA adapter only (small, fast)
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"   💾 LoRA adapter → {lora_path}")

    # Save merged 16-bit model for deployment/upload
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"   💾 Merged model → {merged_path}")

    # Push to HuggingFace Hub (optional)
    hf_token    = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    if hf_token and hf_username:
        repo_id = f"{hf_username}/mistralmind-{specialist}"
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method  = "lora",
            token        = hf_token,
        )
        print(f"   📤 Pushed to HuggingFace → {repo_id}")
        wandb.log({"hf_model_url": f"https://huggingface.co/{repo_id}"})

    print(f"\n✅ [{specialist}] specialist training complete!")
    wandb.finish()
    return lora_path

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MistralMind specialist")
    parser.add_argument("--specialist", choices=list(SPECIALIST_CONFIG.keys()), required=True,
                        help="Which specialist to train")
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--max_samples", type=int, default=10_000,
                        help="Max training samples to use (default: 10000)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MistralMind — Training [{args.specialist.upper()}] Specialist")
    print(f"  Dataset: {SPECIALIST_CONFIG[args.specialist]['dataset_id']}")
    print(f"  Epochs:  {args.epochs} | Max samples: {args.max_samples:,}")
    print(f"{'='*60}")

    train_specialist(args.specialist, args.epochs, args.max_samples)
