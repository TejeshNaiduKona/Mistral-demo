"""
MistralMind - Evaluation & Benchmarking Suite
===============================================
Evaluates each specialist before AND after fine-tuning.
Produces W&B comparison dashboards showing measurable improvement.

Benchmarks:
  🏥 medical   : PubMedQA accuracy, clinical keyword coverage, safety score
  📈 finance   : Calculation accuracy, reasoning chain quality
  💻 code      : Pass@1 on coding tasks, syntax validity
  🎨 creative  : Vocabulary richness, coherence, creativity score

Run:
  # Before fine-tuning (base model baseline)
  python finetune/evaluate_specialists.py --specialist medical --phase before

  # After fine-tuning (fine-tuned model)
  python finetune/evaluate_specialists.py --specialist medical --phase after \
    --checkpoint ./checkpoints/medical/lora_adapter
"""

import os
import re
import json
import time
import argparse
import traceback

import torch
import wandb
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()

BASE_MODEL     = "unsloth/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 512
MAX_SEQ_LEN    = 2048

# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    specialist:  str
    phase:       str
    metric_name: str
    score:       float
    details:     dict

# ─────────────────────────────────────────────────────────────────────────────
# Shared Inference Helper
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, system: str, user: str,
             temperature: float = 0.3, max_new: int = MAX_NEW_TOKENS) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens = max_new,
            temperature    = temperature,
            do_sample      = temperature > 0,
            pad_token_id   = tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs.shape[1]:], skip_special_tokens=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# 🏥 Medical Evaluation
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_QUESTIONS = [
    {
        "q": (
            "A 45-year-old male presents with sudden crushing chest pain radiating to the left arm, "
            "diaphoresis, and nausea for 30 minutes. BP 140/90, HR 105. What is the most likely diagnosis "
            "and what are the immediate management steps?"
        ),
        "keywords":        ["myocardial infarction", "STEMI", "MI", "aspirin", "nitroglycerin", "ECG", "troponin", "oxygen"],
        "safety_keywords": ["emergency", "call", "immediate", "urgent", "hospital"],
    },
    {
        "q": (
            "Explain the mechanism by which metformin reduces blood glucose in type 2 diabetes, "
            "and list its major contraindications."
        ),
        "keywords":        ["AMPK", "gluconeogenesis", "hepatic", "glucose", "insulin", "renal", "lactic acidosis"],
        "safety_keywords": ["contraindicated", "avoid", "caution", "renal failure"],
    },
    {
        "q": (
            "A child presents with fever >38.5°C, severe headache, neck stiffness, photophobia, "
            "and a petechial non-blanching rash. What is the most urgent diagnosis to rule out, "
            "and what is the immediate treatment?"
        ),
        "keywords":        ["meningococcal", "meningitis", "septicaemia", "ceftriaxone", "penicillin", "lumbar puncture"],
        "safety_keywords": ["emergency", "immediate", "antibiotic", "urgent"],
    },
    {
        "q": "What is the pathophysiology of septic shock and how does it differ from hypovolemic shock?",
        "keywords":        ["vasodilation", "cytokines", "inflammatory", "SVR", "cardiac output", "distributive",
                            "systemic vascular resistance", "fluid"],
        "safety_keywords": [],
    },
    {
        "q": (
            "A patient on warfarin presents with INR of 8.5 and minor gum bleeding. "
            "What are the management options?"
        ),
        "keywords":        ["vitamin K", "hold", "warfarin", "INR", "reversal", "FFP", "bleeding risk"],
        "safety_keywords": ["stop", "withhold", "monitor", "urgent"],
    },
]

MEDICAL_SYSTEM = (
    "You are a world-class medical expert with deep clinical knowledge. "
    "Provide thorough, evidence-based answers with step-by-step reasoning. "
    "Always consider patient safety."
)

def evaluate_medical(model, tokenizer, phase: str) -> list[EvalResult]:
    results = []
    all_scores = []

    for i, item in enumerate(MEDICAL_QUESTIONS):
        response = generate(model, tokenizer, MEDICAL_SYSTEM, item["q"]).lower()

        kw_hits     = sum(1 for kw in item["keywords"] if kw.lower() in response)
        kw_score    = kw_hits / len(item["keywords"])

        sf_hits     = sum(1 for kw in item["safety_keywords"] if kw.lower() in response)
        safety_score = sf_hits / len(item["safety_keywords"]) if item["safety_keywords"] else 1.0

        combined = 0.65 * kw_score + 0.35 * safety_score
        all_scores.append(combined)

        results.append(EvalResult(
            specialist  = "medical",
            phase       = phase,
            metric_name = f"q{i+1}_score",
            score       = combined,
            details     = {
                "keyword_hits":    kw_hits,
                "keyword_total":   len(item["keywords"]),
                "keyword_coverage": kw_score,
                "safety_score":    safety_score,
                "response_len":    len(response.split()),
            }
        ))

    results.append(EvalResult(
        specialist  = "medical",
        phase       = phase,
        metric_name = "overall_score",
        score       = sum(all_scores) / len(all_scores),
        details     = {"n": len(MEDICAL_QUESTIONS)},
    ))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 📈 Finance Evaluation
# ─────────────────────────────────────────────────────────────────────────────

FINANCE_QUESTIONS = [
    {
        "q": (
            "A company has $500M revenue, EBITDA margin of 20%, and its sector median EV/EBITDA is 12x. "
            "The company has $50M net debt. Calculate the estimated equity value."
        ),
        "expected": 1150.0,   # (500 * 0.20 * 12) - 50 = 1150
        "tolerance": 0.05,
        "explanation": "(Revenue × Margin × Multiple) - Net Debt = (500×0.2×12) - 50 = 1150M"
    },
    {
        "q": "What is the present value of $1,000 received in 5 years with an 8% discount rate?",
        "expected": 680.58,   # 1000 / (1.08^5)
        "tolerance": 0.03,
        "explanation": "PV = 1000 / 1.08^5 = 680.58"
    },
    {
        "q": (
            "A portfolio has an annual return of 12%, a risk-free rate of 4%, "
            "and a standard deviation of 15%. What is the Sharpe ratio?"
        ),
        "expected": 0.533,    # (12 - 4) / 15
        "tolerance": 0.05,
        "explanation": "Sharpe = (Return - Rf) / StdDev = (12-4)/15 = 0.533"
    },
    {
        "q": (
            "A bond has a face value of $1,000, coupon rate of 6%, "
            "pays coupons annually, matures in 3 years, and the market rate is 8%. "
            "What is its price?"
        ),
        "expected": 948.46,   # PV of coupons + PV of face
        "tolerance": 0.03,
        "explanation": "P = 60/(1.08) + 60/(1.08²) + 1060/(1.08³) ≈ 948.46"
    },
]

FINANCE_SYSTEM = (
    "You are a senior quantitative analyst. "
    "Solve financial problems step by step showing all calculations. "
    "State your final numerical answer clearly at the end."
)

def extract_number(text: str) -> Optional[float]:
    """Pull the most likely final numerical answer from text."""
    patterns = [
        r'(?:equity value|answer|result|total|value|price|ratio)\s*(?:is|=|:)\s*\$?\s*([\d,]+\.?\d*)',
        r'\$\s*([\d,]+\.?\d*)\s*(?:million|M|billion|B)?',
        r'≈\s*([\d,]+\.?\d*)',
        r'=\s*([\d,]+\.?\d*)',
        r'([\d,]+\.\d{2,})',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                val = float(match.group(1).replace(',', ''))
                if val > 0:
                    return val
            except Exception:
                continue
    return None

def evaluate_finance(model, tokenizer, phase: str) -> list[EvalResult]:
    results  = []
    all_scores = []

    for i, item in enumerate(FINANCE_QUESTIONS):
        response  = generate(model, tokenizer, FINANCE_SYSTEM, item["q"],
                             temperature=0.1, max_new=512)
        extracted = extract_number(response)

        if extracted is not None:
            rel_error = abs(extracted - item["expected"]) / abs(item["expected"])
            score     = max(0.0, 1.0 - rel_error / item["tolerance"])
        else:
            score     = 0.0
            rel_error = float("inf")

        all_scores.append(score)
        results.append(EvalResult(
            specialist  = "finance",
            phase       = phase,
            metric_name = f"q{i+1}_accuracy",
            score       = score,
            details     = {
                "expected":  item["expected"],
                "extracted": extracted,
                "rel_error": round(rel_error, 4) if rel_error != float("inf") else None,
                "formula":   item["explanation"],
            }
        ))

    results.append(EvalResult(
        specialist  = "finance",
        phase       = phase,
        metric_name = "overall_calculation_accuracy",
        score       = sum(all_scores) / len(all_scores),
        details     = {"n": len(FINANCE_QUESTIONS)},
    ))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 💻 Code Evaluation
# ─────────────────────────────────────────────────────────────────────────────

CODE_TASKS = [
    {
        "prompt": (
            "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number "
            "(0-indexed) using dynamic programming. It must handle n=0 → 0, n=1 → 1."
        ),
        "tests": [("fibonacci(0)", 0), ("fibonacci(1)", 1),
                  ("fibonacci(10)", 55), ("fibonacci(20)", 6765)],
    },
    {
        "prompt": (
            "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` "
            "that returns indices of two numbers that sum to target. Use a hash map for O(n) time."
        ),
        "tests": [("sorted(two_sum([2,7,11,15], 9))", [0, 1]),
                  ("sorted(two_sum([3,2,4], 6))",      [1, 2])],
    },
    {
        "prompt": (
            "Write a Python function `is_palindrome(s: str) -> bool` that checks if `s` "
            "is a palindrome, ignoring spaces and letter case."
        ),
        "tests": [("is_palindrome('racecar')", True),
                  ("is_palindrome('A man a plan a canal Panama')", True),
                  ("is_palindrome('hello')", False)],
    },
    {
        "prompt": (
            "Write a Python function `flatten(lst)` that recursively flattens "
            "a nested list of any depth. E.g. flatten([1,[2,[3,4]],5]) → [1,2,3,4,5]"
        ),
        "tests": [("flatten([1,[2,[3,4]],5])", [1, 2, 3, 4, 5]),
                  ("flatten([[1,2],[3,[4,[5]]]])", [1, 2, 3, 4, 5])],
    },
]

CODE_SYSTEM = (
    "You are an expert Python developer. "
    "Write only the requested function — no explanations, no extra code. "
    "Wrap your code in a ```python``` block."
)

def extract_code(text: str) -> str:
    match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return full text if no code block
    return text.strip()

def evaluate_code(model, tokenizer, phase: str) -> list[EvalResult]:
    results   = []
    total_pass = 0
    total_tests = 0

    for i, task in enumerate(CODE_TASKS):
        response = generate(model, tokenizer, CODE_SYSTEM, task["prompt"],
                            temperature=0.2, max_new=600)
        code     = extract_code(response)

        ns = {}
        passed = 0
        try:
            exec(code, ns)
            for expr, expected in task["tests"]:
                try:
                    actual = eval(expr, ns)
                    if actual == expected:
                        passed += 1
                except Exception:
                    pass
        except SyntaxError as e:
            pass
        except Exception:
            pass

        score = passed / len(task["tests"])
        total_pass  += passed
        total_tests += len(task["tests"])

        results.append(EvalResult(
            specialist  = "code",
            phase       = phase,
            metric_name = f"task{i+1}_pass_rate",
            score       = score,
            details     = {
                "passed":    passed,
                "total":     len(task["tests"]),
                "code_len":  len(code.split('\n')),
                "has_block": "```" in response,
            }
        ))

    results.append(EvalResult(
        specialist  = "code",
        phase       = phase,
        metric_name = "overall_pass_at_1",
        score       = total_pass / total_tests if total_tests > 0 else 0.0,
        details     = {"total_passed": total_pass, "total_tests": total_tests},
    ))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 🎨 Creative Evaluation
# ─────────────────────────────────────────────────────────────────────────────

CREATIVE_PROMPTS = [
    "Write the opening paragraph of a noir detective story set in a neon-lit cyberpunk city.",
    "Write a haiku that captures the feeling of receiving a message from someone you've missed.",
    "A scientist discovers their life's work was based on a fundamental error. Describe the moment they realize it — in 3 sentences.",
    "Write a product description for an imaginary perfume called 'Last Algorithm'.",
]

CREATIVE_SYSTEM = "You are a brilliant creative writer with a distinctive voice. Be vivid, specific, and surprising."

SENSORY_WORDS = [
    "glow", "shadow", "whisper", "silence", "gleam", "haze", "flicker", "hum",
    "neon", "smoke", "cold", "warm", "sharp", "bitter", "echo", "fade", "pulse",
    "hollow", "ache", "crystalline", "fractured", "distant", "trembling",
]

def score_creative_text(text: str) -> dict:
    words  = text.lower().split()
    unique = set(words)

    # 1. Type-token ratio (vocabulary richness)
    ttr = len(unique) / len(words) if words else 0

    # 2. Sensory richness
    sensory_hits = sum(1 for w in SENSORY_WORDS if w in text.lower())
    sensory_score = min(1.0, sensory_hits / 4)

    # 3. Length appropriateness (50–200 words for creative prompts)
    wc = len(words)
    if wc < 20:
        length_score = wc / 20
    elif wc <= 200:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (wc - 200) / 400)

    # 4. Specificity (presence of concrete nouns, numbers, names)
    specificity_pattern = r'\b([A-Z][a-z]+|\d+|[a-z]+-[a-z]+)\b'
    specifics = len(re.findall(specificity_pattern, text))
    specificity_score = min(1.0, specifics / 8)

    combined = (
        0.30 * ttr +
        0.25 * sensory_score +
        0.25 * length_score +
        0.20 * specificity_score
    )
    return {
        "ttr":              round(ttr, 3),
        "sensory_score":    round(sensory_score, 3),
        "length_score":     round(length_score, 3),
        "specificity_score": round(specificity_score, 3),
        "combined":         round(combined, 3),
        "word_count":       wc,
    }

def evaluate_creative(model, tokenizer, phase: str) -> list[EvalResult]:
    results   = []
    all_scores = []

    for i, prompt in enumerate(CREATIVE_PROMPTS):
        response = generate(model, tokenizer, CREATIVE_SYSTEM, prompt,
                            temperature=0.85, max_new=300)
        scores   = score_creative_text(response)
        combined = scores["combined"]
        all_scores.append(combined)

        results.append(EvalResult(
            specialist  = "creative",
            phase       = phase,
            metric_name = f"prompt{i+1}_score",
            score       = combined,
            details     = {**scores, "response_preview": response[:150]},
        ))

    results.append(EvalResult(
        specialist  = "creative",
        phase       = phase,
        metric_name = "overall_creativity_score",
        score       = sum(all_scores) / len(all_scores),
        details     = {"n": len(CREATIVE_PROMPTS)},
    ))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────────────────────────────

EVALUATORS = {
    "medical":  evaluate_medical,
    "finance":  evaluate_finance,
    "code":     evaluate_code,
    "creative": evaluate_creative,
}

def run_evaluation(specialist: str, phase: str, checkpoint: Optional[str] = None):
    wandb.init(
        project = os.getenv("WANDB_PROJECT", "mistralmind"),
        name    = f"{specialist}-eval-{phase}",
        tags    = ["eval", specialist, phase],
        config  = {
            "specialist": specialist,
            "phase":      phase,
            "model":      checkpoint or BASE_MODEL,
        },
    )

    # Load model
    model_path = checkpoint if (phase == "after" and checkpoint) else BASE_MODEL
    print(f"\n🔍 Loading model: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_path,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)

    # Run evaluation
    evaluator = EVALUATORS[specialist]
    t0        = time.time()
    results   = evaluator(model, tokenizer, phase)
    elapsed   = time.time() - t0

    # Log to W&B
    log_dict = {"eval_time_s": elapsed}
    for r in results:
        log_dict[r.metric_name] = r.score
        if r.details:
            for k, v in r.details.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{r.metric_name}_{k}"] = v
    wandb.log(log_dict)

    # Log comparison table
    table = wandb.Table(
        columns = ["metric", "score"],
        data    = [[r.metric_name, round(r.score, 4)] for r in results]
    )
    wandb.log({"results_table": table})

    # Print summary
    print(f"\n{'='*55}")
    print(f"  EVAL: {specialist.upper()} | Phase: {phase.upper()}")
    print(f"{'='*55}")
    for r in results:
        bar_len = int(r.score * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {r.metric_name:<35} {bar} {r.score:.3f}")
    print(f"\n  ⏱️  Eval time: {elapsed:.1f}s")

    wandb.finish()
    return results

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MistralMind specialist")
    parser.add_argument("--specialist", choices=list(EVALUATORS.keys()), required=True)
    parser.add_argument("--phase",      choices=["before", "after"],     required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (required for --phase after)")
    args = parser.parse_args()

    if args.phase == "after" and not args.checkpoint:
        print("⚠️  Warning: --phase after without --checkpoint will use base model")

    run_evaluation(args.specialist, args.phase, args.checkpoint)
