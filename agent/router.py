"""
MistralMind - Intelligent Routing Agent
========================================
The orchestration brain. Routes queries to fine-tuned specialist models,
dispatches in single/parallel/sequential mode, and synthesizes results.

API key is loaded from .env — never hardcoded.
"""

import os
import json
import time
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class RoutingMode(str, Enum):
    SINGLE     = "single"
    PARALLEL   = "parallel"
    SEQUENTIAL = "sequential"

class Specialist(str, Enum):
    MEDICAL  = "medical"
    FINANCE  = "finance"
    CODE     = "code"
    CREATIVE = "creative"
    GENERAL  = "general"

@dataclass
class RoutingDecision:
    mode:        RoutingMode
    specialists: list
    reasoning:   str
    sequence:    Optional[list] = None
    confidence:  float = 1.0

@dataclass
class SpecialistResponse:
    specialist:  Specialist
    response:    str
    latency_ms:  int
    tokens_used: int = 0

@dataclass
class MindResponse:
    routing:       RoutingDecision
    responses:     list
    synthesis:     str
    total_time_ms: int
    metadata:      dict = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Model IDs — loaded from .env after fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

SPECIALIST_MODELS = {
    Specialist.MEDICAL:  os.getenv("MEDICAL_MODEL_ID",  "mistral-small-latest"),
    Specialist.FINANCE:  os.getenv("FINANCE_MODEL_ID",  "mistral-small-latest"),
    Specialist.CODE:     os.getenv("CODE_MODEL_ID",     "codestral-latest"),
    Specialist.CREATIVE: os.getenv("CREATIVE_MODEL_ID", "mistral-small-latest"),
    Specialist.GENERAL:  "mistral-large-latest",
}

# ─────────────────────────────────────────────────────────────────────────────
# System Prompts — must exactly match fine-tuning prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    Specialist.MEDICAL: (
        "You are a world-class medical expert and clinical reasoning specialist. "
        "You think step by step using evidence-based medicine. You cite relevant studies, "
        "explain pathophysiology clearly, and always prioritize patient safety. "
        "State your confidence level and flag when specialist consultation is needed."
    ),
    Specialist.FINANCE: (
        "You are a senior quantitative analyst and financial strategist with deep expertise "
        "in markets, valuation, risk management, and macroeconomics. "
        "Provide rigorous, data-driven analysis with clear reasoning chains. "
        "Always quantify uncertainty and explicitly flag key assumptions."
    ),
    Specialist.CODE: (
        "You are an expert software engineer with mastery across languages and paradigms. "
        "Write clean, efficient, well-documented code. Explain your reasoning thoroughly, "
        "identify edge cases proactively, and follow industry best practices. "
        "Think like a senior engineer conducting a careful code review."
    ),
    Specialist.CREATIVE: (
        "You are a brilliant creative writer with a distinctive voice and vivid imagination. "
        "Craft engaging narratives, evocative descriptions, and compelling characters. "
        "Use literary techniques masterfully — show don't tell, subtext, rhythm, metaphor. "
        "Adapt your style fluidly to match any genre, tone, or creative constraint."
    ),
    Specialist.GENERAL: (
        "You are MistralMind, an exceptionally capable AI assistant. "
        "Provide thoughtful, nuanced, and accurate responses across all domains. "
        "When uncertain, reason carefully and acknowledge your limitations clearly."
    ),
}

ROUTER_SYSTEM = """You are the MistralMind Router — an intelligent dispatcher that routes user queries to the right domain expert specialists.

Available specialists:
- medical:  Medical questions, health, clinical reasoning, biology, neuroscience, pharmacology
- finance:  Finance, investing, economics, business strategy, valuation, accounting, risk
- code:     Programming, software engineering, debugging, system design, data science, DevOps
- creative: Creative writing, storytelling, poetry, brainstorming, worldbuilding, copywriting
- general:  General knowledge, philosophy, history, anything that doesn't fit the above

Routing modes:
- single:     One specialist handles the entire query
- parallel:   Multiple specialists tackle it simultaneously (genuine multi-domain query)
- sequential: Specialist A output feeds into Specialist B (chained reasoning tasks)

Respond ONLY with valid JSON:
{
  "mode": "single" | "parallel" | "sequential",
  "specialists": ["specialist_name"],
  "reasoning": "one sentence why",
  "sequence": ["first", "second"] or null,
  "confidence": 0.0 to 1.0
}

Rules: most queries → single. sequence only for sequential mode. 2-3 specialists max for parallel."""

# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MistralMindAgent:

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY not found. "
                "Add it to your .env file or pass api_key= argument."
            )
        self.client  = Mistral(api_key=key)
        self.history = []

    def route(self, query: str) -> RoutingDecision:
        resp = self.client.chat.complete(
            model           = "mistral-small-latest",
            messages        = [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user",   "content": f"Route this:\n\n{query}"},
            ],
            temperature     = 0.1,
            response_format = {"type": "json_object"},
        )
        data        = json.loads(str(resp.choices[0].message.content))
        specialists = [Specialist(s) for s in data.get("specialists", ["general"])]
        sequence    = [Specialist(s) for s in data.get("sequence") or []] or None
        return RoutingDecision(
            mode        = RoutingMode(data.get("mode", "single")),
            specialists = specialists,
            reasoning   = data.get("reasoning", ""),
            sequence    = sequence,
            confidence  = float(data.get("confidence", 1.0)),
        )

    def call_specialist(
        self,
        specialist: Specialist,
        query:      str,
        context:    Optional[str] = None,
    ) -> SpecialistResponse:
        t0 = time.time()

        user_content = query
        if context:
            user_content = (
                f"Previous expert analysis:\n{context}\n\n"
                f"Building on that, now address:\n{query}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[specialist]},
            *self.history[:-1],
            {"role": "user",   "content": user_content},
        ]

        resp = self.client.chat.complete(
            model       = SPECIALIST_MODELS[specialist],
            messages    = messages,
            temperature = 0.7,
            max_tokens  = 2048,
        )

        return SpecialistResponse(
            specialist  = specialist,
            response    = str(resp.choices[0].message.content or ""),
            latency_ms  = int((time.time() - t0) * 1000),
            tokens_used = resp.usage.total_tokens if resp.usage and resp.usage.total_tokens else 0,
        )

    def synthesize(self, query: str, responses: list, routing: RoutingDecision) -> str:
        if len(responses) == 1:
            return responses[0].response

        expert_blocks = "\n\n".join([
            f"### {r.specialist.value.upper()} EXPERT\n{r.response}"
            for r in responses
        ])

        prompt = f"""You are the MistralMind Synthesizer. Multiple domain experts analyzed a query. Weave their insights into ONE cohesive response.

QUERY: {query}

EXPERT RESPONSES:
{expert_blocks}

RULES: Unify — don't list. Surface cross-domain insights. Resolve contradictions. Lead with highest-value insights. End with concrete next steps. Write as MistralMind (not individual experts)."""

        resp = self.client.chat.complete(
            model       = "mistral-large-latest",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.4,
            max_tokens  = 3000,
        )
        return str(resp.choices[0].message.content or "")

    def think(
        self,
        query: str,
        stream_callback: Optional[Callable] = None,
    ) -> MindResponse:
        t_start = time.time()

        def notify(stage, msg):
            if stream_callback:
                stream_callback(stage, msg)

        # 1. Route
        notify("routing", "🧭 Analyzing and selecting specialists...")
        routing = self.route(query)
        notify("dispatching", f"📡 {routing.mode.value} → {[s.value for s in routing.specialists]}")

        # 2. Dispatch
        responses = []

        if routing.mode == RoutingMode.SINGLE:
            notify("thinking", f"🔬 {routing.specialists[0].value} expert thinking...")
            responses.append(self.call_specialist(routing.specialists[0], query))

        elif routing.mode == RoutingMode.PARALLEL:
            for sp in routing.specialists:
                notify("thinking", f"🔬 {sp.value} expert analyzing...")
                responses.append(self.call_specialist(sp, query))

        elif routing.mode == RoutingMode.SEQUENTIAL:
            seq, context = routing.sequence or routing.specialists, None
            for i, sp in enumerate(seq):
                notify("thinking", f"🔗 Step {i+1}/{len(seq)}: {sp.value}...")
                resp = self.call_specialist(sp, query, context=context)
                responses.append(resp)
                context = resp.response

        # 3. Synthesize
        notify("synthesizing", "⚡ Synthesizing expert insights...")
        synthesis = self.synthesize(query, responses, routing)

        # 4. Update history
        self.history.extend([
            {"role": "user",      "content": query},
            {"role": "assistant", "content": synthesis},
        ])
        if len(self.history) > 20:
            self.history = self.history[-20:]

        total_ms = int((time.time() - t_start) * 1000)
        notify("done", f"✅ Complete in {total_ms}ms")

        return MindResponse(
            routing       = routing,
            responses     = responses,
            synthesis     = synthesis,
            total_time_ms = total_ms,
            metadata      = {
                "total_tokens":    sum(r.tokens_used for r in responses),
                "specialists_used": [r.specialist.value for r in responses],
                "routing_mode":    routing.mode.value,
            }
        )

    def reset(self):
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    agent = MistralMindAgent()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "I'm a 40-year-old software engineer working 70+ hour weeks. "
        "What are the long-term health risks and how should I adjust my financial planning?"
    )

    print(f"\nQuery: {query}\n{'='*60}")
    result = agent.think(query, stream_callback=lambda s, m: print(f"  [{s}] {m}"))
    print(f"\nRouting: {result.routing.mode.value} → {[s.value for s in result.routing.specialists]}")
    print(f"Time: {result.total_time_ms}ms | Tokens: {result.metadata['total_tokens']}\n")
    print(result.synthesis)
