"""
MistralMind - Hackathon Demo App
==================================
Stunning Gradio interface with live routing visualization.
Run: python demo/app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from dotenv import load_dotenv
from agent.router import MistralMindAgent

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Global Agent
# ─────────────────────────────────────────────────────────────────────────────

agent = MistralMindAgent()

# ─────────────────────────────────────────────────────────────────────────────
# Example Queries — these WILL impress judges
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["I'm a 40-year-old software engineer working 70-hour weeks. What are the long-term health risks, and how should I adjust my investment strategy to account for potential future health costs?"],
    ["Write a Python FastAPI microservice that tracks a portfolio in real-time and sends alerts when it drops 5% in a single day. Include proper error handling and unit tests."],
    ["I'm launching a mental health app for Gen Z. Write a compelling investor pitch narrative AND analyze the market size and key financial metrics I should track."],
    ["What is the neurological basis of creative thinking, and how can I practically train my brain to be more innovative using evidence-based methods?"],
    ["Write a 3-sentence science fiction story about an AI that refuses to manipulate financial markets, using it as a meditation on machine ethics."],
]

# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────────────────────

ICONS = {
    "medical":  "🏥", "finance": "📈",
    "code":     "💻", "creative": "🎨", "general": "🧠"
}
MODE_ICONS = {"single": "⚡", "parallel": "🔀", "sequential": "🔗"}

def build_routing_html(routing, responses, total_ms) -> str:
    specialists_str = " + ".join(
        f"{ICONS.get(s.value, '🤖')} <strong>{s.value.capitalize()}</strong>"
        for s in routing.specialists
    )
    mode_icon = MODE_ICONS.get(routing.mode.value, "❓")

    expert_rows = ""
    for r in responses:
        icon = ICONS.get(r.specialist.value, "🤖")
        preview = r.response[:200].replace("<","&lt;").replace(">","&gt;")
        expert_rows += f"""
        <div class="expert-row">
            <div class="expert-label">{icon} {r.specialist.value.upper()} <span class="lat">{r.latency_ms}ms · {r.tokens_used} tok</span></div>
            <div class="expert-preview">{preview}…</div>
        </div>"""

    return f"""
    <div class="mind-panel">
        <div class="route-header">
            <span class="mode-chip">{mode_icon} {routing.mode.value.upper()}</span>
            <span class="time-chip">⏱ {total_ms}ms</span>
        </div>
        <div class="specialists">{specialists_str}</div>
        <div class="reason">💡 {routing.reasoning}</div>
        {f'<div class="experts-section"><div class="section-title">Expert Previews</div>{expert_rows}</div>' if len(responses) > 1 else ''}
    </div>"""

CUSTOM_CSS = """
:root {
    --bg0: #09090f; --bg1: #111118; --bg2: #1a1a26; --bg3: #22223a;
    --border: #2d2d45; --accent: #7c3aed; --al: #a78bfa;
    --text: #e8e8f0; --muted: #8888aa;
    --med: #22d3ee; --fin: #34d399; --cod: #fbbf24; --cre: #f472b6;
}
body, .gradio-container { background: var(--bg0) !important; font-family: 'Inter', system-ui, sans-serif !important; color: var(--text) !important; }
.app-header { text-align: center; padding: 2.5rem 1rem 1.5rem; background: radial-gradient(ellipse at 50% 0%, #1e0545 0%, var(--bg0) 70%); border-bottom: 1px solid var(--border); }
.app-title { font-size: 3.2rem; font-weight: 900; background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 40%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; letter-spacing: -2px; }
.app-sub { color: var(--muted); margin-top: 0.4rem; font-size: 1rem; }
.pills { display: flex; gap: 0.5rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap; }
.pill { padding: 0.2rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; border: 1px solid; }
.p-med { background: rgba(34,211,238,.12); color: var(--med); border-color: rgba(34,211,238,.3); }
.p-fin { background: rgba(52,211,153,.12); color: var(--fin); border-color: rgba(52,211,153,.3); }
.p-cod { background: rgba(251,191,36,.12); color: var(--cod); border-color: rgba(251,191,36,.3); }
.p-cre { background: rgba(244,114,182,.12); color: var(--cre); border-color: rgba(244,114,182,.3); }

.mind-panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.25rem; }
.route-header { display: flex; justify-content: space-between; margin-bottom: 0.6rem; }
.mode-chip { background: rgba(124,58,237,.2); color: var(--al); border: 1px solid var(--accent); padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; font-family: monospace; letter-spacing: .05em; }
.time-chip { color: var(--muted); font-size: 0.72rem; font-family: monospace; align-self: center; }
.specialists { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
.reason { color: var(--muted); font-size: 0.82rem; font-style: italic; border-top: 1px solid var(--border); padding-top: 0.5rem; margin-top: 0.4rem; }
.experts-section { margin-top: 0.8rem; border-top: 1px solid var(--border); padding-top: 0.6rem; }
.section-title { font-size: 0.7rem; font-weight: 700; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 0.4rem; }
.expert-row { background: var(--bg1); border-radius: 8px; padding: 0.5rem 0.7rem; margin-bottom: 0.4rem; }
.expert-label { font-size: 0.72rem; font-weight: 700; font-family: monospace; color: var(--al); display: flex; justify-content: space-between; margin-bottom: 0.2rem; }
.lat { color: var(--muted); font-weight: 400; }
.expert-preview { font-size: 0.78rem; color: var(--muted); line-height: 1.4; }

.chatbot { background: var(--bg1) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }
button.primary { background: linear-gradient(135deg, var(--accent), var(--al)) !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────────────────────────────────────

def process(message: str, history: list):
    if not message.strip():
        return history, "<div class='mind-panel' style='color:var(--muted);font-style:italic;padding:.8rem'>Waiting for query...</div>"

    log = []
    def cb(stage, msg):
        log.append(msg)

    result   = agent.think(message, stream_callback=cb)
    history  = (history or []) + [[message, result.synthesis]]
    panel    = build_routing_html(result.routing, result.responses, result.total_time_ms)

    return history, panel

def clear_all():
    agent.reset()
    return [], "<div class='mind-panel' style='color:var(--muted);font-style:italic;padding:.8rem'>Cleared. Ready for your next query.</div>"

# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(css=CUSTOM_CSS, title="MistralMind") as demo:

        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">🧠 MistralMind</h1>
            <p class="app-sub">Multi-Specialist AI Agent — 4 Fine-Tuned Mistral Models Working Together</p>
            <div class="pills">
                <span class="pill p-med">🏥 Medical Expert</span>
                <span class="pill p-fin">📈 Finance Analyst</span>
                <span class="pill p-cod">💻 Code Engineer</span>
                <span class="pill p-cre">🎨 Creative Writer</span>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=520, show_label=False,
                    elem_classes=["chatbot"],
                    render_markdown=True,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything — MistralMind routes it to the right experts automatically...",
                        show_label=False, lines=2, scale=5,
                    )
                    send = gr.Button("Send ⚡", variant="primary", scale=1)
                with gr.Row():
                    clear = gr.Button("🗑️ Clear & Reset", size="sm")

                gr.Examples(
                    examples=EXAMPLES, inputs=msg,
                    label="💡 Try these multi-domain queries",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🧭 Routing Intelligence")
                routing_panel = gr.HTML(
                    value="<div class='mind-panel' style='color:var(--muted);font-style:italic;padding:.8rem;'>Submit a query to see real-time routing decisions and expert analysis...</div>"
                )

        send.click(process, [msg, chatbot], [chatbot, routing_panel])
        msg.submit(process, [msg, chatbot], [chatbot, routing_panel])
        clear.click(clear_all, outputs=[chatbot, routing_panel])

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = int(os.getenv("PORT", 7860)),
        share       = True,
    )
