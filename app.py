"""
app.py
======
AI Energy Predictor — Gradio Application

Estimates the energy consumption, cost, and carbon footprint of AI training
workloads described in plain English. Runs inference through three pre-trained
models simultaneously and visualises the results with Plotly charts.

Running locally:
    python app.py

Deploying to Hugging Face Spaces:
    huggingface-cli repo create ai-energy-predictor --type space --space_sdk gradio
    git push
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.build_features import StructuredFeatureExtractor
from scripts.model import NaiveBaseline, ClassicalMLModel, TransformerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR              = Path("models")
ELECTRICITY_USD_PER_KWH = 0.10    # Cloud GPU blended average
CO2_KG_PER_KWH         = 0.385   # US average grid carbon intensity (EPA 2023)

EXAMPLE_PROMPTS: List[str] = [
    "Train ResNet-50 on 4 A100 GPUs for 10 hours with batch size 128",
    "Fine-tune BERT-large on 2 V100 GPUs for 5 hours on a custom NLP dataset",
    "Run inference on GPT-2 using a single T4 GPU for 3 hours",
    "Pretrain LLaMA-7B across 8 H100 GPUs for 72 hours",
    "gonna train a ResNet for like 10h on 4 a100s",
    "Training ViT-Base on ImageNet with 8 A10G GPUs for 48hrs",
    "Fine-tuning Stable Diffusion on 4 RTX 4090s for 20 hours",
    "Whisper-medium evaluation on 1 T4 for 2 hours",
]

# Context comparisons ordered by CO₂ kg threshold
_CO2_COMPARISONS: List[Tuple[float, str]] = sorted([
    (0.404,  "driving 1 mile"),
    (2.5,    "charging a smartphone 300 times"),
    (6.3,    "streaming video for 1 hour (server side)"),
    (21.0,   "burning 1 gallon of gasoline"),
    (500.0,  "one round-trip flight NYC → LA"),
    (2400.0, "one transatlantic flight"),
], key=lambda x: x[0], reverse=True)

CUSTOM_CSS = """
.gradio-container { max-width: 1140px !important; margin: auto !important; }

/* Header */
#header-md h1 { font-size: 2.4rem !important; margin-bottom: 0.2rem !important; }
#header-md p  { font-size: 1rem !important; opacity: 0.75; margin-top: 0 !important; }

/* Predict button */
#predict-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    height: 52px !important;
    transition: opacity 0.15s ease;
}
#predict-btn:hover { opacity: 0.88 !important; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-top: 8px; }
.metric-box {
    flex: 1;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
}

/* Example buttons */
.example-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
"""

THEME = gr.themes.Soft(
    primary_hue   ="violet",
    secondary_hue ="indigo",
    neutral_hue   ="slate",
    font          =gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill          = "linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_hover    = "linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_text_color               = "white",
    block_radius                            = "12px",
    input_radius                            = "10px",
)


# ---------------------------------------------------------------------------
# Model registry  (lazy loading)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Loads all three model tiers from disk on first call.
    Models that fail to load are silently skipped so the app degrades
    gracefully (e.g., when the transformer weights are not yet trained).
    """

    def __init__(self, model_dir: Path):
        self._dir          = model_dir
        self._naive:       Optional[NaiveBaseline]    = None
        self._classical:   Optional[ClassicalMLModel] = None
        self._transformer: Optional[TransformerModel] = None
        self._loaded       = False

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading model artefacts …")

        for name, cls, subdir, attr in [
            ("Naive Baseline",   NaiveBaseline,    "naive",       "_naive"),
            ("Classical ML",     ClassicalMLModel, "classical",   "_classical"),
            ("DistilBERT",       TransformerModel, "transformer", "_transformer"),
        ]:
            path = self._dir / subdir
            try:
                setattr(self, attr, cls.load(path))
                logger.info(f"  ✓ {name} loaded from {path}")
            except Exception as exc:
                logger.warning(f"  ✗ Could not load {name}: {exc}")

        self._loaded = True

    def predict_all(self, text: str) -> Dict[str, float]:
        """Run inference on all available models; return name → kWh dict."""
        self._load()
        results: Dict[str, float] = {}
        if self._naive:
            results["Naive Baseline"] = round(self._naive.predict_single(text), 4)
        if self._classical:
            results["Classical ML (RF)"] = round(self._classical.predict_single(text), 4)
        if self._transformer:
            results["DistilBERT"] = round(self._transformer.predict_single(text), 4)
        return results

    @property
    def models_available(self) -> List[str]:
        self._load()
        return (
            (["Naive Baseline"]    if self._naive       else []) +
            (["Classical ML (RF)"] if self._classical   else []) +
            (["DistilBERT"]        if self._transformer else [])
        )


_registry  = ModelRegistry(MODEL_DIR)
_extractor = StructuredFeatureExtractor()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _co2_context(co2_kg: float) -> str:
    """Return a human-readable CO₂ equivalence string."""
    for threshold, label in _CO2_COMPARISONS:
        if co2_kg >= threshold:
            multiple = co2_kg / threshold
            return f"≈ {multiple:.1f}× {label}"
    grams = co2_kg * 1000
    return f"≈ {grams:.0f} g CO₂e — lighter than a cup of coffee"


def _energy_label(kwh: float) -> str:
    """Return a qualitative energy tier label."""
    if kwh < 1:
        return "🟢 Very Low"
    if kwh < 10:
        return "🟡 Low"
    if kwh < 50:
        return "🟠 Moderate"
    if kwh < 200:
        return "🔴 High"
    return "🚨 Very High"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _make_gauge(energy_kwh: float) -> go.Figure:
    """Gauge chart showing energy intensity against five qualitative tiers."""
    # Map energy to a 0–100 normalised scale (log10 compressed)
    norm = min(np.log10(max(energy_kwh, 0.01) + 1) / np.log10(501) * 100, 100)

    COLOR_STOPS = [
        {"range": [0,  20],  "color": "rgba(34,197,94,0.20)"},
        {"range": [20, 40],  "color": "rgba(132,204,22,0.20)"},
        {"range": [40, 60],  "color": "rgba(245,158,11,0.20)"},
        {"range": [60, 80],  "color": "rgba(239,68,68,0.20)"},
        {"range": [80, 100], "color": "rgba(127,29,29,0.25)"},
    ]
    bar_color = (
        "#22c55e" if norm < 20 else
        "#84cc16" if norm < 40 else
        "#f59e0b" if norm < 60 else
        "#ef4444" if norm < 80 else
        "#7f1d1d"
    )

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = energy_kwh,
        number = {"suffix": " kWh", "font": {"size": 26, "color": "white"}, "valueformat": ".2f"},
        title  = {"text": f"⚡ Energy Usage  ({_energy_label(energy_kwh)})", "font": {"color": "white", "size": 13}},
        gauge  = {
            "axis":        {"range": [0, 100], "visible": False},
            "bar":         {"color": bar_color, "thickness": 0.65},
            "bgcolor":     "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.15)",
            "steps":       COLOR_STOPS,
        },
        delta  = {"reference": 0},
        domain = {"x": [0, 1], "y": [0, 1]},
    ))
    # Annotate tier labels on the gauge arc
    for label, pos in [("Low", 10), ("Moderate", 50), ("High", 90)]:
        theta = np.radians(180 - pos * 1.8)
        r = 0.78
        fig.add_annotation(
            x=0.5 + r * np.cos(theta) * 0.5, y=0.12 + r * np.sin(theta) * 0.5,
            text=label, showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,0.45)"),
            xref="paper", yref="paper",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=230,
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


def _make_bar_chart(predictions: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart comparing predictions across all three models."""
    names  = list(predictions.keys())
    values = list(predictions.values())
    colors = ["#64748b", "#6366f1", "#22c55e"][:len(names)]

    fig = go.Figure(go.Bar(
        x             = values,
        y             = names,
        orientation   = "h",
        marker        = dict(color=colors),
        text          = [f"{v:.3f} kWh" for v in values],
        textposition  = "outside",
        textfont      = dict(color="white", size=12),
        cliponaxis    = False,
    ))
    max_val = max(values) * 1.35 if values else 1

    fig.update_layout(
        title         = dict(text="Model Comparison", font=dict(color="white", size=14)),
        xaxis         = dict(title="Predicted Energy (kWh)", color="white",
                             gridcolor="rgba(255,255,255,0.08)", range=[0, max_val]),
        yaxis         = dict(color="white", tickfont=dict(size=12)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="white"),
        height        = 210,
        margin        = dict(l=10, r=80, t=40, b=40),
    )
    return fig


def _make_feature_radar(extracted: Dict[str, float]) -> go.Figure:
    """Spider chart showing the normalised extracted features."""
    labels = ["GPUs", "Hours", "GPU Watts", "Task Util.", "Model Factor"]
    # Normalise each feature to [0, 1] against known max values
    maxes  = [16, 168, 700, 1.0, 7.0]
    vals   = [
        extracted["num_gpus"]        / maxes[0],
        extracted["hours"]           / maxes[1],
        extracted["gpu_watts"]       / maxes[2],
        extracted["task_multiplier"] / maxes[3],
        extracted["model_factor"]    / maxes[4],
    ]
    vals  += [vals[0]]   # close the polygon
    labels += [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r     = vals,
        theta = labels,
        fill  = "toself",
        fillcolor = "rgba(99,102,241,0.25)",
        line  = dict(color="#818cf8", width=2),
        marker= dict(size=6, color="#818cf8"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor      = "rgba(0,0,0,0)",
            radialaxis   = dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.12)",
                                tickfont=dict(color="rgba(255,255,255,0.3)"), tickvals=[0.25, 0.5, 0.75, 1.0]),
            angularaxis  = dict(tickfont=dict(color="white", size=11), gridcolor="rgba(255,255,255,0.12)"),
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color="white"),
        title         = dict(text="Extracted Features", font=dict(color="white", size=13)),
        height        = 230,
        margin        = dict(l=30, r=30, t=50, b=10),
        showlegend    = False,
    )
    return fig


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict(text: str) -> Tuple:
    """
    Run all three models on the input text and return Gradio outputs.

    Outputs (in order matching .click() outputs list):
        energy_md, cost_md, co2_md, context_md,
        gauge_fig, bar_fig, radar_fig,
        comparison_df, extracted_md, status_md
    """
    # ── Guard: empty input ────────────────────────────────────────────────
    if not text or not text.strip():
        empty = (
            "—", "—", "—", "—",
            go.Figure(), go.Figure(), go.Figure(),
            pd.DataFrame(columns=["Model", "Energy (kWh)", "Cost ($)", "CO₂ (kg)", "Tier"]),
            "No description provided.",
            "⚠️ Please enter a workload description above.",
        )
        return empty

    text = text.strip()

    # ── Get predictions ───────────────────────────────────────────────────
    predictions = _registry.predict_all(text)

    if not predictions:
        msg = ("❌ No models loaded. Run `python setup.py` first to generate "
               "data and train models.")
        return ("—", "—", "—", "—",
                go.Figure(), go.Figure(), go.Figure(),
                pd.DataFrame(), "—", msg)

    # ── Primary estimate (best available model) ───────────────────────────
    primary_kwh = (
        predictions.get("DistilBERT") or
        predictions.get("Classical ML (RF)") or
        predictions.get("Naive Baseline")
    )
    cost_usd = primary_kwh * ELECTRICITY_USD_PER_KWH
    co2_kg   = primary_kwh * CO2_KG_PER_KWH

    # ── Metric markdown ───────────────────────────────────────────────────
    energy_md  = f"## ⚡ {primary_kwh:.3f} kWh\n{_energy_label(primary_kwh)}"
    cost_md    = f"## 💵 ${cost_usd:.3f}\n*at $0.10 / kWh*"
    co2_md     = f"## 🌱 {co2_kg:.3f} kg\n*CO₂ equivalent*"
    context_md = f"**In context:** {_co2_context(co2_kg)}"

    # ── Extracted features markdown ───────────────────────────────────────
    extracted = _extractor.extract(text)
    ext_rows  = [
        ("GPUs detected",    str(int(extracted["num_gpus"]))),
        ("GPU power (TDP)",  f"{extracted['gpu_watts']:.0f} W"),
        ("Duration",         f"{extracted['hours']:.1f} h"),
        ("Task utilisation", f"{extracted['task_multiplier']:.0%}"),
        ("Model factor",     f"×{extracted['model_factor']:.1f}"),
        ("Physics proxy",    f"{np.expm1(extracted['log_energy_proxy']):.1f} Wh (raw)"),
    ]
    ext_table = "\n".join(f"| {k} | **{v}** |" for k, v in ext_rows)
    extracted_md = (
        "| Parameter | Value |\n"
        "|-----------|-------|\n"
        + ext_table
    )

    # ── Comparison table ──────────────────────────────────────────────────
    rows = [
        {
            "Model":         name,
            "Energy (kWh)":  f"{kwh:.4f}",
            "Cost ($)":      f"${kwh * ELECTRICITY_USD_PER_KWH:.4f}",
            "CO₂ (kg)":      f"{kwh * CO2_KG_PER_KWH:.4f}",
            "Tier":          _energy_label(kwh),
        }
        for name, kwh in predictions.items()
    ]
    comparison_df = pd.DataFrame(rows)

    # ── Charts ────────────────────────────────────────────────────────────
    gauge_fig  = _make_gauge(primary_kwh)
    bar_fig    = _make_bar_chart(predictions)
    radar_fig  = _make_feature_radar(extracted)

    status_md = (
        f"✅ Prediction complete using **{', '.join(predictions.keys())}**. "
        f"Primary estimate from **{'DistilBERT' if 'DistilBERT' in predictions else list(predictions.keys())[-1]}**."
    )

    return (
        energy_md, cost_md, co2_md, context_md,
        gauge_fig, bar_fig, radar_fig,
        comparison_df, extracted_md, status_md,
    )


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Assemble and return the complete Gradio Blocks interface."""

    with gr.Blocks(css=CUSTOM_CSS, theme=THEME, title="AI Energy Predictor") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            """
# ⚡ AI Energy Predictor
### Estimate energy consumption, cost & carbon footprint of AI training workloads — in plain English.
            """,
            elem_id="header-md",
        )

        # ── Input column ─────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    label       = "Describe your workload",
                    placeholder = 'e.g.  "Train ResNet-50 on 4 A100 GPUs for 10 hours with batch size 128"',
                    lines       = 3,
                    max_lines   = 6,
                    show_label  = True,
                )
                predict_btn = gr.Button(
                    "⚡  Predict Energy Usage",
                    variant  = "primary",
                    elem_id  = "predict-btn",
                )
                status_md = gr.Markdown("*Enter a description above and press Predict.*")

            # ── Primary metrics (right of input) ─────────────────────────
            with gr.Column(scale=3):
                energy_md  = gr.Markdown("## —\n—")
                cost_md    = gr.Markdown("## —\n—")
                co2_md     = gr.Markdown("## —\n—")
                context_md = gr.Markdown("")

        # ── Quick-try examples ───────────────────────────────────────────
        gr.Markdown("**Quick examples — click to load:**")
        with gr.Row():
            for p in EXAMPLE_PROMPTS[:4]:
                btn = gr.Button(p[:52] + ("…" if len(p) > 52 else ""), size="sm")
                btn.click(fn=lambda x=p: x, outputs=text_input)
        with gr.Row():
            for p in EXAMPLE_PROMPTS[4:]:
                btn = gr.Button(p[:52] + ("…" if len(p) > 52 else ""), size="sm")
                btn.click(fn=lambda x=p: x, outputs=text_input)

        gr.Markdown("---")

        # ── Charts row ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gauge_fig = gr.Plot(label="Energy Gauge", show_label=False)
            with gr.Column(scale=1):
                bar_fig = gr.Plot(label="Model Comparison", show_label=False)
            with gr.Column(scale=1):
                radar_fig = gr.Plot(label="Extracted Features", show_label=False)

        # ── Tables row ────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                comparison_df = gr.DataFrame(
                    label       = "All Model Predictions",
                    interactive = False,
                    wrap        = True,
                )
            with gr.Column(scale=2):
                extracted_md = gr.Markdown(
                    "| Parameter | Value |\n|---|---|\n| — | — |",
                    label="Extracted Parameters",
                )

        # ── How it works tab ─────────────────────────────────────────────
        with gr.Accordion("ℹ️  How it works", open=False):
            gr.Markdown("""
### Three models, one estimate

| Model | Approach | Strength |
|-------|----------|---------|
| **Naive Baseline** | Always predicts training-set mean | Lower-bound benchmark |
| **Classical ML (RF)** | TF-IDF + regex features → Random Forest | Fast, interpretable |
| **DistilBERT** | Fine-tuned transformer regression | Understands semantics, best accuracy |

### Energy formula
```
Energy (kWh) = GPU_watts × num_GPUs × hours × utilisation × PUE / 1000
```
- **PUE = 1.2** — typical datacenter Power Usage Effectiveness overhead
- **Utilisation** = task_multiplier × model_compute_factor (capped at 1.0)
- **Cost** at $0.10 / kWh (blended cloud GPU average)
- **CO₂** at 0.385 kg CO₂e / kWh (US EPA 2023 average grid intensity)

### What "extracted features" means
The classical model (and the radar chart) rely on regex extraction to pull
`num_gpus`, `hours`, `gpu_watts`, `task_multiplier`, and `model_factor`
directly from your text. DistilBERT doesn't need these — it learns the
mapping from raw tokens. Comparing the two is the core experiment of this project.
            """)

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            "*AI Energy Predictor · AI-540 Final Project · "
            "Estimates are for planning purposes only — real workloads vary.*",
        )

        # ── Wire predict button & Enter key ──────────────────────────────
        _outputs = [
            energy_md, cost_md, co2_md, context_md,
            gauge_fig, bar_fig, radar_fig,
            comparison_df, extracted_md, status_md,
        ]
        predict_btn.click(fn=predict, inputs=[text_input], outputs=_outputs)
        text_input.submit(fn=predict, inputs=[text_input], outputs=_outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_port=7860, show_error=True)
