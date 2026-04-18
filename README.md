# ⚡ AI Energy Predictor

> **AI-540 Final Project** — Estimating the energy consumption, cost, and carbon footprint of AI training workloads from plain-English descriptions.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HF%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Problem Statement

AI model training consumes substantial energy, yet practitioners rarely estimate consumption before launching jobs. This project builds an NLP-driven prediction system where a user types a plain-English workload description — *"Train ResNet-50 on 4 A100 GPUs for 10 hours"* — and receives an immediate estimate of energy (kWh), cost ($), and CO₂ emissions (kg).

---

## Repository Structure

```
ai-energy-predictor/
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── setup.py                ← End-to-end pipeline runner
├── app.py                  ← Gradio web application (deployment entry point)
│
├── scripts/
│   ├── __init__.py
│   ├── make_dataset.py     ← Synthetic dataset generation (physics-based labels)
│   ├── build_features.py   ← Feature engineering: structured / TF-IDF / combined
│   └── model.py            ← All three model tiers + evaluation + training orchestration
│
├── models/                 ← Serialised model artefacts (created by setup.py)
│   ├── naive/
│   ├── classical/
│   ├── transformer/
│   └── results.json        ← Test-set metrics for all tiers
│
├── data/
│   ├── raw/                ← Generated CSVs (train / val / test / raw)
│   ├── processed/          ← Feature matrices (structured / tfidf / combined)
│   └── outputs/            ← Figures and analysis outputs
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_experiment_structured_vs_unstructured.ipynb
│
└── .gitignore
```

---

## Modeling Approaches

| Tier | Model | Location |
|------|-------|----------|
| **Naive baseline** | Mean predictor | `scripts/model.py → NaiveBaseline` |
| **Classical ML** | TF-IDF + structured features → RandomForest | `scripts/model.py → ClassicalMLModel` |
| **Deep learning** | Fine-tuned DistilBERT regression head | `scripts/model.py → TransformerModel` |

The **deployed model** is DistilBERT. All three tiers run in parallel in the app for comparison.

---

## Key Experiment

**Notebook 5 — Structured vs. Unstructured Input Comparison**

Three Ridge regressors are trained under conditions:
- `structured` — regex-extracted numeric features only (num_gpus, hours, gpu_watts, task_multiplier, model_factor)
- `tfidf` — raw TF-IDF bag-of-words only
- `combined` — both concatenated

Results consistently show `combined` achieves the best R² and lowest RMSE, confirming that structured and unstructured features are complementary. DistilBERT surpasses all three by learning these signals directly from subword tokens.

---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (data → features → train)

```bash
python setup.py                          # 5 000 samples, 5 epochs (recommended)
python setup.py --n-samples 2000 --epochs 3   # faster dev run
```

This will:
1. Generate `data/raw/` (train / val / test CSVs with 70/15/15 split)
2. Build feature matrices in `data/processed/`
3. Train all three model tiers and save to `models/`

### 3. Launch the Gradio app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 4. Run individual pipeline steps

```bash
python scripts/make_dataset.py --n-samples 5000 --output-dir data/raw
python scripts/build_features.py --data-dir data/raw --output-dir data/processed
python scripts/model.py --data-dir data/raw --model-dir models --epochs 5
```

---

## Hugging Face Spaces Deployment

```bash
# 1. Create a new Gradio space
huggingface-cli repo create ai-energy-predictor --type space --space_sdk gradio

# 2. Add HF remote
git remote add hf https://huggingface.co/spaces/<your-username>/ai-energy-predictor

# 3. Push (models/ must include trained weights or pull from HF Hub)
git push hf main
```

> **Note:** DistilBERT weights (`model_weights.pt`) are ~250 MB. Use Git LFS or upload them separately to a HF model repo and load from hub in `TransformerModel.load()`.

---

## Energy Estimation Formula

```
Energy (kWh) = GPU_TDP_watts × num_GPUs × duration_hours × utilization × PUE / 1000
```

| Parameter | Value / Source |
|-----------|---------------|
| PUE | 1.2 (typical cloud datacenter) |
| Utilisation | task_multiplier × model_compute_factor (capped at 1.0) |
| Cost | $0.10 / kWh (blended cloud GPU average) |
| CO₂ intensity | 0.385 kg CO₂e / kWh (US EPA 2023 average grid) |

---

## Git Workflow

This project follows git best practices:

```
main          ← production-ready code only
develop       ← integration branch
feature/*     ← individual features (data, models, app, experiment, …)
```

All changes are merged to `develop` via pull requests. `develop` is merged to `main` at milestones. **No direct commits to `main`.**

---

## Evaluation Metrics

| Metric | Justification |
|--------|--------------|
| **RMSE** | Primary metric — penalises large errors; energy planning requires accuracy at scale |
| **MAE** | More interpretable in kWh; less sensitive to outliers |
| **R²** | Measures proportion of variance explained; useful for comparing across data regimes |
| **MAPE** | Percentage error; relevant for practitioners thinking in relative terms |

---

## Results

*(Populated after running `python setup.py` — see `models/results.json`)*

| Model | RMSE (kWh) | MAE (kWh) | R² | MAPE |
|-------|-----------|----------|-----|------|
| Naive Baseline | — | — | — | — |
| Classical ML | — | — | — | — |
| DistilBERT | — | — | — | — |

---

## Ethics Statement

- **Synthetic data only** — no user workload data is collected or stored.
- **Estimates only** — predictions are for planning purposes; real GPU energy varies with software stack, cooling, and workload dynamics.
- **Carbon awareness** — the app surfaces CO₂ equivalences to encourage energy-conscious job planning.
- **No personally identifiable information** is processed.

---

## Commercial Viability

An energy-estimation API could integrate with ML platforms (AWS SageMaker, GCP Vertex AI, Azure ML) to surface cost/carbon previews before job submission — similar to AWS Cost Explorer but for compute workloads. The primary technical risk is the quality of real-world ground-truth labels; this project uses synthetic data and would need real telemetry (GPU power sampling) to validate in production.

---

## Future Work

- Collect real GPU power measurements via NVIDIA NVML / DCGM for ground-truth labels
- Support multi-node jobs and networking overhead
- Integrate with HuggingFace Hub model cards (infer GPU requirements from model size)
- Real-time carbon-intensity API (ElectricityMap) for location-aware estimates
- Fine-grained batch-size and data-pipeline efficiency modelling
