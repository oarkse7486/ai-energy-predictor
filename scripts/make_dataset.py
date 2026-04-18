"""
make_dataset.py
===============
Generates a synthetic dataset of AI training job descriptions paired with
physics-based energy consumption labels (kWh).

Energy is computed from: GPU wattage × GPU count × hours × utilization × PUE
Then ±8% Gaussian noise is applied to simulate real-world variance.

Usage:
    python scripts/make_dataset.py --n-samples 5000 --output-dir data/raw
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware & workload specifications
# ---------------------------------------------------------------------------

GPU_SPECS: Dict[str, Dict] = {
    "A100":     {"watts": 400, "vram_gb": 80,  "tier": "enterprise",    "aliases": ["a100", "A100", "a100 80gb"]},
    "H100":     {"watts": 700, "vram_gb": 80,  "tier": "enterprise",    "aliases": ["h100", "H100", "hopper"]},
    "V100":     {"watts": 300, "vram_gb": 32,  "tier": "professional",  "aliases": ["v100", "V100", "volta"]},
    "A10G":     {"watts": 150, "vram_gb": 24,  "tier": "professional",  "aliases": ["a10", "a10g", "A10G", "a10 gpu"]},
    "T4":       {"watts": 70,  "vram_gb": 16,  "tier": "entry",         "aliases": ["t4", "T4", "tesla t4"]},
    "RTX 4090": {"watts": 450, "vram_gb": 24,  "tier": "consumer",      "aliases": ["4090", "rtx 4090", "rtx4090"]},
    "RTX 3090": {"watts": 350, "vram_gb": 24,  "tier": "consumer",      "aliases": ["3090", "rtx 3090", "rtx3090"]},
    "RTX 3080": {"watts": 320, "vram_gb": 10,  "tier": "consumer",      "aliases": ["3080", "rtx 3080", "rtx3080"]},
    "A6000":    {"watts": 300, "vram_gb": 48,  "tier": "professional",  "aliases": ["a6000", "A6000", "rtx a6000"]},
    "P100":     {"watts": 250, "vram_gb": 16,  "tier": "professional",  "aliases": ["p100", "P100", "pascal"]},
}

MODEL_SPECS: Dict[str, Dict] = {
    "ResNet-50":        {"params_m": 25,    "compute_factor": 1.0, "domain": "vision"},
    "ResNet-152":       {"params_m": 60,    "compute_factor": 1.3, "domain": "vision"},
    "VGG-16":           {"params_m": 138,   "compute_factor": 1.2, "domain": "vision"},
    "EfficientNet-B4":  {"params_m": 19,    "compute_factor": 0.9, "domain": "vision"},
    "ViT-Base":         {"params_m": 86,    "compute_factor": 1.4, "domain": "vision"},
    "ViT-Large":        {"params_m": 307,   "compute_factor": 2.1, "domain": "vision"},
    "YOLOv8":           {"params_m": 25,    "compute_factor": 0.8, "domain": "detection"},
    "BERT-base":        {"params_m": 110,   "compute_factor": 1.5, "domain": "nlp"},
    "BERT-large":       {"params_m": 340,   "compute_factor": 2.0, "domain": "nlp"},
    "DistilBERT":       {"params_m": 66,    "compute_factor": 1.2, "domain": "nlp"},
    "RoBERTa":          {"params_m": 125,   "compute_factor": 1.5, "domain": "nlp"},
    "GPT-2":            {"params_m": 117,   "compute_factor": 1.6, "domain": "nlp"},
    "GPT-2 Large":      {"params_m": 774,   "compute_factor": 3.0, "domain": "nlp"},
    "LLaMA-7B":         {"params_m": 7000,  "compute_factor": 5.0, "domain": "llm"},
    "LLaMA-13B":        {"params_m": 13000, "compute_factor": 7.0, "domain": "llm"},
    "Mistral-7B":       {"params_m": 7000,  "compute_factor": 5.2, "domain": "llm"},
    "Stable Diffusion": {"params_m": 860,   "compute_factor": 3.5, "domain": "generative"},
    "Whisper-medium":   {"params_m": 305,   "compute_factor": 1.8, "domain": "audio"},
}

# Fraction of peak GPU TDP actually consumed, by task type
TASK_MULTIPLIERS: Dict[str, float] = {
    "training":          0.90,
    "fine-tuning":       0.65,
    "pretraining":       1.00,
    "inference":         0.30,
    "evaluation":        0.20,
    "feature extraction": 0.25,
}

# Datacenter Power Usage Effectiveness (overhead: cooling, power conversion, etc.)
PUE = 1.2

ELECTRICITY_COST_USD_PER_KWH = 0.10
CO2_KG_PER_KWH               = 0.385   # US average grid carbon intensity


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Generates (description, energy_kwh) pairs for AI training workload prompts.

    Each sample is created by:
    1. Sampling GPU, count, duration, task, and model from realistic distributions.
    2. Computing energy deterministically via a physics formula.
    3. Rendering the parameters into a natural-language description using
       randomly selected template styles (formal, casual, shorthand, question).
    4. Adding ±8 % Gaussian noise to simulate measurement variance.
    """

    # Integer words used in small-count variants
    _WORD_NUMS = {1: "one", 2: "two", 4: "four", 8: "eight", 16: "sixteen"}
    _DATASET_NAMES = [
        "ImageNet", "CIFAR-10", "WikiText-103", "Common Crawl", "LAION-400M",
        "SQuAD", "MS COCO", "a custom NLP", "OpenWebText", "The Pile",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self._gpu_names  = list(GPU_SPECS.keys())
        self._model_names = list(MODEL_SPECS.keys())
        self._task_names  = list(TASK_MULTIPLIERS.keys())

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def _compute_energy_kwh(
        self, gpu: str, num_gpus: int, hours: float, task: str, model: str
    ) -> float:
        """
        Compute energy (kWh) using a physics-inspired formula.

        Formula:
            utilization  = min(task_multiplier × model_compute_factor, 1.0)
            energy_kwh   = (gpu_watts × num_gpus × hours × utilization × PUE) / 1000
        """
        gpu_watts      = GPU_SPECS[gpu]["watts"]
        task_util      = TASK_MULTIPLIERS[task]
        model_factor   = MODEL_SPECS[model]["compute_factor"]
        utilization    = min(task_util * model_factor, 1.0)
        energy_kwh     = (gpu_watts * num_gpus * hours * utilization * PUE) / 1000

        # Add ±8 % Gaussian noise for realism
        noise = self.rng.normal(1.0, 0.08)
        return round(energy_kwh * noise, 4)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _render_description(
        self,
        gpu: str, num_gpus: int, hours: float, task: str, model: str,
        batch_size: Optional[int], dataset_name: Optional[str],
    ) -> str:
        """Return a natural-language description for the given parameters."""

        # Vary the GPU name from its alias list
        gpu_alias = random.choice(GPU_SPECS[gpu]["aliases"])

        # Occasionally spell out small GPU counts as words
        if num_gpus in self._WORD_NUMS and random.random() < 0.15:
            gpu_count_str = self._WORD_NUMS[num_gpus]
        else:
            gpu_count_str = str(num_gpus)

        # Vary the hours format
        hours_formats = (
            [f"{int(hours)} hour{'s' if hours > 1 else ''}",
             f"{int(hours)}h",
             f"{int(hours)}hrs",
             f"{int(hours)} hr{'s' if hours > 1 else ''}"]
            if hours == int(hours)
            else [f"{hours:.1f} hours", f"{hours:.1f}h"]
        )
        hours_str = random.choice(hours_formats)

        plural = "s" if num_gpus > 1 else ""
        batch_str   = f" with batch size {batch_size}" if batch_size   and random.random() < 0.5 else ""
        dataset_str = f" on {dataset_name}"            if dataset_name and random.random() < 0.4 else ""

        templates: List[str] = [
            # -- Formal / technical --
            f"{task.capitalize()} {model} on {gpu_count_str} {gpu_alias} GPU{plural} for {hours_str}{batch_str}.",
            f"Run {task} of {model} using {gpu_count_str}x {gpu_alias} for {hours_str}{dataset_str}.",
            f"Distributed {task} of {model} across {gpu_count_str} {gpu_alias} cards, estimated runtime {hours_str}.",
            f"{model} {task}{dataset_str} — {gpu_count_str} {gpu_alias}{plural}, {hours_str}{batch_str}.",
            f"Launch a {task} job for {model} on {gpu_count_str} {gpu_alias} GPU{plural}, duration {hours_str}{batch_str}.",

            # -- Casual / conversational --
            f"gonna {task} {model} on {gpu_count_str} {gpu_alias}{plural} for about {hours_str}",
            f"need to {task.lower()} {model}{dataset_str}, planning {gpu_count_str} {gpu_alias}{plural} for {hours_str}",
            f"running {model} {task}{dataset_str} on {gpu_count_str} {gpu_alias}{plural}, roughly {hours_str}",

            # -- Question / planning style --
            f"How much will it cost to {task.lower()} {model} on {gpu_count_str} {gpu_alias}{plural} for {hours_str}?",
            f"Planning to {task.lower()} {model}{dataset_str} with {gpu_count_str} {gpu_alias} GPU{plural} for {hours_str}{batch_str}.",

            # -- Shorthand / pipeline style --
            f"{model}, {task}, {gpu_count_str}x {gpu_alias}, {hours_str}",
            f"{task.capitalize()}: {model} | {gpu_count_str} {gpu_alias}{plural} | {hours_str}",
        ]

        return random.choice(templates)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sample(self) -> Dict:
        """Generate a single labelled sample."""
        gpu      = random.choice(self._gpu_names)
        num_gpus = random.choice([1, 2, 4, 8, 16])
        # Sample hours from a realistic distribution (many short jobs, some week-long)
        hours    = round(
            float(self.rng.choice(
                [*np.linspace(0.5, 2, 10),
                 *np.linspace(2, 24, 20),
                 *np.linspace(24, 168, 15)]
            )), 1
        )
        task       = random.choice(self._task_names)
        model      = random.choice(self._model_names)
        batch_size = random.choice([16, 32, 64, 128, 256, 512]) if random.random() < 0.6 else None
        dataset    = random.choice(self._DATASET_NAMES)         if random.random() < 0.5 else None

        energy_kwh  = self._compute_energy_kwh(gpu, num_gpus, hours, task, model)
        description = self._render_description(gpu, num_gpus, hours, task, model, batch_size, dataset)

        return {
            # --- Label ---
            "description":          description,
            "energy_kwh":           energy_kwh,
            # --- Structured ground-truth (used in experiment notebook) ---
            "gpu":                  gpu,
            "num_gpus":             num_gpus,
            "hours":                hours,
            "task":                 task,
            "model":                model,
            "gpu_watts":            GPU_SPECS[gpu]["watts"],
            "model_compute_factor": MODEL_SPECS[model]["compute_factor"],
            "task_multiplier":      TASK_MULTIPLIERS[task],
            "batch_size":           batch_size,
        }

    def generate_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate n_samples labelled examples and return as a DataFrame."""
        logger.info(f"Generating {n_samples} samples...")
        samples = [self.generate_sample() for _ in range(n_samples)]
        df      = pd.DataFrame(samples)
        logger.info(
            f"Dataset stats — min={df['energy_kwh'].min():.3f} kWh  "
            f"max={df['energy_kwh'].max():.3f} kWh  "
            f"mean={df['energy_kwh'].mean():.3f} kWh"
        )
        return df

    def save(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Save raw data and 70/15/15 train/val/test splits to output_dir.
        Also writes a metadata.json with constants used during generation.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reproducible shuffle then split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n         = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        df.to_csv(output_dir / "raw.csv", index=False)
        df.iloc[:train_end].to_csv(        output_dir / "train.csv", index=False)
        df.iloc[train_end:val_end].to_csv( output_dir / "val.csv",   index=False)
        df.iloc[val_end:].to_csv(          output_dir / "test.csv",  index=False)

        logger.info(f"Saved → {output_dir}  (train={train_end} | val={val_end - train_end} | test={n - val_end})")

        meta = {
            "n_samples":                     n,
            "splits":                        {"train": train_end, "val": val_end - train_end, "test": n - val_end},
            "pue":                           PUE,
            "electricity_cost_usd_per_kwh":  ELECTRICITY_COST_USD_PER_KWH,
            "co2_kg_per_kwh":               CO2_KG_PER_KWH,
            "gpu_specs":                     GPU_SPECS,
            "model_specs":                   MODEL_SPECS,
            "task_multipliers":              TASK_MULTIPLIERS,
        }
        with open(output_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AI energy dataset")
    parser.add_argument("--n-samples",  type=int, default=5000,     help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed",       type=int, default=42,        help="Random seed")
    args = parser.parse_args()

    gen = DatasetGenerator(seed=args.seed)
    df  = gen.generate_dataset(n_samples=args.n_samples)
    gen.save(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
