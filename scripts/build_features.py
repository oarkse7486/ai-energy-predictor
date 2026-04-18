"""
build_features.py
=================
Feature engineering pipeline for the AI Energy Predictor.

Three feature representations are provided for the structured-vs-unstructured experiment:

    "structured"  — regex-extracted numeric features only
                    (num_gpus, hours, gpu_watts, task_multiplier, model_factor,
                     plus a physics-derived log_energy_proxy)

    "tfidf"       — TF-IDF bag-of-words on raw text only

    "combined"    — structured + TF-IDF concatenated (best-of-both-worlds)

Usage:
    python scripts/build_features.py --data-dir data/raw --output-dir data/processed
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables for regex extraction
# ---------------------------------------------------------------------------

# GPU model → TDP (watts); order matters — longer keys first to avoid partial matches
GPU_WATTAGE_MAP: Dict[str, float] = {
    "rtx 4090": 450, "rtx 3090": 350, "rtx 3080": 320,
    "rtx a6000": 300, "tesla t4": 70,
    "h100": 700, "a100": 400, "v100": 300,
    "a10g": 150, "a10": 150, "a6000": 300,
    "4090": 450, "3090": 350, "3080": 320,
    "t4": 70, "p100": 250,
}

# Task keyword → utilisation multiplier
TASK_MULT_MAP: Dict[str, float] = {
    "pretraining":       1.00, "pre-training":    1.00,
    "training":          0.90,
    "fine-tuning":       0.65, "fine tuning":     0.65, "finetuning": 0.65,
    "inference":         0.30,
    "evaluation":        0.20,
    "feature extraction": 0.25,
}

# Model keyword → compute factor (higher = more FLOP-intensive per GPU-hour)
MODEL_FACTOR_MAP: Dict[str, float] = {
    "llama-13b": 7.0,  "llama-7b": 5.0, "llama": 5.0,
    "mistral-7b": 5.2, "mistral": 5.2,
    "stable diffusion": 3.5,
    "gpt-2 large": 3.0, "gpt-2": 1.6, "gpt2": 1.6,
    "vit-large": 2.1,
    "bert-large": 2.0,
    "whisper": 1.8, "whisper-medium": 1.8,
    "roberta": 1.5, "bert-base": 1.5, "bert": 1.5,
    "vit-base": 1.4, "vit": 1.4,
    "resnet-152": 1.3, "distilbert": 1.2,
    "vgg-16": 1.2, "vgg": 1.2,
    "resnet-50": 1.0, "resnet50": 1.0, "resnet": 1.0,
    "efficientnet": 0.9, "yolov8": 0.8, "yolo": 0.8,
}


# ---------------------------------------------------------------------------
# Structured feature extractor
# ---------------------------------------------------------------------------

class StructuredFeatureExtractor:
    """
    Extracts six numeric features from free-text AI workload descriptions
    using regex patterns and keyword lookups.

    Output features:
        num_gpus         — integer GPU count
        hours            — job duration in hours
        gpu_watts        — estimated GPU TDP (W)
        task_multiplier  — GPU utilisation fraction by task type
        model_factor     — compute intensity relative to ResNet-50
        log_energy_proxy — log1p(num_gpus × gpu_watts × hours), a physics proxy
    """

    # Matches "4x A100", "4 GPUs", "4 a100s", "GPUs: 4", etc.
    _GPU_COUNT_RE = re.compile(
        r"(?:(\d+)\s*[xX×]\s*(?:gpu|a100|h100|v100|t4|4090|3090|3080|a10|a6000|p100|rtx))"
        r"|(?:(\d+)\s+(?:gpu|a100|h100|v100|t4|4090|3090|3080|a10|a6000|p100|rtx))"
        r"|(?:(?:gpu|card|device)s?[:\s]+(\d+))",
        re.IGNORECASE,
    )
    _WORD_NUMS = {"one": 1, "two": 2, "four": 4, "eight": 8, "sixteen": 16}
    _WORD_NUM_RE = re.compile(r"\b(one|two|four|eight|sixteen)\b", re.IGNORECASE)

    # Matches "10 hours", "10h", "10hrs", "10 hr", "10.5 hours"
    _HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h\b)", re.IGNORECASE)

    def extract(self, text: str) -> Dict[str, float]:
        """Extract features from a single text description."""
        num_gpus        = self._extract_num_gpus(text)
        hours           = self._extract_hours(text)
        gpu_watts       = self._extract_gpu_watts(text)
        task_multiplier = self._extract_task_multiplier(text)
        model_factor    = self._extract_model_factor(text)
        log_proxy       = float(np.log1p(num_gpus * gpu_watts * hours))

        return {
            "num_gpus":         num_gpus,
            "hours":            hours,
            "gpu_watts":        gpu_watts,
            "task_multiplier":  task_multiplier,
            "model_factor":     model_factor,
            "log_energy_proxy": log_proxy,
        }

    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extract features for a list of descriptions; returns a DataFrame."""
        return pd.DataFrame([self.extract(t) for t in texts])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_num_gpus(self, text: str) -> float:
        match = self._GPU_COUNT_RE.search(text)
        if match:
            for group in match.groups():
                if group:
                    return float(group)
        # Fallback: word-form numbers before any GPU keyword
        match = self._WORD_NUM_RE.search(text)
        if match:
            return float(self._WORD_NUMS.get(match.group(1).lower(), 1))
        return 1.0

    def _extract_hours(self, text: str) -> float:
        match = self._HOURS_RE.search(text)
        if match:
            return float(match.group(1))
        return 1.0  # Default: assume 1-hour job

    def _extract_gpu_watts(self, text: str) -> float:
        text_lower = text.lower()
        for keyword, watts in GPU_WATTAGE_MAP.items():
            if keyword in text_lower:
                return watts
        return 250.0  # Default: mid-range GPU

    def _extract_task_multiplier(self, text: str) -> float:
        text_lower = text.lower()
        for keyword, mult in TASK_MULT_MAP.items():
            if keyword in text_lower:
                return mult
        return 0.75  # Default: general ML compute

    def _extract_model_factor(self, text: str) -> float:
        text_lower = text.lower()
        for keyword, factor in MODEL_FACTOR_MAP.items():
            if keyword in text_lower:
                return factor
        return 1.0  # Default: ResNet-50-class model


# ---------------------------------------------------------------------------
# Feature pipeline (structured / TF-IDF / combined)
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Builds feature matrices for a given representation type.

    Supports three modes:
        "structured"  — StructuredFeatureExtractor → StandardScaler
        "tfidf"       — TfidfVectorizer (dense)
        "combined"    — structured + TF-IDF concatenated

    Typical usage::

        pipeline = FeaturePipeline(feature_type="combined")
        X_train, y_train = pipeline.fit_transform(train_df)
        X_test           = pipeline.transform(test_df)
        pipeline.save(Path("data/processed/combined/pipeline"))
    """

    VALID_TYPES = ("structured", "tfidf", "combined")

    def __init__(self, feature_type: str = "combined", max_tfidf_features: int = 5000):
        if feature_type not in self.VALID_TYPES:
            raise ValueError(f"feature_type must be one of {self.VALID_TYPES}")

        self.feature_type = feature_type
        self.extractor    = StructuredFeatureExtractor()
        self.tfidf        = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.scaler   = StandardScaler()
        self._fitted  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on training data, return (X_matrix, y_labels)."""
        texts = df["description"].tolist()
        y     = df["energy_kwh"].values

        if self.feature_type in ("tfidf", "combined"):
            self.tfidf.fit(texts)

        X           = self._build_X(texts, fit_scaler=True)
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform unseen data using the fitted pipeline."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        return self._build_X(df["description"].tolist(), fit_scaler=False)

    def save(self, path: Path) -> None:
        """Persist the fitted pipeline to disk."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf,  path / "tfidf.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump({"feature_type": self.feature_type, "fitted": self._fitted}, path / "meta.pkl")

    @classmethod
    def load(cls, path: Path) -> "FeaturePipeline":
        """Load a previously fitted pipeline from disk."""
        meta     = joblib.load(path / "meta.pkl")
        instance = cls(feature_type=meta["feature_type"])
        instance.tfidf   = joblib.load(path / "tfidf.pkl")
        instance.scaler  = joblib.load(path / "scaler.pkl")
        instance._fitted = meta["fitted"]
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_X(self, texts: List[str], fit_scaler: bool) -> np.ndarray:
        """Build the feature matrix according to self.feature_type."""
        if self.feature_type == "structured":
            struct_arr = self.extractor.extract_batch(texts).values
            return self.scaler.fit_transform(struct_arr) if fit_scaler else self.scaler.transform(struct_arr)

        if self.feature_type == "tfidf":
            return self.tfidf.transform(texts).toarray()

        # --- combined ---
        struct_arr = self.extractor.extract_batch(texts).values
        struct_X   = self.scaler.fit_transform(struct_arr) if fit_scaler else self.scaler.transform(struct_arr)
        tfidf_X    = self.tfidf.transform(texts).toarray()
        return np.hstack([struct_X, tfidf_X])


# ---------------------------------------------------------------------------
# Entry point  (builds all three feature sets from disk)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrices for all experiment conditions")
    parser.add_argument("--data-dir",   default="data/raw",       help="Directory with train/val/test CSVs")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    for ftype in FeaturePipeline.VALID_TYPES:
        logger.info(f"Building '{ftype}' features...")
        pipeline = FeaturePipeline(feature_type=ftype)

        X_train, y_train = pipeline.fit_transform(train_df)
        X_val            = pipeline.transform(val_df)
        y_val            = val_df["energy_kwh"].values
        X_test           = pipeline.transform(test_df)
        y_test           = test_df["energy_kwh"].values

        feature_dir = output_dir / ftype
        feature_dir.mkdir(parents=True, exist_ok=True)

        for name, arr in [("X_train", X_train), ("y_train", y_train),
                           ("X_val",   X_val),   ("y_val",   y_val),
                           ("X_test",  X_test),  ("y_test",  y_test)]:
            np.save(feature_dir / f"{name}.npy", arr)

        pipeline.save(feature_dir / "pipeline")
        logger.info(f"  '{ftype}': X_train.shape={X_train.shape}")

    logger.info("Feature building complete.")


if __name__ == "__main__":
    main()
