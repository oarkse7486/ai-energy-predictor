"""
model.py
========
Trains, evaluates, and serializes all three model tiers:

    Tier 1  NaiveBaseline      — constant mean predictor (lower-bound benchmark)
    Tier 2  ClassicalMLModel   — combined TF-IDF + structured features → RandomForest
    Tier 3  TransformerModel   — fine-tuned DistilBERT with a regression head

Run end-to-end to produce model artefacts under models/:

    python scripts/model.py --data-dir data/raw --model-dir models --epochs 5

Each model exposes a predict_single(text: str) → float interface so app.py
can run inference without knowing which tier is being used.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from scripts.build_features import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R², and MAPE for a set of predictions.

    Returns:
        dict with keys: rmse, mae, r2, mape
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # Guard against near-zero true values when computing MAPE
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "mape": round(mape, 2)}


# ---------------------------------------------------------------------------
# Tier 1 — Naive baseline
# ---------------------------------------------------------------------------

class NaiveBaseline:
    """
    Predicts the training-set mean energy regardless of input text.

    Purpose: establishes the minimum accuracy bar. All other models must
    outperform this to justify their additional complexity.
    """

    def __init__(self):
        self.mean_energy: float = 0.0

    def fit(self, y_train: np.ndarray) -> "NaiveBaseline":
        """Store the mean of the training labels."""
        self.mean_energy = float(np.mean(y_train))
        logger.info(f"NaiveBaseline fitted — mean_energy = {self.mean_energy:.4f} kWh")
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return an array of n copies of the training mean."""
        return np.full(n, self.mean_energy)

    def predict_single(self, text: str) -> float:  # noqa: ARG002  (text unused by design)
        """Return mean energy; text is ignored."""
        return self.mean_energy

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mean_energy": self.mean_energy}, path / "naive_baseline.pkl")

    @classmethod
    def load(cls, path: Path) -> "NaiveBaseline":
        data = joblib.load(path / "naive_baseline.pkl")
        obj  = cls()
        obj.mean_energy = data["mean_energy"]
        return obj


# ---------------------------------------------------------------------------
# Tier 2 — Classical ML (Random Forest)
# ---------------------------------------------------------------------------

class ClassicalMLModel:
    """
    Combined TF-IDF + structured features → RandomForestRegressor.

    Uses FeaturePipeline(feature_type="combined") which concatenates:
        - TF-IDF bag-of-words (captures vocabulary patterns)
        - Regex-extracted structured features (num_gpus, hours, gpu_watts, …)

    This dual representation gives the model both surface-form and numeric
    signals, generally outperforming either alone.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 20):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.pipeline     = FeaturePipeline(feature_type="combined")
        self.regressor    = RandomForestRegressor(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            min_samples_leaf = 2,
            n_jobs        = -1,
            random_state  = 42,
        )

    def fit(self, train_df: pd.DataFrame) -> "ClassicalMLModel":
        """Fit feature pipeline and random-forest regressor on training data."""
        logger.info("Training ClassicalMLModel (RandomForest) …")
        X_train, y_train = self.pipeline.fit_transform(train_df)
        self.regressor.fit(X_train, y_train)
        logger.info("ClassicalMLModel training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict energy (kWh) for a DataFrame with a 'description' column."""
        X    = self.pipeline.transform(df)
        preds = self.regressor.predict(X)

        return np.clip(preds, 0, None)   # Energy is non-negative

    def predict_single(self, text: str) -> float:
        """Predict from a single raw-text description."""
        df = pd.DataFrame({"description": [text]})
        return float(self.predict(df)[0])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.regressor, path / "regressor.pkl")
        joblib.dump({"n_estimators": self.n_estimators, "max_depth": self.max_depth}, path / "meta.pkl")
        self.pipeline.save(path / "pipeline")

    @classmethod
    def load(cls, path: Path) -> "ClassicalMLModel":
        meta = joblib.load(path / "meta.pkl")
        obj  = cls(n_estimators=meta["n_estimators"], max_depth=meta["max_depth"])
        obj.regressor = joblib.load(path / "regressor.pkl")
        obj.pipeline  = FeaturePipeline.load(path / "pipeline")
        return obj


# ---------------------------------------------------------------------------
# Tier 3 — DistilBERT transformer
# ---------------------------------------------------------------------------

class _EnergyDataset(Dataset):
    """
    PyTorch Dataset that tokenizes text on-the-fly.

    Labels are stored in log-space (log1p) so the model learns relative
    magnitude rather than absolute values across a multi-decade energy range.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 128,
    ):
        self.texts      = texts
        self.labels     = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
        }


class _DistilBertRegressor(nn.Module):
    """
    DistilBERT encoder with a two-layer MLP regression head.

    Architecture:
        [CLS] embedding (768-d)  →  Dropout  →  Linear(768, 256)
        →  GELU  →  Dropout  →  Linear(256, 1)

    Output is a single scalar in log-energy space.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.2):
        super().__init__()
        self.encoder     = DistilBertModel.from_pretrained(model_name)
        hidden           = self.encoder.config.hidden_size   # 768
        self.reg_head    = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass; returns shape [batch_size]."""
        out          = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed    = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.reg_head(cls_embed).squeeze(-1)


class TransformerModel:
    """
    Fine-tuned DistilBERT for energy regression.

    Training details:
        - Labels are log1p-transformed to stabilize learning across the wide kWh range
        - AdamW optimizer with linear warmup schedule and gradient clipping (max norm 1.0)
        - Best checkpoint selected by validation RMSE (in raw kWh, not log space)

    Inference:
        - Runs in batches; reverses the log transform with expm1
        - Clips predictions to [0, ∞) since energy is non-negative
    """

    _MODEL_NAME = "distilbert-base-uncased"

    def __init__(
        self,
        max_length: int = 128,
        batch_size: int = 32,
        epochs:     int = 5,
        lr:         float = 2e-5,
    ):
        self.max_length  = max_length
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.lr          = lr
        self.tokenizer   = DistilBertTokenizerFast.from_pretrained(self._MODEL_NAME)
        self._net: Optional[_DistilBertRegressor] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "TransformerModel":
        """Fine-tune DistilBERT; keeps the best checkpoint by val RMSE."""
        y_train_raw = train_df["energy_kwh"].values
        y_val_raw   = val_df["energy_kwh"].values

        # Log-transform to stabilise gradients over a wide target range
        y_train_log = np.log1p(y_train_raw)
        y_val_log   = np.log1p(y_val_raw)

        train_loader = DataLoader(
            _EnergyDataset(train_df["description"].tolist(), y_train_log, self.tokenizer, self.max_length),
            batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            _EnergyDataset(val_df["description"].tolist(), y_val_log, self.tokenizer, self.max_length),
            batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )

        self._net = _DistilBertRegressor(self._MODEL_NAME).to(DEVICE)

        # could tune parameters like weight decay & lr
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=0.01)

        total_steps = len(train_loader) * self.epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )
        criterion = nn.MSELoss()

        best_rmse  = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ---- training pass ----
            self._net.train()
            train_losses: List[float] = []
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self._net(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                )
                loss  = criterion(preds, batch["label"].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            # ---- validation pass ----
            self._net.eval()
            val_log_preds: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    out = self._net(
                        batch["input_ids"].to(DEVICE),
                        batch["attention_mask"].to(DEVICE),
                    )
                    val_log_preds.extend(out.cpu().numpy())

            # TODO: clip doesn't help in this line so consdier getting rid of it bc
            # are taking an exponential positive value.
            val_preds_kWh = np.clip(np.expm1(np.array(val_log_preds)), 0, None)
            val_rmse      = float(np.sqrt(mean_squared_error(y_val_raw, val_preds_kWh)))

            # TODO: could potentially add early stopping

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_loss={np.mean(train_losses):.4f}  val_RMSE={val_rmse:.4f} kWh"
            )

            
            if val_rmse < best_rmse:
                best_rmse  = val_rmse
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

        self._net.load_state_dict(best_state)
        logger.info(f"DistilBERT training complete — best val RMSE={best_rmse:.4f} kWh")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict energy (kWh) for a DataFrame with a 'description' column."""
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        dataset = _EnergyDataset(
            df["description"].tolist(), np.zeros(len(df)), self.tokenizer, self.max_length
        )
        loader   = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        log_preds: List[float] = []

        self._net.eval()
        with torch.no_grad():
            for batch in loader:
                out = self._net(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                )
                log_preds.extend(out.cpu().numpy())

        return np.clip(np.expm1(np.array(log_preds)), 0, None)

    def predict_single(self, text: str) -> float:
        """Predict from a single raw-text description."""
        df = pd.DataFrame({"description": [text]})
        return float(self.predict(df)[0])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "model_weights.pt")
        self.tokenizer.save_pretrained(str(path / "tokenizer"))
        joblib.dump({"max_length": self.max_length, "batch_size": self.batch_size}, path / "meta.pkl")

    @classmethod
    def load(cls, path: Path) -> "TransformerModel":
        meta = joblib.load(path / "meta.pkl")
        obj  = cls(max_length=meta["max_length"], batch_size=meta["batch_size"])
        obj._net = _DistilBertRegressor(cls._MODEL_NAME).to(DEVICE)
        obj._net.load_state_dict(
            torch.load(path / "model_weights.pt", map_location=DEVICE)
        )
        obj._net.eval()
        return obj


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train_all_models(
    data_dir:  Path,
    model_dir: Path,
    epochs:    int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Train all three tiers end-to-end, evaluate on the held-out test set,
    and write per-model results to models/results.json.

    Returns:
        Nested dict  {model_name: {rmse, mae, r2, mape}}
    """
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")
    y_test   = test_df["energy_kwh"].values

    results: Dict[str, Dict] = {}

    # ── Tier 1: Naive baseline ──────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Tier 1 — Naive Baseline")
    naive = NaiveBaseline().fit(train_df["energy_kwh"].values)
    preds = naive.predict(len(y_test))
    results["naive"] = evaluate(y_test, preds)
    naive.save(model_dir / "naive")
    logger.info(f"  Naive   → {results['naive']}")

    # ── Tier 2: Classical ML ────────────────────────────────────────────
    logger.info("Tier 2 — Classical ML (RandomForest)")
    classical = ClassicalMLModel().fit(train_df)
    preds     = classical.predict(test_df)
    results["classical"] = evaluate(y_test, preds)
    classical.save(model_dir / "classical")
    logger.info(f"  Classical → {results['classical']}")

    # ── Tier 3: DistilBERT ──────────────────────────────────────────────
    logger.info("Tier 3 — DistilBERT Transformer")
    transformer = TransformerModel(epochs=epochs).fit(train_df, val_df)
    preds       = transformer.predict(test_df)
    results["transformer"] = evaluate(y_test, preds)
    transformer.save(model_dir / "transformer")
    logger.info(f"  Transformer → {results['transformer']}")

    # ── Persist results ─────────────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("=" * 50)
    logger.info("FINAL TEST-SET RESULTS")
    for name, m in results.items():
        logger.info(f"  {name:12s}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.3f}  MAPE={m['mape']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all model tiers for AI Energy Predictor")
    parser.add_argument("--data-dir",  default="data/raw", help="Directory containing train/val/test CSVs")
    parser.add_argument("--model-dir", default="models",   help="Output directory for model artefacts")
    parser.add_argument("--epochs",    type=int, default=5, help="DistilBERT fine-tuning epochs")
    args = parser.parse_args()

    train_all_models(Path(args.data_dir), Path(args.model_dir), args.epochs)


if __name__ == "__main__":
    main()
