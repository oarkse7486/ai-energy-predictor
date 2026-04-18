"""
setup.py
========
End-to-end setup script for AI Energy Predictor.

Runs the full pipeline in order:
    1. Generate synthetic dataset          (scripts/make_dataset.py)
    2. Build feature matrices              (scripts/build_features.py)
    3. Train all three model tiers         (scripts/model.py)

Usage:
    python setup.py
    python setup.py --n-samples 2000 --epochs 3   # faster for quick dev runs
    python setup.py --skip-features                # if features already built
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command and raise on non-zero exit."""
    logger.info(f"{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"CMD : {' '.join(cmd)}")
    logger.info(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Step '{step_name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    logger.info(f"✓ {step_name} complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Energy Predictor — full pipeline setup")
    parser.add_argument("--n-samples",     type=int, default=5000,
                        help="Number of synthetic training samples to generate (default: 5000)")
    parser.add_argument("--epochs",        type=int, default=5,
                        help="DistilBERT fine-tuning epochs (default: 5)")
    parser.add_argument("--data-dir",      default="data/raw",       help="Raw data output directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed features directory")
    parser.add_argument("--model-dir",     default="models",         help="Model artefacts directory")
    parser.add_argument("--skip-data",     action="store_true",      help="Skip dataset generation")
    parser.add_argument("--skip-features", action="store_true",      help="Skip feature building")
    parser.add_argument("--skip-training", action="store_true",      help="Skip model training")
    parser.add_argument("--seed",          type=int, default=42,     help="Random seed")
    args = parser.parse_args()

    py = sys.executable   # use the same Python interpreter

    # ── Step 1: Generate dataset ──────────────────────────────────────────
    if not args.skip_data:
        run(
            [py, "scripts/make_dataset.py",
             "--n-samples", str(args.n_samples),
             "--output-dir", args.data_dir,
             "--seed", str(args.seed)],
            "Dataset generation",
        )
    else:
        logger.info("Skipping dataset generation (--skip-data)")

    # ── Step 2: Build feature matrices ────────────────────────────────────
    if not args.skip_features:
        run(
            [py, "scripts/build_features.py",
             "--data-dir",   args.data_dir,
             "--output-dir", args.processed_dir],
            "Feature engineering",
        )
    else:
        logger.info("Skipping feature building (--skip-features)")

    # ── Step 3: Train all models ──────────────────────────────────────────
    if not args.skip_training:
        run(
            [py, "scripts/model.py",
             "--data-dir",  args.data_dir,
             "--model-dir", args.model_dir,
             "--epochs",    str(args.epochs)],
            "Model training",
        )
    else:
        logger.info("Skipping model training (--skip-training)")

    logger.info("=" * 60)
    logger.info("Setup complete! Launch the app with:  python app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
