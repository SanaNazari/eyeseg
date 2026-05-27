"""
conda activate eyeseg
scripts/train.py
────────────────
Fine-tune (or train from scratch) a YOLOv8-seg model for eye segmentation.
All metrics and artefacts are logged to Weights & Biases.

Usage
─────
    # Standard training
    python scripts/train.py

    # Override config values via --set key=value
    python scripts/train.py --config configs/config.yaml --set epochs=50 batch_size=8

    # Resume from checkpoint
    python scripts/train.py --set model_weights=runs/my_run/weights/last.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trainer import evaluate, train
from src.utils import get_logger,load_config, resolve_paths

logger = get_logger("train_script")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 eye segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set epochs=50 device=cpu",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only evaluate model_weights on test split",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: Optional[list[str]]) -> dict:
    """Parse KEY=VALUE strings and update cfg in-place."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring malformed override: {item!r}")
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        # Try to cast to int / float
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        cfg[key] = value
        logger.info(f"Config override: {key} = {value!r}")
    return cfg


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args.set)
    cfg  = resolve_paths(cfg)

    logger.info("=" * 60)
    logger.info("  Eye Segmentation – Training")
    logger.info("=" * 60)

    if args.eval_only:
        logger.info("Evaluation-only mode.")
        evaluate(cfg, weights=cfg.get("model_weights"))
        return

    best_pt = train(cfg)
    logger.info(f"✓ Training complete. Best model → {best_pt}")

    # Auto-evaluate on test set after training
    if (cfg["_yolo_dir"] / "images" / "test").exists():
        logger.info("Running evaluation on held-out test set…")
        evaluate(cfg, weights=best_pt)


if __name__ == "__main__":
    main()
