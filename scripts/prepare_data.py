"""
scripts/prepare_data.py
───────────────────────
Parse raw annotations, convert binary masks to YOLO polygon format,
split into train/val/test, write YOLO dataset to disk, and save
diagnostic plots.

Usage
─────
    python scripts/prepare_data.py [--config configs/config.yaml]
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config, prepare_dataset, resolve_paths
from src.utils import get_logger

logger = get_logger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO eye-segmentation dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = resolve_paths(cfg)

    logger.info("=" * 60)
    logger.info("  Eye Segmentation – Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"  data_root  : {cfg['data_root']}")
    logger.info(f"  yolo_dir   : {cfg['_yolo_dir']}")
    logger.info("=" * 60)

    yaml_path = prepare_dataset(cfg)
    logger.info(f"✓ Dataset ready. dataset.yaml → {yaml_path}")


if __name__ == "__main__":
    main()
