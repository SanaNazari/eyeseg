"""
dataset.py – Parse the raw eye-segmentation dataset and convert it to
YOLO segmentation format (images + polygon label .txt files + dataset.yaml).

Expected raw layout
───────────────────
data_root/
  groundtruth/<date_tag>/frames/output_<date_tag>_L/  *.png
  groundtruth/<date_tag>/frames/output_<date_tag>_R/  *.png
  masks/masks_<date_tag>/masks_<date_tag>_L/{iris,pupil,sclera}/  *.png
  masks/masks_<date_tag>/masks_<date_tag>_R/{iris,pupil,sclera}/  *.png
  CleanAnnotations.csv   (columns: subfolder, image_name)
         — one CSV per video tag, living directly inside data_root

YOLO output layout
──────────────────
yolo_dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  dataset.yaml
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import (
    binary_mask_to_polygon,
    get_logger,
    plot_class_distribution,
    plot_split_distribution,
    set_seed,
    write_yolo_label,
)

logger = get_logger("dataset")

# class-name  → YOLO class id
CLASS_MAP: dict[str, int] = {"pupil": 0, "iris": 1, "sclera": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to resolve raw-data paths
# ─────────────────────────────────────────────────────────────────────────────

def _side_from_subfolder(subfolder: str) -> str:
    """Return 'L' or 'R' from a subfolder name like '2024-05-04-08-43-43_L'."""
    return subfolder.rsplit("_", 1)[-1].upper()


def _date_tag_from_subfolder(subfolder: str) -> str:
    """Return the date tag (everything before the trailing -L/-R)."""
    return subfolder.rsplit("_", 1)[0]


def _frame_image_path(
    groundtruth_dir: Path,
    date_tag: str,
    subfolder: str,
    image_name: str,
) -> Path:
    """Resolve full path to a raw frame image."""
    return (
        groundtruth_dir
        / date_tag
        / "frames"
        / f"output_{subfolder}"
        / image_name
    )


def _mask_path(
    masks_dir: Path,
    date_tag: str,
    side: str,
    class_name: str,
    image_name: str,
) -> Path:
    """Resolve full path to a binary class mask for one frame."""
    mask_folder = masks_dir / f"masks_{date_tag}" / f"masks_{date_tag}_{side}"
    return mask_folder / class_name / f"mask_{image_name}"


# ─────────────────────────────────────────────────────────────────────────────
# CSV discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_annotation_csvs(data_root: Path) -> list[Path]:
    """Find all CleanAnnotations.csv files anywhere under data_root."""
    csvs = list(data_root.rglob("CleanAnnotations.csv"))
    logger.info(f"Found {len(csvs)} CleanAnnotations.csv file(s)")
    return csvs


def parse_annotations_csv(csv_path: Path) -> list[dict[str, str]]:
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            subfolder = (row.get("subfolder") or "").strip()
            image_name = (row.get("image_name") or "").strip()

            if not subfolder or not image_name:
                print(f"Skipping malformed row {i}: {row}")
                continue

            rows.append({
                "subfolder": subfolder,
                "image_name": image_name
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Core: build per-sample record list
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_records(
    data_root: Path,
    groundtruth_dir: Path,
    masks_dir: Path,
    min_mask_area: int = 50,
) -> list[dict]:
    """
    Walk all CSVs and produce a list of sample records.  Each record:
      {
        "image_path": Path,
        "label_data": [(class_id, [polygon_coords]), ...],
        "video_tag":  str,   # for stratified splitting
      }
    Samples without any valid mask are skipped.
    """
    csvs = discover_annotation_csvs(data_root)
    samples: list[dict] = []
    skipped = 0

    for csv_path in csvs:
        rows = parse_annotations_csv(csv_path)
        logger.info(f"Processing {csv_path.name} – {len(rows)} rows")

        for row in tqdm(rows, desc=f"  {csv_path.parent.name}", leave=False):
            subfolder = row["subfolder"]
            image_name = row["image_name"]
            side = _side_from_subfolder(subfolder)
            date_tag = _date_tag_from_subfolder(subfolder)

            image_path = _frame_image_path(groundtruth_dir, date_tag, subfolder, image_name)
            if not image_path.exists():
                print("MISSING IMAGE:", image_path)
                skipped += 1
                continue

            annotations: list[tuple[int, list[float]]] = []
            for class_name, class_id in CLASS_MAP.items():
                m_path = _mask_path(masks_dir, date_tag, side, class_name, image_name)
                if not m_path.exists():
                    print("MISSING MASK:", m_path)
                    continue
                polygons = binary_mask_to_polygon(m_path, min_area=min_mask_area)
                if polygons:
                    for poly in polygons:
                        annotations.append((class_id, poly))

            if not annotations:
                skipped += 1
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "label_data": annotations,
                    "video_tag": date_tag,
                }
            )

    logger.info(f"Valid samples: {len(samples)}  |  Skipped: {skipped}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────

def split_samples(
    samples: list[dict],
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Stratified train / val / test split based on video_tag so that
    no video contributes frames only to one split.
    """
    set_seed(seed)
    indices = list(range(len(samples)))
    tags = [s["video_tag"] for s in samples]

    test_frac = test_split / (1.0 - val_split) if test_split > 0 else 0.0
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=tags, random_state=seed
    ) if test_split > 0 else (indices, [])

    train_val_tags = [tags[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1.0 - test_split),
        stratify=train_val_tags,
        random_state=seed,
    )

    def _select(idx_list: list[int]) -> list[dict]:
        return [samples[i] for i in idx_list]

    splits = {
        "train": _select(train_idx),
        "val":   _select(val_idx),
        "test":  _select(test_idx),
    }
    for split, s in splits.items():
        logger.info(f"  {split:5s}: {len(s)} samples")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Write YOLO dataset to disk
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_dataset(
    splits: dict[str, list[dict]],
    yolo_dir: Path,
    use_symlink: bool = True,
) -> Path:
    """
    Materialise the YOLO dataset on disk and return the path to dataset.yaml.
    Images are symlinked (or copied) to avoid duplicating large files.
    """
    yolo_dir.mkdir(parents=True, exist_ok=True)

    for split, samples in splits.items():
        img_dir   = yolo_dir / "images" / split
        label_dir = yolo_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=f"Writing {split}"):
            src_img = sample["image_path"]
            stem    = src_img.stem
            suffix  = src_img.suffix

            dst_img   = img_dir   / (stem + suffix)
            dst_label = label_dir / (stem + ".txt")

            # image
            if not dst_img.exists():
                if use_symlink:
                    try:
                        dst_img.symlink_to(src_img.resolve())
                    except OSError:
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

            # label
            write_yolo_label(dst_label, sample["label_data"])

    # dataset.yaml
    yaml_path = yolo_dir / "dataset.yaml"
    class_names = list(CLASS_MAP.keys())
    yaml_content = (
        f"path: {yolo_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path.write_text(yaml_content)
    logger.info(f"dataset.yaml written → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset_plots(yolo_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Save class-distribution and split-distribution plots; return path map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # Class distribution (train labels)
    train_label_dir = yolo_dir / "labels" / "train"
    if train_label_dir.exists():
        p = output_dir / "class_distribution.png"
        plot_class_distribution(train_label_dir, save_path=p)
        saved["class_distribution"] = p

    # Split sizes
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        d = yolo_dir / "labels" / split
        split_counts[split] = len(list(d.glob("*.txt"))) if d.exists() else 0
    p = output_dir / "split_distribution.png"
    plot_split_distribution(split_counts, save_path=p)
    saved["split_distribution"] = p

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(cfg: dict[str, Any]) -> Path:
    """
    Full pipeline: parse annotations → build samples → split → write YOLO dataset.
    Returns the path to dataset.yaml.
    """
    data_root       = Path(cfg["data_root"])
    groundtruth_dir = cfg["_groundtruth_dir"]
    masks_dir       = cfg["_masks_dir"]
    yolo_dir        = cfg["_yolo_dir"]

    samples = build_sample_records(
        data_root,
        groundtruth_dir,
        masks_dir,
        min_mask_area=cfg.get("min_mask_area", 50),
    )

    splits = split_samples(
        samples,
        val_split =cfg.get("val_split",  0.10),
        test_split=cfg.get("test_split", 0.10),
        seed      =cfg.get("seed", 42),
    )

    yaml_path = write_yolo_dataset(splits, yolo_dir)

    # Save diagnostic plots alongside dataset
    plots_dir = yolo_dir / "plots"
    save_dataset_plots(yolo_dir, plots_dir)

    return yaml_path
