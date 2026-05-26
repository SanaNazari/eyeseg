"""
utils.py – Shared helpers: config loading, logging, metrics, plots.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "eye_seg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and resolve key path helpers."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict) -> dict:
    """Add resolved absolute Path objects for commonly used dirs."""
    root = Path(cfg["data_root"])
    cfg["_groundtruth_dir"] = root / "groundtruth"
    cfg["_masks_dir"] = root / "masks"
    cfg["_yolo_dir"] = root / cfg.get("yolo_dataset_dir", "yolo_dataset")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Mask → YOLO polygon conversion
# ─────────────────────────────────────────────────────────────────────────────

def binary_mask_to_polygon(
    mask_path: Path,
    epsilon: float = 1.0,
    min_area: int = 50,
) -> list[list[float]] | None:
    """
    Convert a binary mask image to a list of normalised YOLO polygon segments.

    Returns a list of flat [x1,y1,x2,y2,...] normalised coordinates,
    one entry per contour (usually one), or None if mask is empty / too small.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)
        polygons.append(pts.flatten().tolist())

    return polygons if polygons else None


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "pupil":  (0,   0,   255),   # blue
    "iris":   (0,   200, 0),     # green
    "sclera": (255, 100, 0),     # orange
}

CLASS_NAMES = ["pupil", "iris", "sclera"]


def overlay_masks_on_image(
    image: np.ndarray,
    masks: list[np.ndarray],
    class_ids: list[int],
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw semi-transparent coloured masks over an image (BGR)."""
    overlay = image.copy()
    for mask, cid in zip(masks, class_ids):
        color = CLASS_COLORS[CLASS_NAMES[cid]]
        colored = np.zeros_like(image)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    return overlay


def plot_class_distribution(label_dir: Path, save_path: Path | None = None) -> plt.Figure:
    """Bar chart of per-class annotation counts in a YOLO label directory."""
    counts = {name: 0 for name in CLASS_NAMES}
    for lf in label_dir.glob("*.txt"):
        for line in lf.read_text().splitlines():
            cid = int(line.split()[0])
            if 0 <= cid < len(CLASS_NAMES):
                counts[CLASS_NAMES[cid]] += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [tuple(v / 255 for v in c) for c in CLASS_COLORS.values()]
    ax.bar(counts.keys(), counts.values(), color=colors)
    ax.set_title("Per-class annotation count")
    ax.set_ylabel("# instances")
    ax.set_xlabel("Class")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_split_distribution(
    split_counts: dict[str, int], save_path: Path | None = None
) -> plt.Figure:
    """Pie chart for train / val / test split sizes."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        split_counts.values(),
        labels=split_counts.keys(),
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Dataset split distribution")
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def symlink_or_copy(src: Path, dst: Path, use_symlink: bool = True) -> None:
    """Create a symlink (or copy as fallback) from src to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_yolo_label(label_path: Path, annotations: list[tuple[int, list[float]]]) -> None:
    """Write YOLO segmentation label file.

    Each annotation is (class_id, [x1,y1,x2,y2,...]) normalised.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, polygon in annotations:
        coords = " ".join(f"{v:.6f}" for v in polygon)
        lines.append(f"{class_id} {coords}")
    label_path.write_text("\n".join(lines))
