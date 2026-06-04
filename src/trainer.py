"""
trainer.py – Train (or fine-tune) a YOLOv8-seg model for eye segmentation
with full Weights & Biases integration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
import yaml
from ultralytics import YOLO
from ultralytics.utils.callbacks.wb import callbacks as wb_callbacks

from .utils import (
    get_logger,
    plot_class_distribution,
    plot_split_distribution,
    set_seed,
)

logger = get_logger("trainer")


# ─────────────────────────────────────────────────────────────────────────────
# W&B setup
# ─────────────────────────────────────────────────────────────────────────────

def init_wandb(cfg: dict) -> wandb.sdk.wandb_run.Run:
    """Initialise a W&B run and return the run object."""
    wb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wb_cfg.get("project", "eye-segmentation"),
        entity=wb_cfg.get("entity") or None,
        tags=wb_cfg.get("tags", []),
        config={
            k: v
            for k, v in cfg.items()
            if not k.startswith("_") and k != "wandb"
        },
        resume="allow",
    )
    logger.info(f"W&B run initialised: {run.url}")
    return run


def log_dataset_plots(run: wandb.sdk.wandb_run.Run, yolo_dir: Path) -> None:
    """Upload pre-computed dataset diagnostic plots to W&B."""
    plots_dir = yolo_dir / "plots"
    if not plots_dir.exists():
        return
    for png in plots_dir.glob("*.png"):
        run.log({f"dataset/{png.stem}": wandb.Image(str(png))})


def log_best_model_artifact(
    run: wandb.sdk.wandb_run.Run,
    best_pt: Path,
    cfg: dict,
) -> None:
    """Upload best.pt as a versioned W&B model artifact."""
    if not best_pt.exists():
        logger.warning(f"best.pt not found at {best_pt}; skipping artifact upload.")
        return
    artifact = wandb.Artifact(
        name="eye-seg-yolo",
        type="model",
        description="YOLOv8-seg fine-tuned for eye segmentation (pupil/iris/sclera)",
        metadata={k: v for k, v in cfg.items() if not k.startswith("_") and k != "wandb"},
    )
    artifact.add_file(str(best_pt), name="best.pt")
    run.log_artifact(artifact, aliases=["best", "latest"])
    logger.info(f"Model artifact uploaded: {best_pt}")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> YOLO:
    weights = cfg.get("model_weights", "").strip()
    if weights and Path(weights).exists():
        logger.info(f"Loading model from checkpoint: {weights}")
        return YOLO(weights)

    variant = cfg.get("model_variant", "yolov8n-seg")
    if cfg.get("pretrained", True):
        model_name = f"{variant}.pt"      # downloads pretrained weights
    else:
        model_name = f"{variant}.yaml"    # random init from architecture config

    logger.info(f"Loading model: {model_name}")
    return YOLO(model_name)

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> Path:
    """
    Full training loop.  Returns the path to best.pt.

    Steps
    ─────
    1. Init W&B.
    2. Log dataset diagnostics.
    3. Build model.
    4. Run YOLO .train() — Ultralytics handles its own W&B callback automatically.
    5. Upload best.pt as W&B artifact.
    6. Finish W&B run.
    """
    set_seed(cfg.get("seed", 42))
    yolo_dir   = cfg["_yolo_dir"]
    output_dir = Path(cfg.get("output_dir", "runs"))
    dataset_yaml = yolo_dir / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}. Run prepare_data.py first."
        )

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = init_wandb(cfg)
    log_dataset_plots(run, yolo_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)

    # ── Training args ────────────────────────────────────────────────────────
    train_args = dict(
        data        = str(dataset_yaml),
        epochs      = cfg.get("epochs", 100),
        batch       = cfg.get("batch_size", 16),
        imgsz       = cfg.get("imgsz", 640),
        optimizer   = cfg.get("optimizer", "AdamW"),
        lr0         = cfg.get("lr0", 0.001),
        lrf         = cfg.get("lrf", 0.01),
        momentum    = cfg.get("momentum", 0.937),
        weight_decay= cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        patience    = cfg.get("patience", 20),
        device      = cfg.get("device", "0"),
        workers     = cfg.get("workers", 8),
        amp         = cfg.get("amp", True),
        project     = str(output_dir),
        name        = run.name,  # tie YOLO run name to W&B run name
        exist_ok    = True,
        # Augmentation
        degrees     = cfg.get("degrees", 10.0),
        translate   = cfg.get("translate", 0.1),
        scale       = cfg.get("scale", 0.5),
        fliplr      = cfg.get("fliplr", 0.5),
        mosaic      = cfg.get("mosaic", 0.5),
        copy_paste  = cfg.get("copy_paste", 0.3),
        # W&B is integrated natively by Ultralytics
        plots       = True,
        save        = True,
        save_period = 100,
        verbose     = True,
    )

    logger.info("Starting training…")
    results = model.train(**train_args)

    # ── Retrieve best.pt ──────────────────────────────────────────────────────
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_pt}")

    # ── Log extra metrics to W&B ─────────────────────────────────────────────
    _log_final_metrics(run, results)

    # ── Upload artifact ───────────────────────────────────────────────────────
    if cfg.get("wandb", {}).get("log_model", True):
        log_best_model_artifact(run, best_pt, cfg)

    wandb.finish()
    return best_pt


def _log_final_metrics(run: wandb.sdk.wandb_run.Run, results: Any) -> None:
    """Extract and log final validation metrics to W&B summary."""
    try:
        metrics = results.results_dict  # dict of metric_name → value
        for k, v in metrics.items():
            run.summary[k] = v
        logger.info(f"Final metrics logged: {list(metrics.keys())}")
    except Exception as exc:
        logger.warning(f"Could not extract final metrics: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: dict, weights: str | Path | None = None) -> None:
    """Run YOLO .val() on the test split and log results to W&B."""
    yolo_dir     = cfg["_yolo_dir"]
    dataset_yaml = yolo_dir / "dataset.yaml"

    weights = weights or cfg.get("model_weights", "")
    if not weights or not Path(str(weights)).exists():
        raise FileNotFoundError("Provide a valid model_weights path for evaluation.")

    run = init_wandb(cfg)
    model = YOLO(str(weights))

    logger.info("Evaluating on test split…")
    results = model.val(
        data  =str(dataset_yaml),
        split ="test",
        imgsz =cfg.get("imgsz", 640),
        device=cfg.get("device", "0"),
        plots =True,
    )

    _log_final_metrics(run, results)
    wandb.finish()
