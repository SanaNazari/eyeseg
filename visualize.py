"""
visualize.py
────────────────────────────────────────────────────────────────────────────
Overlay YOLO segmentation predictions on frames for quick visual inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Class index → (B, G, R) colour
CLASS_COLORS: Dict[int, tuple] = {
    0: (50,  50,  220),   # pupil  – red
    1: (50,  200, 50),    # iris   – green
    2: (200, 150, 50),    # sclera – blue-ish
}
CLASS_NAMES: Dict[int, str] = {0: "pupil", 1: "iris", 2: "sclera"}


def overlay_masks(
    frame: np.ndarray,
    results,                        # ultralytics Results object
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw filled semi-transparent segmentation masks + bounding boxes on a frame.

    Parameters
    ----------
    frame   : BGR image (H, W, 3).
    results : single ultralytics Results object (model(frame)[0]).
    alpha   : mask opacity.

    Returns
    -------
    Annotated BGR image.
    """
    out = frame.copy()
    overlay = frame.copy()

    if results.masks is None:
        return out

    masks_xy = results.masks.xy          # list of (N,2) polygon arrays
    classes   = results.boxes.cls.int().tolist()

    for poly, cls_id in zip(masks_xy, classes):
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        pts = poly.astype(np.int32).reshape((-1, 1, 2))

        # Filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)

        # Contour on output
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        # Class label near centroid
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        cv2.putText(
            out,
            CLASS_NAMES.get(cls_id, str(cls_id)),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # Blend overlay
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def save_prediction_grid(
    frames: List[np.ndarray],
    save_path: Path | str,
    cols: int = 4,
) -> None:
    """
    Tile a list of annotated frames into a grid and save as PNG.
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols

    # Pad to full grid
    while len(frames) < rows * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    row_imgs = [
        np.hstack(frames[i * cols : (i + 1) * cols]) for i in range(rows)
    ]
    grid = np.vstack(row_imgs)
    cv2.imwrite(str(save_path), grid)
