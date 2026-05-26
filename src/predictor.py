"""
predictor.py – Run inference with a trained YOLOv8-seg eye-segmentation model.

Supports:
  • Single image
  • Directory of images
  • Video file
  • Webcam (source=0)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import CLASS_NAMES, get_logger, overlay_masks_on_image

logger = get_logger("predictor")

# Colours per class (BGR)
_COLORS = {
    0: (0,   0,   255),   # pupil  – blue
    1: (0,   200,  0),    # iris   – green
    2: (255, 100,  0),    # sclera – orange
}


# ─────────────────────────────────────────────────────────────────────────────
# Core predictor class
# ─────────────────────────────────────────────────────────────────────────────

class EyeSegmentationPredictor:
    """Wraps a YOLOv8-seg model for eye segmentation inference."""

    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "0",
    ) -> None:
        self.weights = Path(weights)
        self.imgsz   = imgsz
        self.conf    = conf
        self.iou     = iou
        self.device  = device

        logger.info(f"Loading model from {self.weights}")
        self.model = YOLO(str(self.weights))

    # ── Single frame ─────────────────────────────────────────────────────────

    def predict_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run inference on one BGR frame.

        Returns
        ───────
        annotated_frame : np.ndarray  (BGR, same size as input)
        detections      : list of dicts with keys class_id, class_name, conf, mask
        """
        results = self.model.predict(
            source  =frame,
            imgsz   =self.imgsz,
            conf    =self.conf,
            iou     =self.iou,
            device  =self.device,
            verbose =False,
        )
        result     = results[0]
        detections = self._parse_result(result)

        if draw:
            annotated = self._draw(frame, detections)
        else:
            annotated = frame.copy()

        return annotated, detections

    def _parse_result(self, result) -> list[dict]:
        """Extract per-detection info from an Ultralytics Results object."""
        detections = []
        if result.masks is None:
            return detections

        masks   = result.masks.data.cpu().numpy()   # (N, H, W)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs   = result.boxes.conf.cpu().numpy()

        for mask, cid, conf in zip(masks, classes, confs):
            detections.append(
                {
                    "class_id"  : int(cid),
                    "class_name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
                    "conf"      : float(conf),
                    "mask"      : mask,
                }
            )
        return detections

    def _draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Overlay masks and labels on a frame."""
        out = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            cid   = det["class_id"]
            color = _COLORS.get(cid, (200, 200, 200))
            mask  = det["mask"]

            # resize mask to frame size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # semi-transparent fill
            colored = np.zeros_like(out)
            colored[mask > 0.5] = color
            out = cv2.addWeighted(out, 1.0, colored, 0.45, 0)

            # contour border
            bin_mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)

            # label
            if contours:
                cx = int(contours[0][:, 0, 0].mean())
                cy = int(contours[0][:, 0, 1].mean())
                label = f"{det['class_name']} {det['conf']:.2f}"
                cv2.putText(
                    out, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
                )
        return out

    # ── Image / directory ────────────────────────────────────────────────────

    def predict_image(
        self,
        image_path: str | Path,
        save_dir: str | Path | None = None,
    ) -> list[dict]:
        """Run inference on a single image and optionally save the result."""
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        annotated, detections = self.predict_frame(frame)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / image_path.name
            cv2.imwrite(str(out_path), annotated)
            logger.info(f"Saved annotated image → {out_path}")

        return detections

    def predict_directory(
        self,
        dir_path: str | Path,
        save_dir: str | Path | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> dict[str, list[dict]]:
        """Run inference on all images in a directory."""
        dir_path   = Path(dir_path)
        all_results: dict[str, list[dict]] = {}

        image_files = [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in extensions]
        logger.info(f"Running inference on {len(image_files)} images in {dir_path}")

        for img_path in image_files:
            dets = self.predict_image(img_path, save_dir=save_dir)
            all_results[img_path.name] = dets

        return all_results

    # ── Video / webcam ───────────────────────────────────────────────────────

    def predict_video(
        self,
        source: str | Path | int,
        save_path: str | Path | None = None,
        display: bool = False,
        max_frames: int | None = None,
    ) -> None:
        """
        Run inference on a video file or webcam stream.

        Args:
            source      : path to video file, or 0 for webcam.
            save_path   : if given, save annotated video here (.mp4).
            display     : show live window (press 'q' to quit).
            max_frames  : stop after this many frames (None = all).
        """
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w_cap, h_cap))
            logger.info(f"Saving annotated video → {save_path}")

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        logger.info(f"Processing video: {total_frames} frames @ {fps:.1f} fps")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                annotated, _ = self.predict_frame(frame)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # FPS overlay
                cv2.putText(
                    annotated,
                    f"FPS: {1000/elapsed_ms:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )

                if writer:
                    writer.write(annotated)
                if display:
                    cv2.imshow("Eye Segmentation", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(f"Done. Processed {frame_idx} frames.")
