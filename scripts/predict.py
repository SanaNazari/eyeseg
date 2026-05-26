"""
scripts/predict.py
──────────────────
Run inference with a trained eye-segmentation model on an image,
a directory of images, or a video / webcam stream.

Usage
─────
    # Image
    python scripts/predict.py --weights runs/my_run/weights/best.pt \
                               --source path/to/image.png \
                               --save-dir outputs/predictions

    # Video file
    python scripts/predict.py --weights best.pt --source video.mp4 \
                               --save-video outputs/annotated.mp4

    # Webcam (real-time)
    python scripts/predict.py --weights best.pt --source 0 --display
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictor import EyeSegmentationPredictor
from src.utils import get_logger

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye segmentation inference")
    parser.add_argument("--weights",    required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--source",     required=True, help="Image, directory, video path, or 0 for webcam")
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--device",     type=str,   default="0")
    parser.add_argument("--save-dir",   type=str,   default=None, help="Dir to save annotated images")
    parser.add_argument("--save-video", type=str,   default=None, help="Path to save annotated video (.mp4)")
    parser.add_argument("--display",    action="store_true", help="Show live display window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = EyeSegmentationPredictor(
        weights=args.weights,
        imgsz  =args.imgsz,
        conf   =args.conf,
        iou    =args.iou,
        device =args.device,
    )

    source = args.source
    # Webcam
    if source == "0" or source.isdigit():
        predictor.predict_video(
            source   =int(source),
            save_path=args.save_video,
            display  =args.display,
        )
        return

    source_path = Path(source)

    if source_path.is_dir():
        results = predictor.predict_directory(source_path, save_dir=args.save_dir)
        logger.info(f"Processed {len(results)} images.")
        return

    # Detect video by extension
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if source_path.suffix.lower() in video_exts:
        predictor.predict_video(
            source   =source_path,
            save_path=args.save_video,
            display  =args.display,
        )
        return

    # Single image
    dets = predictor.predict_image(source_path, save_dir=args.save_dir)
    logger.info(f"Detections: {[d['class_name'] for d in dets]}")


if __name__ == "__main__":
    main()
