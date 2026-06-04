#!/usr/bin/env python3
"""
predict_video.py – Process a video with a trained YOLO-seg eye-segmentation model.

Usage:
    python scripts/predict_video.py --video /home/falcon/sana/scratch/eyeseg/data/groundtruth/2024-05-04-08-43-43/2024-05-04-08-43-43.mp4 --model /home/falcon/sana/scratch/eyeseg/runs/segment/runs/trim-bush-9/weights/best.pt --save-video
    python scripts/predict_video.py --video /path/to/video.mp4 --model /path/to/model.pt
    python scripts/predict_video.py --video data/input.mp4 --model yolo26n-seg.pt --save-video
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictor import EyeSegmentationPredictor
from src.utils import get_logger

logger = get_logger("predict_video")


def main():
    parser = argparse.ArgumentParser(
        description="Run eye segmentation inference on a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_video.py --video data/input.mp4 --model yolo26n-seg.pt
  python scripts/predict_video.py --video /home/falcon/sana/scratch/eyeseg/data/groundtruth/2024-05-04-08-43-43/2024-05-04-08-43-43.mp4 --model /home/falcon/sana/scratch/eyeseg/runs/segment/runs/trim-bush-9/weights/best.pt --save-video
  python scripts/predict_video.py --video data/video.mp4 --model yolo26n-seg.pt --imgsz 640 --conf 0.3
        """,
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file or webcam (0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO-seg model weights (.pt file).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save annotated video to predicted/ folder (default: True).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Display video in real-time window (press 'q' to quit).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for model (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID or 'cpu' (default: 0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (None = all).",
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    if isinstance(args.video, int) or (isinstance(args.video, str) and args.video == "0"):
        logger.info("Using webcam (source=0)")
    else:
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)

    # Create output directory
    predicted_dir = Path(__file__).parent.parent / "predicted"
    predicted_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    if isinstance(args.video, str) and args.video != "0":
        video_stem = video_path.stem
        output_filename = f"{video_stem}_annotated.mp4"
    else:
        output_filename = "webcam_annotated.mp4"

    output_path = predicted_dir / output_filename

    # Initialize predictor
    logger.info(f"Loading model: {model_path}")
    predictor = EyeSegmentationPredictor(
        weights=model_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    # Run inference
    logger.info(f"Processing video: {args.video}")
    save_path = output_path if args.save_video else None
    predictor.predict_video(
        source=args.video if args.video != "0" else 0,
        save_path=save_path,
        display=args.display,
        max_frames=args.max_frames,
    )

    if save_path:
        logger.info(f"Annotated video saved to: {output_path}")
    else:
        logger.info("Processing complete.")


if __name__ == "__main__":
    main()
