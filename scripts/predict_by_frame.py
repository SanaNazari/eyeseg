import argparse
import glob
from pathlib import Path
from ultralytics import YOLO
import cv2


def find_frame_file(frame_path: Path) -> Path:
    if frame_path.exists():
        return frame_path
    raise FileNotFoundError(f'Frame not found at {frame_path}')




def run_inference(frame_path: str, model_path: str, output_dir: str) -> None:
    frame_dir = Path(frame_path)
    output_path = Path(output_dir)
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f'Model file not found at {model_file}')
    output_path.mkdir(parents=True, exist_ok=True)

    frame_file = find_frame_file(frame_dir)
    
    model = YOLO(str(model_file))
    results = model.predict(source=str(frame_file), save=False)
    if not results:
        raise RuntimeError('No inference results returned')

    result = results[0]
    annotated = result.plot()
    frame_id = frame_file.stem
    save_file = output_path / f'yolo_{frame_id}.png'
    cv2.imwrite(str(save_file), annotated)
    print(f'Saved inference result to {save_file}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO inference on a single extracted frame and save PNG output.')
    parser.add_argument('--frame-path', 
                        default='/home/falcon/sana/scratch/eyeseg/data/groundtruth/2024-05-04-08-43-43/frames/output_2024-05-04-08-43-43_R/2024-05-04-08-43-43_frame_0004905_R.png' ,
                         help='Path to the frame you want to predict.')
    parser.add_argument('--model-path', 
                        default='/home/falcon/sana/scratch/eyeseg/runs/segment/runs/trim-bush-9/weights/best.pt',
                         help='Directory containing saved YOLO model file.')
    parser.add_argument('--output-dir', 
                        default='predicted',
                          help='Directory to save annotated PNG results.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(args.frame_path, args.model_path, args.output_dir)
