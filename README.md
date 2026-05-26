# Eye Segmentation with YOLO

Real-time eye segmentation (pupil, iris, sclera) using YOLOv8-seg, with Weights & Biases tracking.

## Project Structure

```
eye_segmentation/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── src/
│   ├── dataset.py           # Dataset preparation & YOLO conversion
│   ├── trainer.py           # Training loop with W&B integration
│   ├── predictor.py         # Inference & real-time prediction
│   └── utils.py             # Helpers: metrics, plots, I/O
├── scripts/
│   ├── prepare_data.py      # Run dataset preparation only
│   ├── train.py             # Run training
│   └── predict.py           # Run inference on image/video
├── requirements.txt
└── README.md
```

## Data Structure Expected

```
data_root/
├── groundtruth/
│   └── <date_tag>/
│       └── frames/
│           ├── output_<date_tag>_L/   # left eye frames
│           └── output_<date_tag>_R/   # right eye frames
├── masks/
│   └── masks_<date_tag>/
│       ├── masks_<date_tag>_L/
│       │   ├── iris/
│       │   ├── pupil/
│       │   └── sclera/
│       └── masks_<date_tag>_R/
│           ├── iris/
│           ├── pupil/
│           └── sclera/
└── CleanAnnotations.csv     # per video tag (subfolder, image_name)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure paths
Edit `configs/config.yaml` to set your `data_root` and W&B project name.

### 3. Prepare dataset (converts masks → YOLO polygon format)
```bash
python scripts/prepare_data.py
```

### 4. Train
```bash
python scripts/train.py
```

### 5. Predict
```bash
python scripts/predict.py --source path/to/image_or_video.mp4
```

## Classes
| ID | Name   |
|----|--------|
| 0  | pupil  |
| 1  | iris   |
| 2  | sclera |

## Notes
- Uses **YOLOv8n-seg** by default (nano). Change `model_size` in config to `s`, `m`, `l`, `x` for larger models.
- Multi-GPU training is supported via `device: 0,1` in config.
- All metrics and sample predictions are logged to Weights & Biases.
