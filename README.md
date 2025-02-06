# Object Detection and Pose Estimation Pipeline

## Overview
This repository provides an end-to-end pipeline for:
- Preparing data for YOLO-based object detection using Ultralytics’ YOLO.
- Training a YOLO model to detect a specific object.
- Training a simple pose estimation model (SimplePoseNet) on BOP-format data.

## Repository Structure
```
yolo/
  prepare_data.py   # Prepare data in YOLO format
  train.py          # Train YOLO using Ultralytics
  configs/         # YOLO configuration files (.yaml)
  models/          # YOLO trained weights (.pt)

pose/
  checkpoints/     # Stores pose model checkpoints
  models/          # Network definitions, losses, etc.
  trainers/        # Training logic for pose estimation

utils/
  data_utils.py    # BOP dataset loading + transforms
  obj_match.py     # Example usage of matching pipeline

inference/
  # Scripts for epipolar matching, YOLO detection, and match pipelines

datasets/          # Contains BOP and YOLO data
runs/              # YOLO training runs, logs, etc.
output/            # Various output images and results
```

## Environment Setup
To set up the environment, follow these steps (tested on Ubuntu with an NVIDIA GPU). The environment name is `bop`.

### Build Docker Container
```bash
cd docker/
docker build . -t bpc:2025.1.31
```

### Run Docker
```bash
docker run -p 8888:8888 --shm-size=1g --runtime nvidia --gpus all -v $(pwd):/code -ti bpc:2025.1.31 bash
cd /code
```

### Download Data
```bash
bash download_data.sh
```

## Training Pipeline

### Prepare YOLO Data
Convert BOP data to YOLO format:
```bash
python3 bpc/yolo/prepare_data.py \
    --dataset_path "datasets/train_pbr" \
    --output_path "datasets/yolo11/train_obj_8" \
    --obj_id 8
```

### Train YOLO Model
```bash
python3 bpc/yolo/train.py \
    --obj_id 11 \
    --data_path "yolo/configs/data_obj_11.yaml" \
    --epochs 20 \
    --imgsz 640 \
    --batch 16 \
    --task detection
```

### Train Pose Model
```bash
python3 train_pose.py \
  --root_dir datasets/ \
  --target_obj_id 8 \
  --epochs 50 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_workers 16 \
  --checkpoints_dir bpc/pose/pose_checkpoints/ \
  --loss_type quathi
```

### Download Pretrained Models
```bash
wget https://storage.googleapis.com/akasha-public/IPBC/baseline_solution/v1/models.zip
unzip models.zip
rm models.zip
```

### Run Inference
```bash
jupyter notebook --ip=0.0.0.0 --allow-root --port=8888
# Go to localhost:8888 on your browser
# Run "Inference Notebook.ipynb"
```

### Notes
- Ensure CUDA 12.1 drivers are installed and PyTorch recognizes the GPU (`nvidia-smi`).
- BOP dataset must follow standard conventions (train_pbr, test, etc.).
- Update `yolo/configs/data_obj_11.yaml` with the correct dataset paths.
- If encountering module import errors, try:
  ```bash
  python -m idp_codebase.pose.train ...
  ```
  or add `__init__.py` files where necessary.

