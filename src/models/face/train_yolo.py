import os
import torch
import shutil
import argparse
from datetime import datetime
from ultralytics import YOLO
from src.constants import FaceDetection
from src.config import FaceConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--device', type=str, default='0,1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--config', type=str, default=str(FaceDetection.DATA_CONFIG_PATH),
                      help=f'Path to YOLO data config file (default: {FaceDetection.DATA_CONFIG_PATH})')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_output_dir = FaceDetection.OUTPUT_DIR

    # Load the YOLO model - changed default to 'm'
    model_name = FaceConfig.MODEL_NAME
    print(f"Loading model: {model_name}")

    # huggingface yolo12 face model
    #model = YOLO(FaceDetection.FACE_MODEL_PATH)
    model = YOLO(f"{FaceConfig.MODEL_NAME}.pt")

    experiment_name = f"{FaceConfig.MODEL_NAME}_{timestamp}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print("-" * 50)

    # Determine config file path
    if args.config:
        config_file_path = FaceDetection.DATA_CONFIG_PATH.parent / args.config

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=config_file_path,
        epochs=FaceConfig.NUM_EPOCHS,
        imgsz=FaceConfig.IMG_SIZE,
        batch=FaceConfig.BATCH_SIZE,
        project=str(base_output_dir),
        name=experiment_name,
        
        patience=FaceConfig.PATIENCE,
        workers=FaceConfig.NUM_WORKERS,
        device=args.device,
        degrees=FaceConfig.DEGREES,

        augment=True,
        mosaic=1.0,
        scale=0.5,
        
        plots=True,
        val=True,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_{FaceConfig.MODEL_NAME}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()