"""
YOLO Krypton - Configuration Settings
"""

import os
from pathlib import Path

# Application Settings
APP_NAME = "YOLO Krypton"
APP_VERSION = "2.1.0"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800

# Paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
for directory in [ASSETS_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# YOLO Model Settings
DEFAULT_MODEL = "yolov8n.pt"  # nano model for faster inference
AVAILABLE_MODELS = {
    "YOLOv8n (Nano - Fastest)": "yolov8n.pt",
    "YOLOv8s (Small)": "yolov8s.pt",
    "YOLOv8m (Medium)": "yolov8m.pt",
    "YOLOv8l (Large)": "yolov8l.pt",
    "YOLOv8x (Extra Large - Most Accurate)": "yolov8x.pt"
}

# Detection Settings
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# UI Theme Settings
THEME_COLORS = {
    "primary": "#1e88e5",
    "primary_dark": "#1565c0",
    "secondary": "#00acc1",
    "success": "#43a047",
    "warning": "#fb8c00",
    "error": "#e53935",
    "background": "#1e1e1e",
    "surface": "#2d2d2d",
    "text_primary": "#ffffff",
    "text_secondary": "#b0b0b0"
}

# Supported File Formats
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

# Performance Settings
MAX_IMAGE_DISPLAY_SIZE = (1280, 720)
VIDEO_PREVIEW_FPS = 30
BATCH_SIZE = 1
FRAME_SKIP = 0  # Process every Nth frame (0 = no skip)
MAX_FPS = 30  # Maximum FPS for camera
DEVICE_OPTIONS = ["Auto", "CPU", "GPU (CUDA)"]
DEFAULT_DEVICE = "Auto"

# Export Settings
EXPORT_FORMATS = ["JSON", "CSV", "TXT", "YOLO"]
