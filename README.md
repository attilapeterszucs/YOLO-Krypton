# 🚀 YOLO Krypton

**Advanced Object Detection System with GPU Acceleration**

A professional, camera-first GUI application for real-time object detection using YOLOv8 (Ultralytics). YOLO Krypton starts with live camera detection by default and provides an intuitive interface for detecting objects in real-time, with additional support for images and videos through a convenient menu system.

## 🎮 GPU Support
This application supports both CPU and NVIDIA GPU processing for optimal performance. With GPU acceleration, achieve up to 10x faster detection speeds!

## ✨ Features

### Core Functionality
- **Camera-First Design**: Automatically starts with live camera detection
- **Multiple YOLO Models**: Support for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x
- **Multiple Input Sources**: 
  - Live webcam feed (default)
  - Static images (JPG, PNG, BMP, TIFF, WebP)
  - Video files (MP4, AVI, MOV, MKV, WMV)
- **Real-time Detection**: Continuous object detection from camera
- **Snapshot Feature**: Capture and save frames from live camera feed
- **Batch Processing**: Process entire videos frame by frame

### Advanced Features
- **GPU/CPU Selection**: 
  - Automatic GPU detection
  - Hot-swap between CPU and GPU
  - VRAM usage monitoring
- **Performance Optimization**:
  - Frame skipping (0-5 levels)
  - Real-time FPS counter
  - Optimized video processing
- **Adjustable Parameters**: 
  - Confidence threshold slider (0.1 - 1.0)
  - IOU threshold slider for NMS (0.1 - 1.0)
- **Multiple Export Formats**: JSON, CSV, TXT, YOLO format
- **Statistics Dashboard**: 
  - Total objects detected
  - Unique classes found
  - Average confidence scores
  - Processing time metrics
  - Class distribution visualization
- **Professional UI**: 
  - Dark theme with modern design
  - Tabbed interface for different views
  - Real-time progress indicators
  - Status bar with live updates

### UI/UX Elements
- **Menu Bar**: Professional menu system for easy navigation
- **Camera Controls**: Dedicated pause/resume and snapshot buttons
- **Source Indicator**: Visual indication of current input source
- **Keyboard Shortcuts**: 
  - `Ctrl+O`: Open image
  - `Ctrl+V`: Open video
  - `Ctrl+W`: Switch to camera
  - `Ctrl+S`: Take snapshot
  - `Ctrl+E`: Export results
  - `Space`: Pause/Resume camera
  - `Esc`: Stop detection
- **Visual Feedback**: 
  - Color-coded bounding boxes for different object classes
  - Real-time object count
- **Responsive Design**: Resizable window with adaptive layout

## 📋 Requirements

### Minimum Requirements
- Python 3.8+
- Windows/Linux/macOS
- Webcam (optional, for live detection)
- 4GB+ RAM recommended

### For GPU Acceleration (Optional but Recommended)
- NVIDIA GPU with CUDA Compute Capability 3.5+
- NVIDIA Driver 450.80.02 or higher
- CUDA 11.8 compatible GPU
- 4GB+ VRAM recommended

## 🚀 Installation

### Option 1: With GPU Support (Recommended for NVIDIA GPUs)

1. Clone the repository:
```bash
git clone https://github.com/attilapeterszucs/YOLO-Krypton.git
cd YOLO-Krypton
```

2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: CPU Only Installation

1. Clone the repository:
```bash
git clone https://github.com/attilapeterszucs/YOLO-Krypton.git
cd YOLO-Krypton
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🏃 Running the Application

```bash
python main.py
```

The application will automatically download the required YOLO models on first use.

## 📖 Usage Guide

### Getting Started

1. **Launch YOLO Krypton** by running `python main.py`
2. **Camera starts automatically** - YOLO Krypton will begin detecting objects from your webcam immediately
3. **Adjust detection parameters** (optional):
   - Confidence Threshold: Higher values = fewer but more confident detections
   - IOU Threshold: Controls overlap tolerance for duplicate detections
4. **Select a different YOLO model** from the Model menu if needed (YOLOv8n is default for real-time performance)

### Using Different Input Sources

#### Camera (Default):
- **Starts automatically** when the application launches
- Use **"⏸ Pause Camera"** to temporarily stop
- Use **"▶ Resume Camera"** to continue
- Click **"📸 Take Snapshot"** to save current frame

#### For Images:
1. Go to **File → Open Image** (or press `Ctrl+O`)
2. Select an image file from your computer
3. Detection runs automatically

#### For Videos:
1. Go to **File → Open Video** (or press `Ctrl+V`)
2. Select a video file
3. Processing starts automatically

#### Quick Switch:
- Use the **quick switch buttons** (📹 📷 🎥) in the sidebar
- Or use **File menu** options
- Press `Ctrl+W` to quickly return to camera

### Viewing Results

- **Detection View Tab**: Shows the annotated image/video with bounding boxes
- **Statistics Tab**: Displays detailed metrics and class distribution
- **Results Panel**: Lists all detected objects with confidence scores and locations

### Exporting Results
1. Select export format from dropdown (JSON, CSV, TXT, or YOLO)
2. Click **"💾 Export Results"**
3. Choose save location
4. Results will be saved in selected format

## 🏗️ Project Structure

```
YOLO Krypton/
├── main.py              # Main application with camera-first design
├── yolo_detector.py     # YOLO detection engine
├── ui_components.py     # Reusable UI components
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
├── models/            # YOLO model files (auto-created)
├── output/            # Export results and snapshots (auto-created)
├── temp/              # Temporary files (auto-created)
└── assets/            # Application assets (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Default model selection
- Window dimensions
- Detection parameters
- Color themes
- File format support
- Performance settings

## 🎯 Model Information

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.3 MB | Fastest | Good | Real-time applications |
| YOLOv8s | 22.5 MB | Fast | Better | Balanced performance |
| YOLOv8m | 52 MB | Medium | High | General purpose |
| YOLOv8l | 87.7 MB | Slower | Higher | High accuracy needed |
| YOLOv8x | 137 MB | Slowest | Highest | Maximum accuracy |

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'ultralytics'"**
   - Solution: Run `pip install ultralytics`

2. **Webcam not working**
   - Ensure webcam is connected and not used by another application
   - Check webcam permissions in system settings

3. **Slow performance**
   - Use a smaller model (YOLOv8n or YOLOv8s)
   - Reduce input image/video resolution
   - Enable GPU acceleration if available

4. **CUDA/GPU not detected**
   - Install CUDA toolkit and cuDNN
   - Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 🚦 Performance Tips

- **For real-time camera detection**: Use YOLOv8n (default) with moderate confidence threshold
- **For accuracy**: Use YOLOv8l or YOLOv8x with higher confidence threshold
- **Camera optimization**: The app automatically optimizes for real-time performance
- **For batch processing**: Switch to video mode for offline processing
- **GPU acceleration**: Significantly improves processing speed
- **Snapshot feature**: Capture interesting detections without interrupting the feed

## 📝 Export Formats

- **JSON**: Structured data with all detection details
- **CSV**: Spreadsheet-compatible format
- **TXT**: Simple text format with basic information
- **YOLO**: Standard YOLO annotation format for training

## 🎨 Features Showcase

- **Professional Dark Theme**: Easy on the eyes for extended use
- **Real-time Statistics**: Live updates of detection metrics
- **Multi-threaded Processing**: Non-blocking UI during detection
- **Smart Class Coloring**: Consistent colors for each object class
- **Progress Indicators**: Visual feedback for long operations

## 📄 License

This project is created for educational and professional use. Feel free to modify and extend it according to your needs.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Optimize performance

## 📧 Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

---

**Developed with ❤️ using YOLOv8 by Ultralytics and CustomTkinter**
