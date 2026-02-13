# API Server - Robot Navigation System

## Overview

This API server provides image recognition and path planning services for the robot navigation system. It integrates the YOLO model from the `ModelTraining` folder to perform real-time symbol detection.

## What Was Integrated

✅ **YOLO Model Integration** - The trained model (`bestL160epoch.pt`) from `ModelTraining/` folder has been integrated into the API server with:

- **Image Preprocessing**: Contrast enhancement, sharpness adjustment, and noise reduction
- **Confusion Filtering**: Filters out ambiguous detections (e.g., Bullseye vs Up Arrow)
- **Class ID Mapping**: Automatically maps detected classes to the IDs expected by Raspberry Pi
- **Optimized Performance**: Auto-resizes large images for faster processing (3-5 seconds)

## Installation

### 1. Install Python Dependencies

```bash
cd Main
pip install -r requirements.txt
```

**Note**: PyTorch installation may take some time depending on your system.

### 2. Verify Model File

Ensure the trained model exists:
```bash
ls ../ModelTraining/bestL160epoch.pt
```

If the file is missing, you need to train the model or copy it from your training environment.

## Running the Server

### Start the API Server

```bash
python3 api_server.py
```

You should see output like:
```
============================================================
Starting Robot Navigation API Server
============================================================
Loading YOLO model from /path/to/bestL160epoch.pt...
✅ Model loaded successfully!

Server Configuration:
  Host: 0.0.0.0
  Port: 5000
  Model: /path/to/bestL160epoch.pt
  YOLO Available: True
  Model Loaded: True

Endpoints:
  GET  /status  - Health check
  POST /image   - Image recognition
  POST /path    - Path planning
  GET  /stitch  - Image stitching
============================================================
```

### Find Your Server IP Address

**On Mac/Linux:**
```bash
ifconfig | grep "inet "
```

**On Raspberry Pi:**
Update the `settings.py` file in your robot scripts with this IP address.

## API Endpoints

### 1. Health Check - `/status` (GET)

Check if the server and model are running:

```bash
curl http://localhost:5000/status
```

**Response:**
```json
{
  "status": "ok",
  "model": "loaded",
  "yolo_available": true
}
```

### 2. Image Recognition - `/image` (POST)

Send an image for symbol detection:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/image
```

**Response Format:**
```json
{
  "obstacle_id": "1",
  "image_id": "39",
  "segments": [
    {
      "class_id": "39",
      "class_name": "left",
      "confidence": 0.89,
      "is_ambiguous": false
    }
  ]
}
```

**Response Fields:**
- `obstacle_id`: Extracted from filename or "1" by default
- `image_id`: Primary detected symbol ID (for backward compatibility)
- `segments`: Array of all detected symbols sorted by confidence
  - `class_id`: Symbol ID (matches SYMBOL_MAP in consts.py)
  - `class_name`: Human-readable name
  - `confidence`: Detection confidence (0-1)
  - `is_ambiguous`: Warning flag for low-confidence confusing classes

**Symbol ID Mapping:**
```
10 = Bullseye    20-35 = A-Z      36 = Up Arrow
11-19 = 1-9      17 = circle      37 = Down Arrow
                                   38 = Right Arrow
                                   39 = Left Arrow
                                   40 = Stop/circle
```

### 3. Path Planning - `/path` (POST)

Request path from algorithm (TODO - implement your algorithm):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"obstacles":[{"x":5,"y":10,"id":1,"d":2}],"robot_x":1,"robot_y":1,"robot_dir":0}' \
  http://localhost:5000/path
```

### 4. Image Stitching - `/stitch` (GET)

Stitch captured images (TODO - implement stitching):

```bash
curl http://localhost:5000/stitch
```

## Testing the Image Recognition

### Test with Sample Images

If you have test images in the `TestPic` folder:

```bash
# Test with a specific image
curl -X POST -F "file=@../TestPic/1758859514_2_C.jpg" http://localhost:5000/image

# Or use Python
python3 << EOF
import requests

with open('../TestPic/1758859514_2_C.jpg', 'rb') as f:
    files = {'file': ('test.jpg', f)}
    response = requests.post('http://localhost:5000/image', files=files)
    print(response.json())
EOF
```

### Test from Raspberry Pi

Update `settings.py` on your Raspberry Pi:
```python
API_IP = '192.168.1.100'  # Your laptop's IP address
API_PORT = 5000
```

Then run your robot script (e.g., `Week_9_final.py`).

## Features Integrated from ModelTraining

### 1. Image Preprocessing
- **Auto-resize**: Large images (>1280px) automatically resized
- **Contrast enhancement**: 30% increase for better feature detection
- **Sharpness boost**: 20% increase for clearer edges
- **Noise reduction**: Fast bilateral filtering

### 2. Confusion Filtering
Prevents misidentification of similar symbols:
- Bullseye ↔ Up Arrow
- Left Arrow ↔ Right Arrow
- Down Arrow ↔ Up Arrow
- Circle ↔ Bullseye

### 3. Optimized Performance
- **Processing time**: 3-5 seconds per image (down from 54s for large images)
- **Model caching**: Loaded once at startup
- **Efficient inference**: Fixed image size (1280px) for consistent performance

## Integration with Robot Scripts

The API server is designed to work with your Week 8 and Week 9 robot scripts:

1. **Week_8.py**: Full autonomous navigation
   - Sends images via `snap_and_rec()` function
   - Receives symbol IDs and updates obstacle tracking
   
2. **Week_9_final.py**: CLK-triggered detection
   - Captures images on STM32 CLK command
   - Gets arrow directions for navigation decisions

3. **A_5.py**: Standalone mode
   - Captures multiple faces per obstacle
   - Retries until valid symbol detected

## Troubleshooting

### Model Not Loading

```
ERROR: Model file not found at /path/to/bestL160epoch.pt
```

**Solution**: Copy your trained model to the `ModelTraining` folder:
```bash
cp /path/to/your/model/bestL160epoch.pt ../ModelTraining/
```

### Ultralytics Not Installed

```
WARNING: ultralytics not installed
```

**Solution**: Install dependencies:
```bash
pip install ultralytics torch torchvision
```

### Slow Performance

If detection takes >10 seconds:
1. Check image size - large images (>1280px) are auto-resized but may still be slow on older hardware
2. Ensure GPU support if available (PyTorch will auto-detect)
3. Consider reducing confidence threshold or disabling preprocessing

### Port Already in Use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Kill existing process or change port:
```bash
# Find and kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or edit api_server.py to use different port
app.run(host='0.0.0.0', port=5001, debug=True)
```

## Next Steps (TODO)

### 1. Path Planning Algorithm
Replace the mock response in `/path` endpoint with your actual algorithm:

```python
# In path_planning() function
from your_algorithm_module import PathPlanner

planner = PathPlanner()
result = planner.plan(data['obstacles'], data['robot_x'], 
                     data['robot_y'], data['robot_dir'])
```

### 2. Image Stitching
Implement panorama creation in `/stitch` endpoint:
- Collect all captured images from a run
- Use OpenCV stitching or similar
- Return stitched result

### 3. Enhanced Logging
Add logging for debugging:
- Save detection results to database
- Log failed detections for model improvement
- Track API performance metrics

## Performance Metrics

Based on ModelTraining optimizations:

| Metric | Value |
|--------|-------|
| Average inference time | 3-5 seconds |
| Preprocessing time | ~0.5 seconds |
| Supported image size | Up to 4284x5712 (auto-resized) |
| Detection accuracy | ~95% (with filtering) |
| Confidence threshold | 0.35 (configurable) |

## Credits

- YOLO model training: ModelTraining folder
- Optimization techniques: app_optimized.py and rpi_camera_app_improved.py
- Integration: Main/api_server.py

---

**Last Updated**: 2026/2/12

For more information, see `SETUP_GUIDE.md` in the parent directory.
