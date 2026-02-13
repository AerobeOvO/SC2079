#!/usr/bin/env python3
"""
API Server for Robot Navigation System
- Image recognition using YOLO model
- Path planning (to be implemented)
- Image stitching (to be implemented)
"""

from flask import Flask, request, jsonify
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import time

# Add ModelTraining and Algorithm directories to path
model_dir = Path(__file__).parent.parent / 'ModelTraining'
algo_dir = Path(__file__).parent.parent / 'Algorithm'
sys.path.insert(0, str(model_dir))
sys.path.insert(0, str(algo_dir))

# Import YOLO after adding path
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: ultralytics not installed. Image recognition will not work.")
    print("Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# Import Algorithm modules
try:
    from algo.algo import MazeSolver
    from helper import command_generator
    ALGO_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Algorithm modules not available: {e}")
    print("Path planning will not work.")
    ALGO_AVAILABLE = False

app = Flask(__name__)

# Global variables for model (loaded once at startup)
MODEL = None
MODEL_PATH = model_dir / 'bestL160epoch.pt'

# Class names matching your trained model
CLASS_NAMES = [
    'A','B','Bullseye','C','D','E','F','G','H','S','T','U','V','W','X','Y','Z',
    'circle','down','eight','five','four','left','nine','one','right',
    'seven','six','three','two','up'
]

# Map class names to IDs expected by Raspberry Pi (from consts.py SYMBOL_MAP)
CLASS_NAME_TO_ID = {
    'Bullseye': '10',
    'one': '11',
    'two': '12',
    'three': '13',
    'four': '14',
    'five': '15',
    'six': '16',
    'seven': '17',
    'eight': '18',
    'nine': '19',
    'A': '20', 'B': '21', 'C': '22', 'D': '23', 'E': '24', 'F': '25',
    'G': '26', 'H': '27', 'S': '28', 'T': '29', 'U': '30', 'V': '31',
    'W': '32', 'X': '33', 'Y': '34', 'Z': '35',
    'up': '36',      # Up Arrow
    'down': '37',    # Down Arrow
    'right': '38',   # Right Arrow
    'left': '39',    # Left Arrow
    'circle': '40'   # Stop (or could be a different symbol)
}

# Define potentially confusing class pairs
CONFUSING_CLASSES = {
    2: [29],   # Bullseye vs up arrow
    29: [2],   # up arrow vs Bullseye
    17: [2],   # circle vs Bullseye
    18: [29],  # down vs up
    22: [24],  # left vs right
    24: [22],  # right vs left
}

def load_model():
    """Load YOLO model at startup"""
    global MODEL
    if not YOLO_AVAILABLE:
        print("ERROR: Cannot load model - ultralytics not installed")
        return False
    
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return False
    
    try:
        print(f"Loading YOLO model from {MODEL_PATH}...")
        MODEL = YOLO(str(MODEL_PATH))
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

def resize_image(image, max_dimension=1280):
    """Resize image to reduce processing time while maintaining aspect ratio"""
    width, height = image.size
    
    if width <= max_dimension and height <= max_dimension:
        return image
    
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def preprocess_image(image, enhance_contrast=True):
    """Fast preprocessing optimized for API use"""
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize large images
    image = resize_image(image, max_dimension=1280)
    
    # Enhance contrast and sharpness
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(1.2)
    
    # Convert to numpy for OpenCV operations
    img_np = np.array(image)
    
    # Apply fast bilateral filter for noise reduction
    img_np = cv2.bilateralFilter(img_np, 5, 50, 50)
    
    return img_np

def filter_confusing_detections(boxes, threshold_diff=0.15):
    """Filter out potentially confusing detections"""
    if len(boxes) == 0:
        return []
    
    # Get all detections with class and confidence
    detections = []
    for i, box in enumerate(boxes):
        cls_idx = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        detections.append((i, cls_idx, conf))
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x[2], reverse=True)
    
    keep_indices = []
    
    for i, cls_idx, conf in detections:
        is_confusing = False
        
        for kept_idx in keep_indices:
            kept_cls = int(boxes[kept_idx].cls.cpu().numpy()[0])
            kept_conf = float(boxes[kept_idx].conf.cpu().numpy()[0])
            
            # Check if classes are in the confusing pairs
            if cls_idx in CONFUSING_CLASSES.get(kept_cls, []):
                if abs(conf - kept_conf) < threshold_diff:
                    is_confusing = True
                    break
        
        if not is_confusing:
            keep_indices.append(i)
    
    return keep_indices

def predict_with_model(image_path, confidence=0.35, use_preprocessing=True, use_filtering=True):
    """
    Run YOLO prediction on image
    Returns: detection results in format expected by Raspberry Pi
    """
    if MODEL is None:
        return None, "Model not loaded"
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Preprocess if enabled
        if use_preprocessing:
            processed_image = preprocess_image(image, enhance_contrast=True)
        else:
            processed_image = np.array(image)
        
        # Run prediction
        results = MODEL.predict(
            source=processed_image,
            conf=confidence,
            save=False,
            iou=0.4,
            verbose=False,
            imgsz=1280
        )
        
        # Get detection boxes
        boxes = results[0].boxes
        
        # Filter confusing detections if enabled
        if use_filtering and len(boxes) > 0:
            keep_indices = filter_confusing_detections(boxes, threshold_diff=0.15)
            boxes = [boxes[i] for i in keep_indices]
        
        # Process detections
        segments = []
        for box in boxes:
            cls_idx = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = CLASS_NAMES[cls_idx]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
            
            # Get class_id for Raspberry Pi
            class_id = CLASS_NAME_TO_ID.get(class_name, 'NA')
            
            segments.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'is_ambiguous': cls_idx in CONFUSING_CLASSES and conf < 0.7,
                'bbox': bbox
            })
        
        return segments, None
        
    except Exception as e:
        return None, str(e)

@app.route('/status', methods=['GET'])
def status():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not loaded"
    return jsonify({
        "status": "ok",
        "model": model_status,
        "yolo_available": YOLO_AVAILABLE,
        "algorithm_available": ALGO_AVAILABLE
    }), 200

@app.route('/image', methods=['POST'])
def image_recognition():
    """
    Image recognition endpoint
    Expected: multipart/form-data with 'file' field
    Returns: {"obstacle_id": "1", "segments": [...]}
    """
    if not YOLO_AVAILABLE or MODEL is None:
        return jsonify({
            "error": "Model not available",
            "obstacle_id": "NA",
            "segments": []
        }), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    
    # Extract obstacle_id from filename if present (format: timestamp_obstacleId_signal.jpg)
    obstacle_id = "1"  # default
    filename = file.filename
    if filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                obstacle_id = parts[1]
            except:
                pass
    
    # Save file temporarily
    temp_path = f"/tmp/{filename}"
    file.save(temp_path)
    
    try:
        # Run prediction using YOLO model
        segments, error = predict_with_model(
            temp_path,
            confidence=0.35,
            use_preprocessing=True,
            use_filtering=True
        )
        
        if error:
            return jsonify({
                "error": error,
                "obstacle_id": obstacle_id,
                "segments": []
            }), 500
        
        # Determine primary result (highest confidence)
        if segments and len(segments) > 0:
            # Sort by confidence
            segments.sort(key=lambda x: x['confidence'], reverse=True)
            primary_result = segments[0]
            
            # For backward compatibility, also return image_id
            response = {
                "obstacle_id": obstacle_id,
                "image_id": primary_result['class_id'],
                "segments": segments
            }
        else:
            # No detection
            response = {
                "obstacle_id": obstacle_id,
                "image_id": "NA",
                "segments": []
            }
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(response), 200
        
    except Exception as e:
        # Detailed error logging
        import traceback
        error_trace = traceback.format_exc()
        print("=" * 60)
        print("ERROR IN IMAGE RECOGNITION ENDPOINT:")
        print(error_trace)
        print("=" * 60)
        
        return jsonify({
            "error": str(e),
            "traceback": error_trace,
            "obstacle_id": obstacle_id,
            "segments": []
        }), 500

@app.route('/path', methods=['POST'])
def path_planning():
    """
    Path planning endpoint using MazeSolver algorithm
    Expected JSON: {
        "obstacles": [{"x": 5, "y": 10, "id": 1, "d": 2}],
        "robot_x": 1, "robot_y": 1, "robot_dir": 0,
        "big_turn": "0", "retrying": false
    }
    Returns: {"data": {"commands": [...], "path": [...], "distance": float}}
    """
    if not ALGO_AVAILABLE:
        return jsonify({
            "error": "Algorithm not available",
            "data": {
                "commands": [],
                "path": [],
                "distance": 0
            }
        }), 500
    
    try:
        # Get the json data from the request
        content = request.json
        
        # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction
        obstacles = content['obstacles']
        retrying = content.get('retrying', False)
        robot_x = content.get('robot_x', 1)
        robot_y = content.get('robot_y', 1)
        robot_direction = int(content.get('robot_dir', 0))
        
        print(f"Path planning request: {len(obstacles)} obstacles, robot at ({robot_x},{robot_y}), dir={robot_direction}, retrying={retrying}")
        
        # Initialize MazeSolver with robot size 20x20, at given position, facing given direction
        maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=None)
        
        # Add each obstacle to the MazeSolver
        for ob in obstacles:
            maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['id'])
        
        start_time = time.time()
        
        # Get optimal path using A* search and dynamic programming
        optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Path found in {elapsed_time:.2f}s - Distance: {distance} units")
        
        # Generate commands for the robot based on the optimal path
        commands = command_generator(optimal_path, obstacles)
        
        # Build path results - start with initial position
        path_results = [optimal_path[0].get_dict()]
        
        # Process each command and append robot position after executing it
        i = 0
        for command in commands:
            if command.startswith("SNAP") or command.startswith("FIN"):
                continue
            elif command.startswith("FW") or command.startswith("FS"):
                i += int(command[2:]) // 10
            elif command.startswith("BW") or command.startswith("BS"):
                i += int(command[2:]) // 10
            else:
                i += 1
            
            if i < len(optimal_path):
                path_results.append(optimal_path[i].get_dict())
        
        return jsonify({
            "data": {
                'distance': distance,
                'path': path_results,
                'commands': commands
            },
            "error": None
        }), 200
        
    except Exception as e:
        print(f"ERROR in path planning: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "data": {
                "commands": [],
                "path": [],
                "distance": 0
            }
        }), 500

@app.route('/stitch', methods=['GET'])
def stitch():
    """Image stitching endpoint"""
    # TODO: Implement image stitching functionality
    # This could stitch together all captured images into a panorama
    return jsonify({"status": "success"}), 200

def main():
    """Main function to start the server"""
    print("="*60)
    print("Starting Robot Navigation API Server")
    print("="*60)
    
    # Load model at startup
    if YOLO_AVAILABLE:
        model_loaded = load_model()
        if not model_loaded:
            print("WARNING: Server starting without model loaded!")
            print("Image recognition will not work until model is loaded.")
    else:
        print("WARNING: ultralytics not installed!")
        print("Install with: pip install ultralytics")
    
    print("\nServer Configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  Model: {MODEL_PATH}")
    print(f"  YOLO Available: {YOLO_AVAILABLE}")
    print(f"  Model Loaded: {MODEL is not None}")
    print(f"  Algorithm Available: {ALGO_AVAILABLE}")
    print("\nEndpoints:")
    print("  GET  /status  - Health check")
    print("  POST /image   - Image recognition (YOLO)")
    print("  POST /path    - Path planning (A* + TSP)")
    print("  GET  /stitch  - Image stitching")
    print("="*60)
    print("\nServer starting...\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
