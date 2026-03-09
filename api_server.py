#!/usr/bin/env python3
"""
API Server for Robot Navigation System
- Image recognition using YOLO model  (POST /image, POST /image-new)
- Path planning via MazeSolver A* + TSP  (POST /path)
- Image stitching  (GET /stitch)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import io
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import time

# Add ModelTraining and Algorithm directories to path
model_dir = Path(__file__).parent.parent / 'ModelTraining'
algo_dir = Path(__file__).parent.parent / 'Algo' / 'Algorithm'
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
CORS(app)

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

# CLASS_NAMES (for reference when editing this dict):
# [0]A [1]B [2]Bullseye [3]C [4]D [5]E [6]F [7]G [8]H [9]S [10]T
# [11]U [12]V [13]W [14]X [15]Y [16]Z [17]circle [18]down [19]eight
# [20]five [21]four [22]left [23]nine [24]one [25]right [26]seven
# [27]six [28]three [29]two [30]up

# Define potentially confusing class pairs (symmetric).
# Values are lists of class indices that this index can be confused WITH.
CONFUSING_CLASSES = {
    2:  [30],       # Bullseye  ↔ Up Arrow   (both have circular/round shapes)
    30: [2, 18],    # Up Arrow  ↔ Bullseye AND Up ↔ Down  (NOTE: no dup keys)
    17: [2],        # circle    ↔ Bullseye   (both round)
    18: [30],       # Down      ↔ Up         (mirror images of each other)
    22: [25],       # Left      ↔ Right      (mirror images)
    25: [22],       # Right     ↔ Left
    6:  [8],        # F         ↔ H          (shadow/texture adds phantom right stroke to F)
    8:  [6],        # H         ↔ F
    7:  [10],       # G         ↔ T          (G's curved top arc misread as T crossbar
    10: [7],        #                          on dark/colored/wood-texture backgrounds)
    9:  [28],       # S         ↔ Three      (curved shapes look similar at distance)
    28: [9],        # Three     ↔ S
}

# Classes that trigger the background-suppression disambiguation pass when
# detected with confidence below this threshold.
_AMBIGUITY_CONF_THRESHOLD = 0.65
_AMBIGUOUS_CLASSES = {'H', 'G', 'T', 'F', 'S', 'three'}


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

def preprocess_image_for_textured_background(image):
    """
    Enhanced preprocessing specifically for images with busy/textured backgrounds
    (e.g. checkered, crumpled paper, fabric patterns).

    Strategy:
      1. Convert to LAB color space to separate luminance from color.
      2. Apply CLAHE on the L-channel to boost local contrast of dark letter
         shapes against the repetitive background pattern.
      3. Merge back and convert to RGB.
      4. Apply a mild bilateral filter to smooth background noise while
         preserving the hard edges of the letter strokes.
    """
    if isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        img_np = np.array(image)

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. CLAHE on LAB luminance channel
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Bilateral filter to smooth texture noise
    img_np = cv2.bilateralFilter(img_np, 9, 75, 75)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np

def preprocess_suppress_background(image):
    """
    Background-suppression preprocessing: replaces the textured background
    with pure white while keeping the dark letter strokes intact.

    This is specifically useful when textured backgrounds (crumpled paper,
    fabric shadows) create phantom dark strokes that cause F to be misread
    as H (shadow on the right of F looks like H's second vertical bar).

    NOTE: This function assumes dark letters on a relatively light/textured
    background.  For colored or dark backgrounds use preprocess_color_adaptive()
    instead.

    Algorithm:
      1. Convert to grayscale and apply Gaussian blur to suppress fine texture.
      2. Adaptive thresholding creates a binary mask: dark letter pixels → 0,
         light background pixels → 255.
      3. Morphological opening removes isolated noise dots (tiny shadows that
         aren't part of the actual letter).
      4. Wherever the mask says "background", paint that pixel white in the
         original colour image.

    Result: the YOLO model sees a clean black letter on a white background,
    with no texture artefacts to confuse visually similar shapes.
    """
    if isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        img_np = np.array(image)

    if img_np.ndim == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur first so fine paper/fabric texture doesn't fool the threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold: background pixels → 255, dark letter pixels → 0
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # large block to handle uneven illumination across card
        C=10
    )

    # Remove isolated noise dots (e.g. tiny shadows in crumpled paper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Where binary == 255 (background), paint white in the colour image
    result = img_bgr.copy()
    result[binary == 255] = [255, 255, 255]

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def preprocess_color_adaptive(image):
    """
    Color-agnostic background normalisation that works with ANY card background
    color: white, wood-texture, brown, blue, red, dark, or any other color.

    Unlike preprocess_suppress_background() which assumes DARK letters on a
    LIGHT background, this function automatically detects which pixels are
    letter strokes vs background by comparing lightness across the image using
    Otsu's optimal threshold on the LAB L (lightness) channel.

    Algorithm
    ─────────
    1. Convert to LAB color space – the L channel captures lightness
       independent of hue/saturation, so a blue background and a red
       background with the same perceived brightness are treated identically.
    2. Gaussian blur on L-channel to suppress fine texture grain.
    3. Otsu's global threshold finds the optimal dark/light split without
       any hard-coded threshold value.
    4. Orientation detection: if fewer than 45 % of pixels are white after
       thresholding, the card has a DARK background with LIGHT letters →
       invert the binary mask so background always = white in the output.
    5. Morphological open (noise removal) + close (fill stroke holes).
    6. Compose result: background pixels → white, letter pixels → original
       colour (so the model still benefits from colour cues).

    Returns an RGB numpy array (black/dark letter strokes on white background).
    """
    if isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        img_np = np.array(image)

    if img_np.ndim == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. LAB L-channel (lightness, hue-independent)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # 2. Blur to suppress fine texture / grain
    blurred = cv2.GaussianBlur(l_channel, (7, 7), 0)

    # 3. Otsu's global threshold
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Auto-detect orientation (dark-on-light vs light-on-dark)
    white_fraction = np.sum(binary == 255) / binary.size
    if white_fraction < 0.45:
        # Most pixels are dark → light letters on dark background → invert
        binary = cv2.bitwise_not(binary)

    # 5. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. Compose: background (binary==255) → white; letter pixels → original
    result = img_bgr.copy()
    result[binary == 255] = [255, 255, 255]

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

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

def _run_yolo(processed_image, confidence, augment=False):
    """Internal helper: run MODEL.predict and return raw boxes."""
    results = MODEL.predict(
        source=processed_image,
        conf=confidence,
        save=False,
        iou=0.4,
        verbose=False,
        imgsz=1280,
        augment=augment,
    )
    return results[0].boxes

def _boxes_to_segments(boxes):
    """Convert a list/tensor of boxes to the segments list format."""
    segments = []
    for box in boxes:
        cls_idx = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        class_name = CLASS_NAMES[cls_idx]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bbox = {
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        }

        class_id = CLASS_NAME_TO_ID.get(class_name, 'NA')

        segments.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': conf,
            'is_ambiguous': cls_idx in CONFUSING_CLASSES and conf < 0.7,
            'bbox': bbox
        })
    return segments

def predict_with_model(image_path, confidence=0.35, use_preprocessing=True, use_filtering=True):
    """
    Run YOLO prediction on image with automatic fallback for hard cases.

    Pass 1 – standard preprocessing at the given confidence threshold.
    Pass 2 – if nothing found, retry with CLAHE/bilateral preprocessing
             designed for busy/textured backgrounds (checkered, crumpled
             paper, wood) at confidence 0.20 with YOLO TTA augmentation.
    Pass 3 – if Pass 1+2 both found nothing, retry with color-adaptive
             Otsu normalisation (preprocess_color_adaptive).  This handles
             ANY background color (white, colored, dark, wood-texture) by
             auto-detecting letter vs background without assuming which is
             darker, and normalising to black-on-white.
    Pass 4 – if any earlier pass returned an ambiguous class (G, T, H, F, S …)
             with confidence below _AMBIGUITY_CONF_THRESHOLD (0.65), retry with
             background-suppression (adaptive threshold) to strip texture artefacts.
             Takes the higher-confidence result from Pass 3 vs Pass 4.

    Returns: (segments, error_str)
    """
    if MODEL is None:
        return None, "Model not loaded"

    try:
        image = Image.open(image_path)
        image = resize_image(image, max_dimension=1280)
        img_array = np.array(image)   # keep a clean copy for fallback passes

        fallback_conf = min(confidence, 0.20)

        def _filter(boxes):
            if use_filtering and len(boxes) > 0:
                keep = filter_confusing_detections(boxes, threshold_diff=0.15)
                return [boxes[i] for i in keep]
            return list(boxes)

        # ── Pass 1: standard preprocessing ───────────────────────────────
        if use_preprocessing:
            processed_image = preprocess_image(image, enhance_contrast=True)
        else:
            processed_image = img_array

        boxes1 = _run_yolo(processed_image, confidence, augment=False)
        segments = _boxes_to_segments(_filter(boxes1))

        # ── Pass 2: CLAHE for textured / busy backgrounds ─────────────────
        # augment=False: the preprocessing itself provides the quality uplift;
        # TTA augmentation (augment=True) adds 3-5× latency per pass which
        # can push total inference time past the 45 s API timeout.
        if len(segments) == 0:
            print("⚠️  Pass 1: no detection – retrying with texture-aware "
                  "preprocessing (CLAHE, conf=0.20) …")
            img2 = preprocess_image_for_textured_background(img_array)
            boxes2 = _run_yolo(img2, fallback_conf, augment=False)
            segments = _boxes_to_segments(_filter(boxes2))
            if segments:
                print(f"✅ Pass 2 found {len(segments)} detection(s).")
            else:
                print("❌ Pass 2 also found nothing.")

        # ── Pass 3: color-adaptive Otsu normalization ─────────────────────
        # Triggered when both Pass 1 & 2 found nothing.
        # Handles ANY background color (dark, colored, wood, inverted) by
        # auto-detecting letter/background orientation before normalising.
        if len(segments) == 0:
            print("⚠️  Pass 3: retrying with color-adaptive Otsu normalization …")
            img3 = preprocess_color_adaptive(img_array)
            boxes3 = _run_yolo(img3, fallback_conf, augment=False)
            segments = _boxes_to_segments(_filter(boxes3))
            if segments:
                print(f"✅ Pass 3 (color-adaptive) found {len(segments)} detection(s).")
            else:
                print("❌ Pass 3 also found nothing.")

        # ── Pass 4: background-suppression for ambiguous classes ──────────
        # Triggered when any earlier pass returned a class that is known to
        # be ambiguous on non-white backgrounds (G↔T, F↔H, S↔Three …)
        # AND the confidence is below the ambiguity threshold.
        # Compares Pass-4 result against current best and keeps the winner.
        top_ambiguous = (
            len(segments) > 0
            and segments[0]['class_name'] in _AMBIGUOUS_CLASSES
            and segments[0]['confidence'] < _AMBIGUITY_CONF_THRESHOLD
        )

        if top_ambiguous:
            prev_name = segments[0]['class_name']
            prev_conf = segments[0]['confidence']
            print(f"⚠️  Pass 4: '{prev_name}' detected at low confidence "
                  f"{prev_conf:.2f} < {_AMBIGUITY_CONF_THRESHOLD} – "
                  "retrying with background-suppression to disambiguate …")

            img4 = preprocess_suppress_background(img_array)
            boxes4 = _run_yolo(img4, fallback_conf, augment=False)
            segments4 = _boxes_to_segments(_filter(boxes4))

            if segments4:
                new_conf = segments4[0]['confidence']
                new_name = segments4[0]['class_name']
                if new_conf > prev_conf:
                    print(f"✅ Pass 4 resolved to '{new_name}' "
                          f"(conf={new_conf:.2f} > {prev_conf:.2f}) "
                          f"from '{prev_name}'.")
                    segments = segments4
                else:
                    print(f"ℹ️  Pass 4 result '{new_name}' "
                          f"(conf={new_conf:.2f}) not better than "
                          f"'{prev_name}' ({prev_conf:.2f}); keeping original.")
            else:
                print(f"ℹ️  Pass 4 found nothing; keeping '{prev_name}' result.")

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
    
    # Save file temporarily (cross-platform)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
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

@app.route('/image-new', methods=['POST'])
def predict_image_new():
    """
    Image detection endpoint (from Algo/Algorithm/main.py).
    Saves the original upload and an annotated result image to disk.
    Expected: multipart/form-data with 'file' field
    Optional form field: 'confidence' (float, default 0.25)
    Returns: {
        "success": bool,
        "count": int,
        "detections": [{"class": str, "confidence": float}, ...],
        "result_image": str   # path to annotated image saved on server
    }
    """
    if not YOLO_AVAILABLE or MODEL is None:
        return jsonify({"error": "Model not available"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read file content once
    file_content = file.read()

    # Save original image to uploads folder (relative to server working dir)
    uploads_dir = Path(__file__).parent / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_without_ext = os.path.splitext(file.filename)[0]
    file_ext = os.path.splitext(file.filename)[1]
    upload_path = uploads_dir / f'{timestamp}_{filename_without_ext}{file_ext}'
    with open(upload_path, 'wb') as f:
        f.write(file_content)

    # Get confidence threshold from request, default to 0.25
    confidence = float(request.form.get('confidence', 0.25))

    # Convert file bytes to PIL Image and run prediction
    image = Image.open(io.BytesIO(file_content))
    results = MODEL.predict(
        source=image,
        conf=confidence,
        save=False
    )

    # Get annotated image with bounding boxes drawn
    annotated_img = results[0].plot()

    # Save annotated result image
    results_dir = Path(__file__).parent / 'own_results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f'result_{timestamp}_{filename_without_ext}.jpg'
    cv2.imwrite(str(output_path), annotated_img)

    # Build detection list
    boxes = results[0].boxes
    detections = []
    if len(boxes) > 0:
        for box in boxes:
            cls_idx = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            detections.append({
                "class": CLASS_NAMES[cls_idx],
                "confidence": conf
            })
        result = {
            "success": True,
            "count": len(boxes),
            "detections": detections,
            "result_image": str(output_path)
        }
    else:
        result = {
            "success": False,
            "count": 0,
            "detections": [],
            "message": "No objects detected. Try lowering the confidence threshold.",
            "result_image": str(output_path)
        }

    return jsonify(result), 200


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
    print("  GET  /status     - Health check")
    print("  POST /image      - Image recognition (YOLO + preprocessing + filtering)")
    print("  POST /image-new  - Image recognition (YOLO, saves annotated result to disk)")
    print("  POST /path       - Path planning (A* + TSP via MazeSolver)")
    print("  GET  /stitch     - Image stitching")
    print("="*60)
    print("\nServer starting...\n")
    
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
