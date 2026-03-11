#!/usr/bin/env python3
"""
API Server for Robot Navigation System
- Image recognition using YOLO model  (POST /image, POST /image-new)
- Path planning via MazeSolver A* + TSP  (POST /path)
- Image stitching / tiled display  (GET /stitch, GET /stitch-reset)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import time

# ── Path setup ────────────────────────────────────────────────────────────────
model_dir = Path(__file__).parent.parent / 'ModelTraining'
algo_dir  = Path(__file__).parent.parent / 'Algo' / 'Algorithm'
sys.path.insert(0, str(model_dir))
sys.path.insert(0, str(algo_dir))

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: ultralytics not installed. Image recognition will not work.")
    YOLO_AVAILABLE = False

import torch

# Import bbox selection directly from model.py
try:
    from model import find_largest_or_central_bbox
except ImportError as e:
    print(f"WARNING: Could not import from model.py: {e}")
    def find_largest_or_central_bbox(bboxes, signal):
        return "NA", 0.0

# ── Algorithm ─────────────────────────────────────────────────────────────────
try:
    from algo.algo import MazeSolver
    from helper import command_generator
    ALGO_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Algorithm modules not available: {e}")
    ALGO_AVAILABLE = False

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL      = None
MODEL_PATH = model_dir / 'best_JH.pt'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Folder where annotated result images are saved (one per obstacle SNAP)
RESULTS_DIR = Path(__file__).parent / 'snap_results'

# ── Class mappings for best_JH.pt ─────────────────────────────────────────────
# The model uses numeric string IDs ("10"–"40") as its class names.
# Index layout:
#   [0]"10"=Bullseye  [1-9]"11"-"19"=1-9
#   [10-17]"20"-"27"=A-H   [18-25]"28"-"35"=S-Z
#   [26]"36"=Up  [27]"37"=Down  [28]"38"=Right  [29]"39"=Left
#   [30]"40"=circle/target
CLASS_NAMES = [
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29', '30', '31', '32', '33', '34', '35',
    '36', '37', '38', '39', '40',
]

DISPLAY_NAMES = {
    '10': 'Bullseye',
    '11': '1',  '12': '2',  '13': '3',  '14': '4',  '15': '5',
    '16': '6',  '17': '7',  '18': '8',  '19': '9',
    '20': 'A',  '21': 'B',  '22': 'C',  '23': 'D',  '24': 'E',
    '25': 'F',  '26': 'G',  '27': 'H',  '28': 'S',  '29': 'T',
    '30': 'U',  '31': 'V',  '32': 'W',  '33': 'X',  '34': 'Y',  '35': 'Z',
    '36': 'Up Arrow',    '37': 'Down Arrow',
    '38': 'Right Arrow', '39': 'Left Arrow',
    '40': 'Stop (Circle)',
}

# Identity map — model class names ARE the string IDs
CLASS_NAME_TO_ID = {str(i): str(i) for i in range(10, 41)}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load YOLO model at startup."""
    global MODEL
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed")
        return False
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False
    try:
        print(f"Loading YOLO model from {MODEL_PATH} …")
        MODEL = YOLO(str(MODEL_PATH))
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


def resize_image(image: Image.Image, max_dimension: int = 1280) -> Image.Image:
    """Downscale large images while keeping aspect ratio."""
    w, h = image.size
    if w <= max_dimension and h <= max_dimension:
        return image
    if w > h:
        return image.resize((max_dimension, int(h * max_dimension / w)), Image.Resampling.LANCZOS)
    return image.resize((int(w * max_dimension / h), max_dimension), Image.Resampling.LANCZOS)


def predict_with_model(image_path, confidence: float = 0.35, signal: str = "C"):
    """
    Single-pass YOLO prediction mirroring model.py's predict_image() logic.
    Uses find_largest_or_central_bbox() imported from model.py.

    Returns: (segments, annotated_bgr, error_str)
      segments      – list of detection dicts
      annotated_bgr – BGR numpy array with bounding boxes drawn (for saving)
      error_str     – non-None string on failure
    """
    if MODEL is None:
        return None, None, "Model not loaded"

    try:
        results = MODEL.predict(
            source=str(image_path),
            conf=confidence,
            imgsz=640,
            device=DEVICE,
            verbose=False,
        )

        # Annotated image (BGR numpy) — used for saving / stitching
        annotated_bgr = results[0].plot()

        # Build bbox list in the format model.py expects
        bboxes = []
        if results[0].boxes:
            for result in results:
                for box in result.boxes:
                    cls_idx   = int(box.cls.tolist()[0])
                    label     = result.names[cls_idx]       # string ID e.g. "27"
                    xywh      = box.xywh.tolist()[0]
                    conf_val  = box.conf.tolist()[0]
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bboxes.append({
                        "label":      label,
                        "xywh":       xywh,
                        "bbox_area":  xywh[2] * xywh[3],
                        "confidence": conf_val,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

        # Selection logic from model.py
        selected_label, _ = find_largest_or_central_bbox(bboxes, signal)

        # Build API-ready segment list
        segments = [
            {
                "class_id":    b["label"],
                "class_name":  b["label"],
                "display_name": DISPLAY_NAMES.get(b["label"], b["label"]),
                "confidence":  b["confidence"],
                "bbox": {"x1": b["x1"], "y1": b["y1"],
                         "x2": b["x2"], "y2": b["y2"]},
            }
            for b in bboxes
        ]

        # Put the selected detection first
        if selected_label != "NA":
            segments.sort(
                key=lambda s: s["confidence"] if s["class_id"] == selected_label else -1,
                reverse=True,
            )

        return segments, annotated_bgr, None

    except Exception as e:
        return None, None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# STITCH / TILED DISPLAY  (Point 7)
# ══════════════════════════════════════════════════════════════════════════════

def _save_snap_result(obstacle_id: str, detected_id: str, annotated_bgr: np.ndarray):
    """
    Save the annotated image for one obstacle into RESULTS_DIR.
    Filename: <obstacle_id>_<display_name>_id<detected_id>.jpg
    e.g.  "2_H_id27.jpg"
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    display = DISPLAY_NAMES.get(detected_id, detected_id)
    fname   = f"{int(obstacle_id):02d}_{display}_id{detected_id}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), annotated_bgr)
    print(f"💾  Saved snap result: {fname}")


def _build_tiled_image(results_dir: Path) -> Path | None:
    """
    Load all annotated images from results_dir, arrange them in a 3-column
    grid (like the example in Point 7), and save as 'tiled_result.jpg'.
    Returns the path to the saved file, or None if no images found.
    """
    image_files = sorted(results_dir.glob("*.jpg"),
                         key=lambda p: p.name if 'tiled' not in p.name else 'zzz')
    image_files = [p for p in image_files if 'tiled' not in p.name]

    if not image_files:
        return None

    THUMB_W, THUMB_H = 480, 360
    LABEL_H          = 44
    PADDING          = 12
    COLS             = min(3, len(image_files))
    ROWS             = math.ceil(len(image_files) / COLS)

    grid_w = COLS * THUMB_W + (COLS + 1) * PADDING
    grid_h = ROWS * (THUMB_H + LABEL_H) + (ROWS + 1) * PADDING

    grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)

    # Try to load a font; fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for idx, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Resize thumbnail
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize(
            (THUMB_W, THUMB_H), Image.Resampling.LANCZOS
        )

        row, col = divmod(idx, COLS)
        x = PADDING + col * (THUMB_W + PADDING)
        y = PADDING + row * (THUMB_H + LABEL_H + PADDING)

        grid.paste(pil_img, (x, y))

        # Label below thumbnail
        label = img_path.stem          # e.g. "02_H_id27"
        draw.text(
            (x + THUMB_W // 2, y + THUMB_H + 6),
            label, fill=(255, 255, 255), font=font, anchor='mt',
        )

    # Fill unused slots with a dark red placeholder
    for idx in range(len(image_files), ROWS * COLS):
        row, col = divmod(idx, COLS)
        x = PADDING + col * (THUMB_W + PADDING)
        y = PADDING + row * (THUMB_H + LABEL_H + PADDING)
        draw.rectangle([x, y, x + THUMB_W, y + THUMB_H], fill=(80, 20, 20))
        draw.text(
            (x + THUMB_W // 2, y + THUMB_H // 2),
            "—", fill=(160, 60, 60), font=font, anchor='mm',
        )

    out_path = results_dir / 'tiled_result.jpg'
    grid.save(str(out_path), quality=92)
    print(f"🖼️   Tiled image saved: {out_path}")
    return out_path


def _open_file(path: Path):
    """Open an image file with the system's default viewer."""
    try:
        sys_name = platform.system()
        if sys_name == 'Darwin':
            subprocess.Popen(['open', str(path)])
        elif sys_name == 'Windows':
            subprocess.Popen(['start', str(path)], shell=True)
        else:
            subprocess.Popen(['xdg-open', str(path)])
    except Exception as e:
        print(f"WARNING: Could not open file viewer: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/status', methods=['GET'])
def status():
    """Health check endpoint."""
    return jsonify({
        "status":               "ok",
        "model":                "loaded" if MODEL is not None else "not loaded",
        "model_path":           str(MODEL_PATH),
        "device":               DEVICE,
        "yolo_available":       YOLO_AVAILABLE,
        "algorithm_available":  ALGO_AVAILABLE,
    }), 200


@app.route('/image', methods=['POST'])
def image_recognition():
    """
    Primary image-recognition endpoint called by main_runner.py after each SNAP.
    Expected: multipart/form-data  'file' field
    Filename convention: <timestamp>_<obstacleId>_<signal>.jpg
    Returns: {"obstacle_id", "image_id", "segments": [...]}
    """
    if not YOLO_AVAILABLE or MODEL is None:
        return jsonify({"error": "Model not available", "obstacle_id": "NA", "segments": []}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']

    # Parse obstacle_id and signal from filename
    obstacle_id, signal = "1", "C"
    filename = file.filename or "image.jpg"
    parts = filename.split('_')
    if len(parts) >= 2:
        try:    obstacle_id = parts[1]
        except: pass
    if len(parts) >= 3:
        try:    signal = parts[2].split('.')[0].upper()
        except: pass

    # Save upload to temp file
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)

    try:
        segments, annotated_bgr, error = predict_with_model(
            temp_path, confidence=0.35, signal=signal
        )

        if error:
            return jsonify({"error": error, "obstacle_id": obstacle_id, "segments": []}), 500

        if segments:
            primary = segments[0]
            detected_id = primary['class_id']

            # ── Save annotated image for later stitch display (Point 7) ──
            if annotated_bgr is not None:
                _save_snap_result(obstacle_id, detected_id, annotated_bgr)

            response = {
                "obstacle_id": obstacle_id,
                "image_id":    detected_id,
                "segments":    segments,
            }
        else:
            # No detection — still save the unannotated image as placeholder
            if annotated_bgr is not None:
                _save_snap_result(obstacle_id, "NA", annotated_bgr)
            response = {
                "obstacle_id": obstacle_id,
                "image_id":    "NA",
                "segments":    [],
            }

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return jsonify(response), 200

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print("=" * 60)
        print("ERROR IN /image:")
        print(trace)
        print("=" * 60)
        return jsonify({"error": str(e), "traceback": trace,
                        "obstacle_id": obstacle_id, "segments": []}), 500


@app.route('/path', methods=['POST'])
def path_planning():
    """
    Path planning endpoint using MazeSolver A* + TSP.
    Expected JSON: {"obstacles":[{"x","y","id","d"}], "robot_x","robot_y","robot_dir","retrying"}
    """
    if not ALGO_AVAILABLE:
        return jsonify({"error": "Algorithm not available",
                        "data": {"commands": [], "path": [], "distance": 0}}), 500
    try:
        content        = request.json
        obstacles      = content['obstacles']
        retrying       = content.get('retrying', False)
        robot_x        = content.get('robot_x', 1)
        robot_y        = content.get('robot_y', 1)
        robot_dir      = int(content.get('robot_dir', 0))

        print(f"Path planning: {len(obstacles)} obstacles, robot=({robot_x},{robot_y}) dir={robot_dir}")

        maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_dir, big_turn=None)
        for ob in obstacles:
            maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['id'])

        t0 = time.time()
        optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
        print(f"✅ Path in {time.time()-t0:.2f}s — distance={distance}")

        commands     = command_generator(optimal_path, obstacles)
        path_results = [optimal_path[0].get_dict()]
        i = 0
        for cmd in commands:
            if cmd.startswith(("SNAP", "FIN")):
                continue
            elif cmd.startswith(("FW", "FS", "BW", "BS")):
                i += int(cmd[2:]) // 10
            else:
                i += 1
            if i < len(optimal_path):
                path_results.append(optimal_path[i].get_dict())

        return jsonify({"data": {"distance": distance,
                                 "path": path_results,
                                 "commands": commands},
                        "error": None}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e),
                        "data": {"commands": [], "path": [], "distance": 0}}), 500


@app.route('/image-new', methods=['POST'])
def predict_image_new():
    """
    Debug endpoint — saves raw upload + annotated result to disk.
    """
    if not YOLO_AVAILABLE or MODEL is None:
        return jsonify({"error": "Model not available"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    file_content = file.read()
    ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem, ext    = os.path.splitext(file.filename)

    uploads_dir = Path(__file__).parent / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    with open(uploads_dir / f'{ts}_{stem}{ext}', 'wb') as f:
        f.write(file_content)

    confidence = float(request.form.get('confidence', 0.25))
    image      = Image.open(io.BytesIO(file_content))
    results    = MODEL.predict(source=image, conf=confidence, save=False)

    annotated_img = results[0].plot()
    own_dir       = Path(__file__).parent / 'own_results'
    own_dir.mkdir(exist_ok=True)
    output_path   = own_dir / f'result_{ts}_{stem}.jpg'
    cv2.imwrite(str(output_path), annotated_img)

    boxes      = results[0].boxes
    detections = []
    for box in boxes:
        cls_idx = int(box.cls.cpu().numpy()[0])
        conf    = float(box.conf.cpu().numpy()[0])
        label   = CLASS_NAMES[cls_idx]
        detections.append({"class_id": label,
                            "class":    DISPLAY_NAMES.get(label, label),
                            "confidence": conf})

    result = {
        "success":      len(detections) > 0,
        "count":        len(detections),
        "detections":   detections,
        "result_image": str(output_path),
    }
    if not detections:
        result["message"] = "No objects detected. Try lowering the confidence threshold."
    return jsonify(result), 200


@app.route('/stitch', methods=['GET'])
def stitch():
    """
    Point 7: Build a tiled grid of all annotated SNAP images and open it on
    the PC screen.  Called by main_runner.py at the end of every run.
    """
    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob('*.jpg')):
        return jsonify({"status": "no snap images found"}), 200

    def _do_stitch():
        out = _build_tiled_image(RESULTS_DIR)
        if out:
            _open_file(out)

    threading.Thread(target=_do_stitch, daemon=True).start()
    count = len([p for p in RESULTS_DIR.glob('*.jpg') if 'tiled' not in p.name])
    return jsonify({"status": "building tiled display", "snap_count": count}), 200


@app.route('/stitch-reset', methods=['GET'])
def stitch_reset():
    """
    Clear all saved snap images from the previous run so the next run starts
    with a clean slate.  Call this at the start of each run.
    """
    removed = 0
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob('*.jpg'):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    print(f"🗑️   Snap results cleared ({removed} file(s) removed).")
    return jsonify({"status": "cleared", "removed": removed}), 200


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Robot Navigation API Server")
    print("=" * 60)

    if YOLO_AVAILABLE:
        if not load_model():
            print("WARNING: Server starting without model — image recognition disabled.")
    else:
        print("WARNING: ultralytics not installed — run: pip install ultralytics")

    print(f"\n  Model   : {MODEL_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  YOLO    : {YOLO_AVAILABLE}  |  Algo: {ALGO_AVAILABLE}")
    print(f"  Results : {RESULTS_DIR}")
    print("\nEndpoints:")
    print("  GET  /status        Health check")
    print("  POST /image         Image recognition (main_runner SNAP)")
    print("  POST /image-new     Debug recognition (saves annotated image)")
    print("  POST /path          Path planning (A* + TSP)")
    print("  GET  /stitch        Build & display tiled result grid (end of run)")
    print("  GET  /stitch-reset  Clear snap images (start of run)")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
