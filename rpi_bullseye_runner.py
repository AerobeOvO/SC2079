#!/usr/bin/env python3
"""
RPI Bullseye Runner
-------------------
Captures image from PiCamera, sends to API server for recognition.
When a Bullseye is detected, executes one round of movement via serial to STM32,
then captures the next image. Repeats until quit.

No camera preview window is shown (safe to run over PuTTY / SSH terminal).

Usage:
    python3 rpi_bullseye_runner.py

Press Ctrl+C to quit.
"""

import cv2
import requests
import serial
import time
import os
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import API_IP, API_PORT, SERIAL_PORT, BAUD_RATE
from consts import SYMBOL_MAP

# ─── Camera settings ──────────────────────────────────────────────────────────
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE   = 30

# Class ID for Bullseye in SYMBOL_MAP
BULLSEYE_CLASS_ID = "10"

# ─── Serial helper ────────────────────────────────────────────────────────────
def send(ser: serial.Serial, cmd: str, sleep: float = 3.0):
    """Send a command over serial and wait for reply."""
    msg = (cmd + "\r\n").encode()
    print(f"  [TX] {repr(msg)}")
    ser.write(msg)
    ser.flush()
    time.sleep(sleep)
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        print(f"  [RX] {line}")

# ─── Movement sequence ────────────────────────────────────────────────────────
def navigate_obstacle(ser: serial.Serial, bullseye_count: int):
    """
    Execute one round of movement after detecting a bullseye.
    bullseye_count: how many bullseyes have been seen so far (1-indexed).

    Uncomment the movement commands below and adjust parameters to suit your
    actual course layout.
    """
    print(f"\n[MOVE] Running navigation sequence (bullseye #{bullseye_count}) ...")

    send(ser, "OLED:Bullseye!", sleep=1.0)

    if bullseye_count == 1:
        # ── 1st bullseye movement ─────────────────────────────────────────────
        send(ser, "FWD:10",      sleep=5.0)
        send(ser, "TURNL:90",    sleep=8.0)   # 90° turn needs more time
        send(ser, "REV:90",      sleep=8.0)
        send(ser, "TURNR:50",    sleep=8.0)   # turn to face side
        send(ser, "TURNL:55",    sleep=5.0)

    else:
        # ── subsequent bullseyes movement ─────────────────────────────────────
        send(ser, "TURNR:130",   sleep=8.0)   # 90° turn needs more time
        send(ser, "TURNL:210",   sleep=12.0)  # turn to face side
        send(ser, "FWD:5",       sleep=5.0)

        send(ser, "TURNR:140",   sleep=8.0)
        send(ser, "TURNL:240",   sleep=12.0)
        send(ser, "REV:20",      sleep=5.0)

        send(ser, "TURNR:140",   sleep=8.0)
        send(ser, "TURNL:230",   sleep=12.0)
        send(ser, "FWD:25",      sleep=5.0)

    print("[MOVE] Navigation sequence complete.\n")

# ─── Image / API helpers ──────────────────────────────────────────────────────
def capture_image(camera, raw_capture):
    """Capture a single BGR frame from the PiCamera."""
    from picamera.array import PiRGBArray  # imported here so import error is clear
    raw_capture.truncate(0)
    camera.capture(raw_capture, format="bgr")
    return raw_capture.array.copy()


def send_image_to_api(image, filename="capture.jpg"):
    """Save image to disk, POST it to the API, return (result_dict, error_str)."""
    cv2.imwrite(filename, image)
    url = f"http://{API_IP}:{API_PORT}/image"
    try:
        with open(filename, "rb") as f:
            response = requests.post(url, files={"file": (filename, f)}, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "API Timeout – server took too long to respond"
    except requests.exceptions.ConnectionError:
        return None, f"Connection Error – cannot reach {API_IP}:{API_PORT}"
    except Exception as e:
        return None, f"Error: {e}"


def print_results(result: dict):
    """Pretty-print detection results to the terminal."""
    print(f"  Obstacle ID : {result.get('obstacle_id', 'N/A')}")
    print(f"  Image ID    : {result.get('image_id', 'NA')}")
    segments = result.get("segments", [])
    if segments:
        print(f"  Detections  : {len(segments)} object(s)")
        for i, seg in enumerate(segments):
            cid  = seg.get("class_id", "NA")
            conf = seg.get("confidence", 0)
            name = SYMBOL_MAP.get(str(cid), seg.get("class_name", "Unknown"))
            ambig = " [AMBIGUOUS]" if seg.get("is_ambiguous") else ""
            print(f"    {i+1}. {name} (ID {cid}) – {conf*100:.1f}%{ambig}")
    else:
        print("  Detections  : none")


def is_bullseye_detected(result: dict) -> bool:
    """Return True if any detected segment is a Bullseye."""
    if result is None:
        return False
    for seg in result.get("segments", []):
        if str(seg.get("class_id", "")) == BULLSEYE_CLASS_ID:
            return True
    return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Late import so a missing picamera gives a clear error message
    try:
        from picamera import PiCamera
        from picamera.array import PiRGBArray
    except ImportError:
        print("ERROR: 'picamera' package not found. Are you running on a Raspberry Pi?")
        sys.exit(1)

    print("=" * 60)
    print("  RPI Bullseye Runner")
    print("=" * 60)
    print(f"  API Server : http://{API_IP}:{API_PORT}")
    print(f"  Serial     : {SERIAL_PORT}  @  {BAUD_RATE} baud")
    print("  (No camera preview – safe for SSH/PuTTY sessions)")
    print("=" * 60)

    # ── Check API ─────────────────────────────────────────────────────────────
    print("\n[INIT] Testing API connection ...")
    try:
        r = requests.get(f"http://{API_IP}:{API_PORT}/status", timeout=5)
        if r.status_code == 200:
            st = r.json()
            print(f"  ✅ API server is running")
            print(f"     Model     : {st.get('model', 'unknown')}")
            print(f"     YOLO      : {st.get('yolo_available', False)}")
            print(f"     Algorithm : {st.get('algorithm_available', False)}")
        else:
            print(f"  ⚠️  API returned status {r.status_code}")
    except Exception as e:
        print(f"  ❌ Cannot connect to API: {e}")
        print(f"     Make sure the server is running on {API_IP}:{API_PORT}")
        sys.exit(1)

    # ── Open serial port ──────────────────────────────────────────────────────
    print(f"\n[INIT] Opening serial port {SERIAL_PORT} ...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        time.sleep(2)
        ser.reset_input_buffer()
        print("  ✅ Serial port open")
    except serial.SerialException as e:
        print(f"  ❌ Could not open serial port: {e}")
        sys.exit(1)

    # ── Init camera ───────────────────────────────────────────────────────────
    print("\n[INIT] Initialising camera ...")
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate  = CAMERA_FRAMERATE
    raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)
    time.sleep(2)   # warm-up
    print("  ✅ Camera ready")

    # ── Main loop ─────────────────────────────────────────────────────────────
    bullseye_count  = 0
    capture_number  = 0
    just_moved      = False   # True right after a movement round completes

    print("\n[INFO] Starting capture loop. Press Ctrl+C to quit.\n")
    print("=" * 60)

    try:
        while True:
            capture_number += 1
            timestamp = int(time.time())
            filename  = f"capture_{timestamp}_{capture_number}.jpg"

            # Capture
            print(f"[CAP #{capture_number}] Capturing image → {filename}")
            image = capture_image(camera, raw_capture)
            cv2.imwrite(filename, image)   # save raw capture

            # Send to API
            print(f"[CAP #{capture_number}] Sending to API ...")
            t0 = time.time()
            result, error = send_image_to_api(image, filename)
            elapsed = time.time() - t0

            if error:
                print(f"  ❌ {error}")
                print("  Retrying in 3 s ...\n")
                time.sleep(3)
                continue

            print(f"  ✅ Response in {elapsed:.2f} s")
            print_results(result)

            # Save annotated image (uses OpenCV only, no display window)
            annotated_filename = f"annotated_{timestamp}_{capture_number}.jpg"
            _save_annotated(image, result, annotated_filename)
            print(f"  💾 Annotated image saved: {annotated_filename}")

            # ── Decision logic ────────────────────────────────────────────────
            if is_bullseye_detected(result):
                # Another bullseye → do another movement round
                bullseye_count += 1
                print(f"\n🎯 BULLSEYE detected! (total seen: {bullseye_count})")
                navigate_obstacle(ser, bullseye_count)
                just_moved = True
                print("[INFO] Movement done. Taking next picture ...\n")
                print("=" * 60)

            elif just_moved:
                # We moved after a bullseye and now see a NON-bullseye symbol
                # → this is the final target image; stop here.
                segments = result.get("segments", [])
                if segments:
                    cid  = str(segments[0].get("class_id", "NA"))
                    name = SYMBOL_MAP.get(cid, segments[0].get("class_name", "Unknown"))
                    conf = segments[0].get("confidence", 0)
                    print(f"\n🏁 TARGET IMAGE REACHED: {name} ({conf*100:.1f}%)")
                else:
                    print("\n🏁 TARGET REACHED but no symbol was recognised.")
                print("[INFO] Task complete – stopping.\n")
                print("=" * 60)
                break   # ← exit the loop

            else:
                # Haven't moved yet and no bullseye – keep scanning
                print("\n[INFO] No bullseye yet – waiting 2 s then capturing again ...\n")
                print("=" * 60)
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n[INFO] Ctrl+C received – shutting down.")

    finally:
        camera.close()
        ser.close()
        print("[INFO] Camera and serial port closed. Bye!")


# ─── Annotated image helper (no GUI / imshow) ─────────────────────────────────
def _save_annotated(image, results, out_path: str):
    """Draw bounding boxes on image and save to file (no display window)."""
    annotated = image.copy()
    height, width = annotated.shape[:2]

    if results is None:
        cv2.imwrite(out_path, annotated)
        return

    segments = results.get("segments", [])
    if not segments:
        cv2.putText(annotated, "No Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(out_path, annotated)
        return

    for i, seg in enumerate(segments):
        cid        = str(seg.get("class_id", "NA"))
        conf       = seg.get("confidence", 0)
        is_ambig   = seg.get("is_ambiguous", False)
        bbox       = seg.get("bbox", None)
        name       = SYMBOL_MAP.get(cid, seg.get("class_name", "Unknown"))
        color      = (0, 165, 255) if is_ambig else (0, 255, 0)

        if bbox and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            label = f"{name} {conf*100:.0f}%"
            if is_ambig:
                label += " [!]"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ly = max(y1 - 10, lh + 10)
            cv2.rectangle(annotated, (x1, ly - lh - 10), (x1 + lw + 10, ly), color, -1)
            cv2.putText(annotated, label, (x1 + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated, f"#{i+1}", (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            y_off = 30 + i * 35
            label = f"{i+1}. {name} ({conf*100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (5, y_off - th - 5), (15 + tw, y_off + 5), color, -1)
            cv2.putText(annotated, label, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    summary = f"Detected: {len(segments)} object(s)"
    cv2.putText(annotated, summary, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(out_path, annotated)


if __name__ == "__main__":
    main()
