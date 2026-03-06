#!/usr/bin/env python3
"""
main_runner.py  –  SC2079 Full Navigation + Image Recognition Runner
=====================================================================

FLOW
────
  1. Connect to STM32 via serial (UART).
  2. Wait for Android tablet to connect via Bluetooth.
  3. Receive obstacle JSON from Android:
       {"cat": "obstacles", "value": [{"x":4,"y":13,"id":2,"d":2}, ...]}
  4. POST obstacles to API server (/path) → receive optimal command list
     (A* path-finding + TSP ordering via MazeSolver).
  5. Execute each command in order:
       • Algo motion commands are CONVERTED to STM32 text format before sending:
           FW{n}  → FWD:{n}     (forward n cm)
           BW{n}  → REV:{n}     (reverse n cm)
           FR00   → TURNR:90    (forward-right arc  = turn right 90°)
           FL00   → TURNL:90    (forward-left arc   = turn left 90°)
           BR00   → REVR:90     (backward-right arc, from movement.py)
           BL00   → REVL:90     (backward-left  arc, from movement.py)
           TL{n}  → TURNL:90    (point turn left)
           TR{n}  → TURNR:90    (point turn right)
       • SNAP{id}[_signal]             → capture image, POST to /image,
                                         store recognised class_id
       • FIN                           → stop, report results to Android
  6. Send final image-recognition summary back to Android.

DIRECTION CONVENTIONS
─────────────────────
  Android JSON 'd':  0=North  1=East  2=South  3=West   (0-based, step 1)
  Algo Direction:    0=North  2=East  4=South  6=West   (enum step 2)
  The script maps Android→Algo before posting to /path, and maps
  Algo direction values back to human-readable names in all log output.

ANDROID JSON FORMAT
───────────────────
  Inbound (Android → RPi):
    {"cat": "obstacles",
     "value": [{"x": int, "y": int, "id": int, "d": int}, ...]}

  Direction 'd' mapping:
    0 = North  |  1 = East  |  2 = South  |  3 = West

  Outbound (RPi → Android):
    {"cat": "info",    "value": "<message>"}
    {"cat": "path",    "value": {"commands": [...], "distance": float, "path": [...]}}
    {"cat": "snap",    "value": "<status message>"}
    {"cat": "result",  "value": {"obstacle_id": int, "image_id": str, "signal": str}}
    {"cat": "results", "value": [{"obstacle_id": int, "image_id": str}, ...]}
    {"cat": "error",   "value": "<error message>"}

CONFIGURATION
─────────────
  Edit settings.py for API_IP, API_PORT, SERIAL_PORT, BAUD_RATE.
  IMAGE_REC_DISTANCE (25 cm) is the target robot-to-obstacle viewing distance;
  the MazeSolver positions the robot at the closest feasible grid cell (~30 cm).

USAGE (on Raspberry Pi)
───────────────────────
  cd /path/to/SC2079
  python3 main_runner.py

Press Ctrl+C to quit cleanly.
"""

import bluetooth
import json
import os
import requests
import serial
import sys
import time
from typing import Dict, List, Optional

# ─── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import API_IP, API_PORT, SERIAL_PORT, BAUD_RATE
from logger import prepare_logger

# ─── Logger ───────────────────────────────────────────────────────────────────
log = prepare_logger()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Bluetooth service identifiers (must match Android app's UUID)
BT_SERVICE_NAME = "MDP-RPi-Runner"
BT_UUID         = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

# Target robot-to-obstacle distance for image capture (centimetres).
# The MazeSolver places the robot at the nearest valid grid cell (~30 cm).
IMAGE_REC_DISTANCE = 25

# Robot starting state on the 20×20 arena grid
ROBOT_START_X   = 1    # grid units (bottom-left corner)
ROBOT_START_Y   = 1
ROBOT_START_DIR = 0    # Direction.NORTH in algo convention (0=N, 2=E, 4=S, 6=W)

# How long to wait for a STM32 ACK before moving on (seconds)
STM_ACK_TIMEOUT = 15.0

# Serial read timeout for individual readline() calls (seconds).
# Keep this SHORT so the polling loop can re-check the deadline accurately.
# The outer deadline (STM_ACK_TIMEOUT) controls the total wait time.
STM_SERIAL_READ_TIMEOUT = 0.1

# If this many consecutive motion commands time out without an ACK the runner
# will stop sending further commands and raise an error.
# Set to 0 to disable the guard (not recommended).
STM_MAX_CONSECUTIVE_TIMEOUTS = 5

# Max capture + recognition attempts per obstacle before giving up
MAX_SNAP_ATTEMPTS = 6

# When True the script loops indefinitely: after each run finishes (or fails)
# it resets and waits for the next Android Bluetooth connection automatically.
# Set to False to exit after a single run.
AUTO_RESTART = False

# Seconds to pause between runs before accepting the next connection
RESTART_DELAY = 3.0

# ─── Direction mappings ───────────────────────────────────────────────────────
# Android JSON 'd': 0=North  1=East  2=South  3=West   (step 1)
# Algo Direction:   0=North  2=East  4=South  6=West   (step 2, matches Direction enum)
ANDROID_TO_ALGO_DIR: Dict[int, int] = {0: 0, 1: 2, 2: 4, 3: 6}

# Human-readable names for ANDROID direction values (used when logging received obstacles)
ANDROID_DIR_NAMES: Dict[int, str] = {0: "North", 1: "East", 2: "South", 3: "West"}

# Human-readable names for ALGO direction values (used when logging path states)
ALGO_DIR_NAMES: Dict[int, str] = {0: "North", 2: "East", 4: "South", 6: "West", 8: "Skip"}

# ─── Algo → STM32 command conversion ─────────────────────────────────────────
#
# The path planner (helper.py) emits compact "algo-format" commands.
# The STM32 firmware understands the longer text-format commands below.
# convert_algo_to_stm32() maps one format to the other.
#
# Degrees used for every 90° arc/point turn sent to STM32.
# Adjust to match your robot's physical calibration.
ARC_TURN_ANGLE: int = 90

# Algo-format prefixes (generated by helper.py / path planner)
ALGO_CMD_PREFIXES = (
    "FW", "BW",       # straight forward / backward  (e.g. FW10, BW30)
    "FS", "BS",       # forward / backward step       (e.g. FS10)
    "FR", "FL",       # forward arc right / left      (e.g. FR00)
    "BR", "BL",       # backward arc right / left     (e.g. BR00)
    "TL", "TR",       # point turn left / right       (e.g. TL00)
)

# STM32 text-format prefixes (already in the correct format, pass through)
STM32_TEXT_PREFIXES = (
    "FWD:", "REV:",             # straight motion
    "TURNL:", "TURNR:",         # point / forward-arc turns
    "REVL:", "REVR:",           # backward-arc turns (e.g. REVL:90, REVR:90)
    "OLED:", "STOP", "RS",      # misc
)

# STM32 acknowledgement prefixes – the firmware may reply with either "OK" or "ACK".
# Both are treated as successful completion of the previous command.
STM32_ACK_PREFIXES = ("OK", "ACK")

# ─── Human-readable symbol names (for logging) ────────────────────────────────
SYMBOL_MAP: Dict[str, str] = {
    "10": "Bullseye",
    "11": "One",      "12": "Two",   "13": "Three", "14": "Four",
    "15": "Five",     "16": "Six",   "17": "Seven", "18": "Eight", "19": "Nine",
    "20": "A",  "21": "B",  "22": "C",  "23": "D",  "24": "E",
    "25": "F",  "26": "G",  "27": "H",  "28": "S",  "29": "T",
    "30": "U",  "31": "V",  "32": "W",  "33": "X",  "34": "Y",  "35": "Z",
    "36": "Up Arrow",   "37": "Down Arrow",
    "38": "Right Arrow","39": "Left Arrow",
    "40": "Stop (Circle)",
}


# ══════════════════════════════════════════════════════════════════════════════
# ANDROID BLUETOOTH LINK
# ══════════════════════════════════════════════════════════════════════════════

class AndroidBT:
    """
    Bluetooth RFCOMM server.
    Advertises a service and waits for the Android tablet to connect.
    All messages are newline-delimited UTF-8 text (typically JSON).
    """

    def __init__(self) -> None:
        self._server: Optional[bluetooth.BluetoothSocket] = None
        self._client: Optional[bluetooth.BluetoothSocket] = None
        self._buf: str = ""

    # ── Connect ───────────────────────────────────────────────────────────────
    def connect(self) -> None:
        log.info("Making RPi Bluetooth-discoverable …")
        os.system("sudo hciconfig hci0 piscan")

        self._server = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self._server.bind(("", bluetooth.PORT_ANY))
        self._server.listen(1)
        port = self._server.getsockname()[1]

        bluetooth.advertise_service(
            self._server,
            BT_SERVICE_NAME,
            service_id=BT_UUID,
            service_classes=[BT_UUID, bluetooth.SERIAL_PORT_CLASS],
            profiles=[bluetooth.SERIAL_PORT_PROFILE],
        )

        log.info(f"📡  Waiting for Android on RFCOMM channel {port} …")
        log.info(f"     Service: {BT_SERVICE_NAME}  |  UUID: {BT_UUID}")
        self._client, info = self._server.accept()
        log.info(f"✅  Android connected from {info}")

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, text: str) -> None:
        """Send a newline-terminated UTF-8 text line to Android."""
        msg = text.rstrip("\n") + "\n"
        try:
            self._client.send(msg.encode("utf-8"))
            log.debug(f"→ Android: {msg.strip()!r}")
        except OSError as exc:
            log.error(f"BT send error: {exc}")

    # ── Receive ───────────────────────────────────────────────────────────────
    def recv(self) -> Optional[str]:
        """
        Return the next non-empty line from Android.
        Buffers partial reads internally.
        Raises OSError if the connection is closed.
        """
        while True:
            if "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    log.debug(f"← Android: {line!r}")
                    return line
                continue
            chunk = self._client.recv(4096)
            if not chunk:
                raise OSError("Android socket closed (empty recv)")
            self._buf += chunk.decode("utf-8", errors="ignore")

    # ── Disconnect ────────────────────────────────────────────────────────────
    def close(self) -> None:
        for sock in (self._client, self._server):
            try:
                if sock:
                    sock.close()
            except Exception as exc:
                log.warning(f"BT close warning: {exc}")
        log.info("Bluetooth closed.")


# ══════════════════════════════════════════════════════════════════════════════
# STM32 SERIAL LINK
# ══════════════════════════════════════════════════════════════════════════════

class STMSerial:
    """
    Thin wrapper around pyserial for talking to the STM32 over UART.
    Commands are CR+LF-terminated; STM32 replies with lines starting with 'ACK'.
    """

    def __init__(self) -> None:
        self._ser: Optional[serial.Serial] = None

    # ── Connect ───────────────────────────────────────────────────────────────
    def connect(self) -> None:
        log.info(f"Opening serial port {SERIAL_PORT} @ {BAUD_RATE} baud …")
        # Use a SHORT per-read timeout so readline() never blocks for longer
        # than STM_SERIAL_READ_TIMEOUT; the outer deadline loop in
        # _send_stm_and_wait_ack() then controls the true total wait.
        self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=STM_SERIAL_READ_TIMEOUT)
        time.sleep(2)                       # allow STM32 to finish reset
        self._ser.reset_input_buffer()
        log.info("✅  STM32 serial connected.")

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, cmd: str) -> None:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("STM32 serial port is not open.")
        payload = (cmd + "\r\n").encode("utf-8")
        log.debug(f"→ STM32: {cmd!r}")
        self._ser.write(payload)
        self._ser.flush()

    # ── Receive ───────────────────────────────────────────────────────────────
    def recv(self) -> Optional[str]:
        """
        Read one line from STM32 (blocks up to STM_ACK_TIMEOUT seconds).
        Returns the decoded string, or None on timeout / empty read.
        """
        if self._ser is None or not self._ser.is_open:
            return None
        raw = self._ser.readline()
        if raw:
            decoded = raw.strip().decode("utf-8", errors="ignore")
            log.debug(f"← STM32: {decoded!r}")
            return decoded
        return None

    # ── Disconnect ────────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            log.info("STM32 serial closed.")


# ══════════════════════════════════════════════════════════════════════════════
# PATH PLANNING  (via API server /path endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def plan_path(obstacles_android: List[dict]) -> Optional[dict]:
    """
    Convert Android-format obstacles to algo-format, POST to /path, return result.

    Android obstacle: {"x": int, "y": int, "id": int, "d": int (0–3)}
    Algo obstacle:    {"x": int, "y": int, "id": int, "d": int (0,2,4,6)}

    Returns the full API response dict on success, or None on failure.
    """
    # Remap direction values: Android 0-3  →  Algo 0/2/4/6
    algo_obstacles = [
        {
            "x":  int(ob["x"]),
            "y":  int(ob["y"]),
            "id": int(ob["id"]),
            "d":  ANDROID_TO_ALGO_DIR.get(int(ob["d"]), 0),
        }
        for ob in obstacles_android
    ]

    payload = {
        "obstacles": algo_obstacles,
        "robot_x":   ROBOT_START_X,
        "robot_y":   ROBOT_START_Y,
        "robot_dir": ROBOT_START_DIR,
        "big_turn":  "0",
        "retrying":  False,
    }

    url = f"http://{API_IP}:{API_PORT}/path"
    log.info(f"Requesting path from {url}  ({len(algo_obstacles)} obstacle(s)) …")
    log.debug(f"Path payload: {json.dumps(payload, indent=2)}")

    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            cmds = data.get("data", {}).get("commands", [])
            dist = data.get("data", {}).get("distance", 0)
            log.info(f"✅  Path received: {len(cmds)} command(s), distance={dist:.1f} units")
            return data
        else:
            log.error(f"Path API HTTP {resp.status_code}: {resp.text[:300]}")
            return None

    except requests.exceptions.Timeout:
        log.error("Path API timed out (>60 s).")
        return None
    except requests.exceptions.ConnectionError as exc:
        log.error(f"Path API connection error: {exc}")
        return None
    except Exception as exc:
        log.error(f"Path API unexpected error: {exc}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE CAPTURE  (PiCamera)
# ══════════════════════════════════════════════════════════════════════════════

def capture_image(filename: str, attempt: int = 1) -> None:
    """
    Capture a JPEG image using the legacy picamera module.

    'attempt' drives exposure bracketing:
      attempts 1-3: progressively brighter (longer shutter, positive EV)
      attempts 4+:  progressively darker  (shorter shutter, negative EV)

    Raises RuntimeError if picamera is not installed.
    """
    try:
        import picamera                       # type: ignore  (RPi only)
    except ImportError:
        raise RuntimeError(
            "The 'picamera' package is not installed. "
            "Run this script on a Raspberry Pi with picamera enabled."
        )

    # Exposure bracketing
    # ─────────────────────────────────────────────────────────────────────────
    # Obstacles are typically mounted on a large DARK board.  Auto-exposure
    # will aggressively boost the overall brightness, which can overexpose the
    # symbol tile and blur fine strokes (e.g. the bottom hook of 'G').
    # Starting at EV=0 (neutral) keeps the symbol tile well-exposed on the
    # first attempt instead of already over-brightening.
    #   attempt 1 → EV  0  (neutral, let auto-gain settle)
    #   attempt 2 → EV +2  (slightly brighter)
    #   attempt 3 → EV +4  (brighter still)
    #   attempt 4 → EV -2  (darker, in case symbol is blown out)
    #   attempt 5 → EV -4
    #   attempt 6 → EV -6
    if attempt <= 3:
        shutter_us = min(1_000_000, 10_000 * (2 ** (attempt - 1)))
        ev_comp    = min(4, 2 * (attempt - 1))   # 0, +2, +4  (was 2, 4, 6)
        iso_value  = 200
    else:
        factor     = max(1, 2 ** (attempt - 3))
        shutter_us = max(1_000, 10_000 // factor)
        ev_comp    = -min(6, 2 * (attempt - 3))
        iso_value  = 100

    with picamera.PiCamera() as cam:
        cam.resolution            = (1280, 960)
        cam.framerate             = 30
        cam.iso                   = iso_value
        cam.exposure_mode         = "auto"
        cam.awb_mode              = "auto"
        cam.exposure_compensation = ev_comp
        time.sleep(0.4)                       # let auto-gain settle
        cam.shutter_speed = shutter_us
        time.sleep(0.1)                       # apply new shutter
        cam.capture(filename, format="jpeg", quality=85)

    log.info(f"📸  Captured → {filename}  (attempt {attempt}, ev={ev_comp}, shutter={shutter_us}µs)")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE RECOGNITION  (via API server /image endpoint)
# ══════════════════════════════════════════════════════════════════════════════

# ── Symbol-region crop ────────────────────────────────────────────────────────
# When the robot faces a tall black obstacle board the symbol tile occupies
# only the upper-centre portion of the frame.  Cropping to that region before
# sending to YOLO removes the large dark board area that throws off both
# auto-exposure and the classifier (dark context makes G look like T, etc.).
#
# Crop box as fractions of (width, height):
#   horizontal: keep centre 70 %  →  x from 15 % to 85 %
#   vertical:   keep top    60 %  →  y from  0 % to 60 %
#
# Set SYMBOL_CROP_ENABLED = False to disable (useful for debugging).
SYMBOL_CROP_ENABLED = True
SYMBOL_CROP_X0 = 0.15   # left  boundary (fraction of image width)
SYMBOL_CROP_X1 = 0.85   # right boundary
SYMBOL_CROP_Y0 = 0.00   # top   boundary
SYMBOL_CROP_Y1 = 0.60   # bottom boundary


def _crop_image_for_symbol(src_path: str) -> bytes:
    """
    Return JPEG bytes of the symbol-region crop of the image at *src_path*.
    Falls back to the raw file bytes if Pillow is not available.
    """
    try:
        from PIL import Image as _Image   # type: ignore
        import io as _io

        with _Image.open(src_path) as img:
            w, h = img.size
            box = (
                int(w * SYMBOL_CROP_X0),
                int(h * SYMBOL_CROP_Y0),
                int(w * SYMBOL_CROP_X1),
                int(h * SYMBOL_CROP_Y1),
            )
            cropped = img.crop(box)
            buf = _io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            log.debug(
                f"Symbol crop applied: {w}×{h} → "
                f"{box[2]-box[0]}×{box[3]-box[1]}  box={box}"
            )
            return buf.getvalue()

    except ImportError:
        log.debug("Pillow not available — sending full image (no crop).")
    except Exception as exc:
        log.warning(f"Crop failed ({exc}) — sending full image.")

    with open(src_path, "rb") as f:
        return f.read()


def send_image_to_api(filename: str) -> str:
    """
    POST a JPEG file to the /image endpoint and return the top class_id string.

    When SYMBOL_CROP_ENABLED is True the image is cropped to the centre-upper
    region (where the symbol tile is) before posting.  This removes the large
    dark obstacle board from the frame, improving YOLO accuracy for symbols
    whose fine strokes (e.g. the bottom hook of 'G') can be washed out by
    auto-exposure bias caused by the dark background.

    Returns "NA" on API error or no detection.
    """
    url = f"http://{API_IP}:{API_PORT}/image"
    try:
        if SYMBOL_CROP_ENABLED:
            img_bytes = _crop_image_for_symbol(filename)
        else:
            with open(filename, "rb") as f:
                img_bytes = f.read()

        resp = requests.post(
            url,
            files={"file": (os.path.basename(filename), img_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 200:
            data     = resp.json()
            segments = data.get("segments", [])
            if segments:
                # Sort by confidence descending; take the top result
                segments.sort(key=lambda s: s.get("confidence", 0), reverse=True)
                top      = segments[0]
                img_id   = str(top.get("class_id", "NA"))
                conf     = top.get("confidence", 0)
                sym_name = SYMBOL_MAP.get(img_id, top.get("class_name", "Unknown"))
                log.info(f"🔍  Recognised: {sym_name} (class_id={img_id}, conf={conf*100:.1f}%)")
                return img_id
            else:
                log.info("🔍  No object detected in image.")
                return "NA"
        else:
            log.error(f"Image API HTTP {resp.status_code}: {resp.text[:200]}")
            return "NA"

    except requests.exceptions.Timeout:
        log.error("Image API timed out (>30 s).")
        return "NA"
    except Exception as exc:
        log.error(f"Image API error: {exc}", exc_info=True)
        return "NA"


def recognise_obstacle(obstacle_id: int, signal: str) -> str:
    """
    Capture image (with up to MAX_SNAP_ATTEMPTS retries) and return the
    recognised class_id string, or "NA" if nothing valid is found.

    Filename format:  <unix_timestamp>_<obstacle_id>_<signal>.jpg
    This matches what api_server.py expects for obstacle_id extraction.
    """
    VALID_IDS = {str(i) for i in range(11, 41)}   # 11–40 are valid symbol IDs
    img_id    = "NA"

    for attempt in range(1, MAX_SNAP_ATTEMPTS + 1):
        filename = f"{int(time.time())}_{obstacle_id}_{signal}.jpg"

        try:
            capture_image(filename, attempt)
        except RuntimeError as exc:
            log.error(f"Capture failed: {exc}")
            break                                  # no point retrying without camera

        img_id = send_image_to_api(filename)

        if img_id in VALID_IDS:
            log.info(f"✅  Obstacle {obstacle_id}: recognised image_id={img_id} on attempt {attempt}")
            return img_id

        log.info(
            f"↩️   Obstacle {obstacle_id}: attempt {attempt}/{MAX_SNAP_ATTEMPTS} → "
            f"got {img_id!r} (not a valid symbol) — retrying …"
        )

    log.warning(f"⚠️  Obstacle {obstacle_id}: gave up after {MAX_SNAP_ATTEMPTS} attempts — returning {img_id!r}")
    return img_id


# ══════════════════════════════════════════════════════════════════════════════
# ALGO → STM32 COMMAND CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_algo_to_stm32(algo_cmd: str) -> Optional[str]:
    """
    Translate a command produced by the algo path planner (helper.py) into the
    STM32 serial text format understood by the firmware.

    Algo format  │  STM32 format        │  Notes
    ─────────────┼──────────────────────┼─────────────────────────────────────
    FW{n}        │  FWD:{n}             │  Forward n cm  (e.g. FW10 → FWD:10)
    BW{n}        │  REV:{n}             │  Reverse n cm  (e.g. BW30 → REV:30)
    FS{n}        │  FWD:{n}             │  Forward step
    BS{n}        │  REV:{n}             │  Backward step
    FR{p}        │  TURNR:{angle}       │  Forward-Right arc  = turn right
    FL{p}        │  TURNL:{angle}       │  Forward-Left  arc  = turn left
    BR{p}        │  REVR:{angle}        │  Backward-Right arc (movement.py: REVR)
    BL{p}        │  REVL:{angle}        │  Backward-Left  arc (movement.py: REVL)
    TL{p}        │  TURNL:{angle}       │  Point turn left
    TR{p}        │  TURNR:{angle}       │  Point turn right
    FWD:* …      │  (pass through)      │  Already STM32 text format
    ─────────────┴──────────────────────┴─────────────────────────────────────
    {n}     = numeric distance in cm  (e.g. 10, 20, 30 …)
    {p}     = algo step param, always "00" for turns (ignored on conversion)
    {angle} = ARC_TURN_ANGLE constant (default 90°); tune to your robot

    Returns the STM32 command string, or None if the input is unrecognised.
    """
    # Already STM32 text format – pass through unchanged
    for pfx in STM32_TEXT_PREFIXES:
        if algo_cmd == pfx.rstrip(":") or algo_cmd.startswith(pfx):
            return algo_cmd

    # Forward straight: FW{n} or FS{n}
    for pfx in ("FW", "FS"):
        if algo_cmd.startswith(pfx):
            n = algo_cmd[len(pfx):]
            if n.isdigit():
                return f"FWD:{int(n)}"

    # Backward straight: BW{n} or BS{n}
    for pfx in ("BW", "BS"):
        if algo_cmd.startswith(pfx):
            n = algo_cmd[len(pfx):]
            if n.isdigit():
                return f"REV:{int(n)}"

    # Forward arc turns
    # FR = Forward-Right  →  TURNR:{angle}
    # FL = Forward-Left   →  TURNL:{angle}
    if algo_cmd.startswith("FR"):
        return f"TURNR:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("FL"):
        return f"TURNL:{ARC_TURN_ANGLE}"

    # Backward arc turns (movement.py pattern: REVL / REVR)
    # BR = Backward-Right  →  REVR:{angle}
    # BL = Backward-Left   →  REVL:{angle}
    if algo_cmd.startswith("BR"):
        return f"REVR:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("BL"):
        return f"REVL:{ARC_TURN_ANGLE}"

    # Point turns
    if algo_cmd.startswith("TL"):
        return f"TURNL:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("TR"):
        return f"TURNR:{ARC_TURN_ANGLE}"

    return None   # unrecognised


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class MainRunner:
    """
    Orchestrates the complete SC2079 task:
      Android BT  →  parse obstacles  →  path plan (API)
      →  execute STM32 commands  →  image recognition (API)
      →  report results to Android
    """

    def __init__(self) -> None:
        self.android = AndroidBT()
        self.stm     = STMSerial()
        # Stores image-recognition results: obstacle_id → class_id string
        self.results: Dict[int, str] = {}

    # ── Entry point (outer loop) ──────────────────────────────────────────────
    def start(self) -> None:
        """
        Main entry point.

        When AUTO_RESTART=True (default) the runner loops indefinitely:
          boot  →  connect STM32  →  wait for Android  →  receive JSON
               →  plan path  →  execute  →  report  →  reset  →  repeat

        This means the script is *always* waiting for Android as soon as the
        RPi finishes one run (or encounters an error), with no manual restart
        needed.  Press Ctrl+C to stop cleanly.
        """
        log.info("=" * 62)
        log.info("  SC2079 Main Runner  –  Navigation + Image Recognition")
        log.info("=" * 62)
        log.info(f"  API server   :  http://{API_IP}:{API_PORT}")
        log.info(f"  STM32 port   :  {SERIAL_PORT}  @  {BAUD_RATE} baud")
        log.info(f"  Image dist   :  {IMAGE_REC_DISTANCE} cm (target viewing distance)")
        log.info(f"  Robot start  :  ({ROBOT_START_X}, {ROBOT_START_Y}) facing North")
        log.info(f"  Auto-restart :  {AUTO_RESTART}")
        log.info("=" * 62)

        run_count = 0
        try:
            while True:
                run_count += 1
                if run_count > 1:
                    log.info(
                        f"🔄  Auto-restart in {RESTART_DELAY:.0f}s "
                        f"(run #{run_count}) …"
                    )
                    time.sleep(RESTART_DELAY)

                should_continue = self._run_once()

                if not should_continue:
                    # Ctrl+C was raised inside _run_once()
                    break
                if not AUTO_RESTART:
                    log.info("AUTO_RESTART=False — exiting after single run.")
                    break

        except KeyboardInterrupt:
            log.info("\nCtrl+C received at outer loop — shutting down.")
        finally:
            log.info("Runner stopped.")

    # ── Single-run logic ──────────────────────────────────────────────────────
    def _run_once(self) -> bool:
        """
        Execute one full task cycle:
          connect STM32  →  wait for Android + obstacle JSON
          →  path plan  →  execute commands  →  report results  →  cleanup

        Returns True  if the run completed normally (success or error)
                      so the caller can restart.
        Returns False if Ctrl+C was pressed (caller should stop looping).
        """
        # Reset per-run state
        self.results   = {}
        self.android   = AndroidBT()
        self.stm       = STMSerial()

        try:
            # Step 1 – Open STM32 serial connection
            self.stm.connect()

            # Step 2 – Verify API server is reachable
            self._check_api()

            # Step 3 – Wait for Android Bluetooth connection
            self.android.connect()
            self._bt_send_json("info", "Connected to RPi. Please send obstacle data.")

            # Step 4 – Receive and parse obstacle JSON
            # The runner blocks here automatically until Android sends the JSON.
            obstacles = self._recv_obstacles()
            if not obstacles:
                log.error("No valid obstacles received — aborting this run.")
                self._bt_send_json("error", "No obstacles received.")
                return True   # allow restart

            log.info(f"Received {len(obstacles)} obstacle(s):")
            for ob in obstacles:
                # ob['d'] is Android format (0=North 1=East 2=South 3=West)
                dir_name = ANDROID_DIR_NAMES.get(ob["d"], f"d={ob['d']}")
                algo_d   = ANDROID_TO_ALGO_DIR.get(ob["d"], "?")
                log.info(
                    f"  id={ob['id']:2d}  pos=({ob['x']:2d},{ob['y']:2d})  "
                    f"face={dir_name}  (android_d={ob['d']} → algo_d={algo_d})"
                )

            self._bt_send_json(
                "info",
                f"Received {len(obstacles)} obstacle(s). Computing optimal path …"
            )

            # Step 5 – Path planning via API
            path_data = plan_path(obstacles)
            if path_data is None:
                log.error("Path planning failed — aborting this run.")
                self._bt_send_json("error", "Path planning failed. Check API server.")
                return True   # allow restart

            commands: List[str] = path_data.get("data", {}).get("commands", [])
            distance: float     = path_data.get("data", {}).get("distance", 0.0)
            path_states: list   = path_data.get("data", {}).get("path", [])

            log.info(
                f"Path ready: {len(commands)} command(s), "
                f"estimated distance={distance:.1f} units"
            )

            # Step 6 – Immediately relay full path info to Android
            self._bt_send_json("path", {
                "commands": commands,
                "distance": distance,
                "path":     path_states,
            })

            # Step 7 – Execute every command (motion + SNAP + FIN)
            self._execute_commands(commands)

            # Step 8 – Send consolidated results summary to Android
            self._report_results()

            return True   # completed normally — allow restart

        except KeyboardInterrupt:
            log.info("Ctrl+C inside run — stopping.")
            return False  # signal outer loop to stop

        except Exception as exc:
            log.error(f"Unhandled error in run: {exc}", exc_info=True)
            try:
                self._bt_send_json("error", str(exc))
            except Exception:
                pass
            return True   # allow restart after error

        finally:
            # Always clean up hardware handles at the end of a run
            try:
                self.stm.send("STOP")
            except Exception:
                pass
            self.stm.close()
            self.android.close()
            log.info("Run cleanup complete.")

    # ── Check API server health ────────────────────────────────────────────────
    def _check_api(self) -> None:
        url = f"http://{API_IP}:{API_PORT}/status"
        log.info(f"Checking API server at {url} …")
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                st = resp.json()
                log.info(
                    f"✅  API running — "
                    f"model={st.get('model')}  "
                    f"yolo={st.get('yolo_available')}  "
                    f"algo={st.get('algorithm_available')}"
                )
            else:
                log.warning(f"API status endpoint returned HTTP {resp.status_code}")
        except Exception as exc:
            log.warning(f"API check failed ({exc}) — continuing anyway.")

    # ── Receive obstacle JSON from Android ────────────────────────────────────
    def _recv_obstacles(self) -> Optional[List[dict]]:
        """
        Block until Android sends a valid obstacle message:
          {"cat": "obstacles", "value": [...]}

        Each obstacle must have keys: x, y, id, d
          d: 0=North  1=East  2=South  3=West
        """
        log.info("Waiting for obstacle JSON from Android …")

        while True:
            raw = self.android.recv()
            if raw is None:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"Non-JSON message ignored: {raw!r}")
                continue

            cat = str(msg.get("cat", "")).lower()
            if cat != "obstacles":
                log.info(f"Ignoring message with cat={cat!r} (waiting for 'obstacles')")
                continue

            value = msg.get("value")
            if not isinstance(value, list) or len(value) == 0:
                log.warning("Obstacle 'value' is empty or not a list — waiting again.")
                continue

            valid_obs: List[dict] = []
            for item in value:
                if all(k in item for k in ("x", "y", "id", "d")):
                    try:
                        valid_obs.append({
                            "x":  int(item["x"]),
                            "y":  int(item["y"]),
                            "id": int(item["id"]),
                            "d":  int(item["d"]),
                        })
                    except (ValueError, TypeError) as exc:
                        log.warning(f"Skipping obstacle with invalid values {item}: {exc}")
                else:
                    log.warning(f"Skipping malformed obstacle entry: {item}")

            if valid_obs:
                return valid_obs
            else:
                log.warning("No valid obstacle entries found in message — waiting again.")

    # ── Execute full command list ──────────────────────────────────────────────
    def _execute_commands(self, commands: List[str]) -> None:
        """
        Iterate through every command from the path planner and dispatch:

          SNAP{id}[_signal]   →  capture + recognise image, store result
          FIN                 →  send STOP to STM32, break
          Algo-format motion  →  convert to STM32 text, send, wait OK/ACK
              FW{n} → FWD:{n}    BW{n} → REV:{n}
              FR00  → TURNR:90   FL00  → TURNL:90
              BR00  → REVR:90    BL00  → REVL:90
              TL00  → TURNL:90   TR00  → TURNR:90
          STM32-text motion   →  pass through, send, wait OK/ACK
              FWD:*  REV:*  TURNL:*  TURNR:*  REVL:*  REVR:*  OLED:*  STOP  RS
          Unknown             →  log warning, skip
        """
        total = len(commands)
        self._stm_timeout_count = 0          # reset consecutive-timeout counter
        log.info(f"Starting execution of {total} command(s) …")

        for idx, cmd in enumerate(commands, 1):
            log.info(f"[{idx:3d}/{total}]  {cmd}")

            # ── SNAP ──────────────────────────────────────────────────────────
            if cmd.startswith("SNAP"):
                self._handle_snap(cmd)

            # ── FIN ───────────────────────────────────────────────────────────
            elif cmd == "FIN":
                log.info("FIN — navigation complete.")
                try:
                    self.stm.send("STOP")
                except Exception:
                    pass
                break

            # ── Motion: algo-format OR already STM32 text format ──────────────
            elif (any(cmd.startswith(pfx) for pfx in ALGO_CMD_PREFIXES)
                  or any(cmd.startswith(pfx) for pfx in STM32_TEXT_PREFIXES)
                  or cmd == "STOP"):
                stm_cmd = convert_algo_to_stm32(cmd)
                if stm_cmd:
                    if stm_cmd != cmd:
                        log.info(f"    ↳ converted: {cmd!r}  →  {stm_cmd!r}")
                    time.sleep(0.5)   # 0.5 s settling lag before each motion command
                    acked = self._send_stm_and_wait_ack(stm_cmd)
                    if not acked:
                        self._stm_timeout_count += 1
                        log.warning(
                            f"  Consecutive STM32 timeouts: "
                            f"{self._stm_timeout_count}/{STM_MAX_CONSECUTIVE_TIMEOUTS}"
                        )
                        if (STM_MAX_CONSECUTIVE_TIMEOUTS > 0
                                and self._stm_timeout_count
                                    >= STM_MAX_CONSECUTIVE_TIMEOUTS):
                            log.error(
                                f"🛑  STM32 unresponsive: "
                                f"{self._stm_timeout_count} consecutive timeouts. "
                                f"Aborting command execution."
                            )
                            raise RuntimeError(
                                f"STM32 stopped responding after "
                                f"{self._stm_timeout_count} consecutive timeouts."
                            )
                    else:
                        self._stm_timeout_count = 0   # reset on success
                else:
                    log.warning(f"Conversion returned None for {cmd!r} — skipped.")

            # ── Unknown ───────────────────────────────────────────────────────
            else:
                log.warning(f"Unknown command skipped: {cmd!r}")

        log.info("Command execution finished.")

    # ── Handle SNAP command ────────────────────────────────────────────────────
    def _handle_snap(self, snap_cmd: str) -> None:
        """
        Parse the SNAP command, capture + recognise the image, and store the result.

        Expected formats:
          SNAP{obstacle_id}           e.g.  SNAP3
          SNAP{obstacle_id}_{signal}  e.g.  SNAP3_C   SNAP3_L   SNAP3_R
        """
        body = snap_cmd[4:]        # strip leading 'SNAP'

        if "_" in body:
            parts, signal = body.split("_", 1), body.split("_", 1)[1]
            obstacle_id_str = body.split("_", 1)[0]
        else:
            obstacle_id_str = body
            signal          = "C"    # default centre signal

        try:
            obstacle_id = int(obstacle_id_str)
        except ValueError:
            log.error(f"Cannot parse obstacle_id from SNAP command: {snap_cmd!r}")
            return

        log.info(f"📷  SNAP  obstacle_id={obstacle_id}  signal={signal}")
        self._bt_send_json("snap", f"Capturing image for obstacle {obstacle_id} (signal={signal}) …")

        img_id   = recognise_obstacle(obstacle_id, signal)
        sym_name = SYMBOL_MAP.get(img_id, img_id)

        self.results[obstacle_id] = img_id
        log.info(f"🏷️   Obstacle {obstacle_id}  →  {sym_name} (class_id={img_id})")

        # Send individual result to Android immediately
        self._bt_send_json("result", {
            "obstacle_id": obstacle_id,
            "image_id":    img_id,
            "image_name":  sym_name,
            "signal":      signal,
        })

        # Send TARGET message: plain-text "TARGET,<obstacle_id>,<image_id>"
        # obstacle_id = the original obstacle being scanned
        # img_id      = the recognised class_id (detected value)
        target_msg = f"TARGET,{obstacle_id},{img_id}"
        log.info(f"📡  Sending TARGET → {target_msg!r}")
        self.android.send(target_msg)

    # ── Send STM32 command and wait for OK / ACK ──────────────────────────────
    def _send_stm_and_wait_ack(self, cmd: str) -> bool:
        """
        Send one STM32 text-format command over serial and block until the
        firmware replies with a line starting with "OK" or "ACK".

        Design notes
        ────────────
        • The serial port is opened with a SHORT per-read timeout
          (STM_SERIAL_READ_TIMEOUT = 0.1 s) so each readline() call returns
          quickly even when no data arrives.  This prevents readline() from
          blocking for 15 s when only a partial line is in the RX buffer.
        • The outer deadline loop (STM_ACK_TIMEOUT) is therefore always
          honoured accurately.
        • The RX buffer is flushed BEFORE sending so stale bytes from a
          previous late ACK cannot be mistaken for the current command's ACK.

        Returns True if an ACK was received, False on timeout.
        """
        ser = self.stm._ser
        if ser is None or not ser.is_open:
            log.error(f"STM32 serial not open — cannot send {cmd!r}")
            return False

        # Flush stale bytes so we don't read a leftover ACK from a prior cmd
        ser.reset_input_buffer()

        try:
            self.stm.send(cmd)
        except Exception as exc:
            log.error(f"STM32 send failed for {cmd!r}: {exc}")
            return False

        deadline    = time.time() + STM_ACK_TIMEOUT
        line_buf    = b""            # accumulate bytes across short reads

        while time.time() < deadline:
            # readline() returns after STM_SERIAL_READ_TIMEOUT seconds if no '\n'
            chunk = ser.readline()   # non-blocking style (short timeout on port)

            if chunk:
                line_buf += chunk
                # Check for complete line(s) in buffer
                while b"\n" in line_buf:
                    raw_line, line_buf = line_buf.split(b"\n", 1)
                    line = raw_line.strip().decode("utf-8", errors="ignore")
                    if not line:
                        continue
                    log.debug(f"← STM32: {line!r}")
                    if any(line.upper().startswith(pfx) for pfx in STM32_ACK_PREFIXES):
                        log.info(f"✅  STM32 OK for: {cmd!r}  (reply: {line!r})")
                        return True
                    # Non-ACK lines (debug prints, status, etc.) – keep waiting
                    log.debug(f"STM32 non-ACK while waiting for {cmd!r}: {line!r}")
            # readline() returned empty (timeout) – check deadline and loop

        log.warning(
            f"⚠️  No OK/ACK within {STM_ACK_TIMEOUT:.0f}s for {cmd!r} — continuing."
        )
        return False

    # ── Report final results to Android ───────────────────────────────────────
    def _report_results(self) -> List[dict]:
        """
        Print the final image-recognition results to the terminal and send them
        to Android as a single JSON message.

        Terminal output format:
          ════════════════════════════════════════════════════
           FINAL RESULTS  –  Image Recognition Complete
          ════════════════════════════════════════════════════
           Obstacle  6  →  [20] A
           Obstacle  5  →  [14] Four
           Obstacle  7  →  [11] One
           Obstacle  4  →  [36] Up Arrow
          ════════════════════════════════════════════════════
           4 obstacle(s) processed.
          ════════════════════════════════════════════════════

        Android JSON:
          {"cat": "results",
           "value": [{"obstacle_id": int, "image_id": str, "image_name": str}, ...]}

        Returns the summary list (also stored in self.results).
        """
        div = "═" * 52

        # ── Terminal / log ─────────────────────────────────────────────────────
        print(f"\n{div}")
        print("  FINAL RESULTS  –  Image Recognition Complete")
        print(div)
        log.info(div)
        log.info("  FINAL RESULTS  –  Image Recognition Complete")
        log.info(div)

        summary = []
        for ob_id, img_id in sorted(self.results.items()):
            sym_name = SYMBOL_MAP.get(img_id, img_id)
            line = f"  Obstacle {ob_id:2d}  →  [{img_id:>3}] {sym_name}"
            print(line)
            log.info(line)
            summary.append({
                "obstacle_id": ob_id,
                "image_id":    img_id,
                "image_name":  sym_name,
            })

        print(div)
        print(f"  {len(summary)} obstacle(s) processed.")
        print(f"{div}\n")
        log.info(div)
        log.info(f"  {len(summary)} obstacle(s) processed.")
        log.info(div)

        # ── Send to Android ────────────────────────────────────────────────────
        self._bt_send_json("results", summary)
        log.info("✅  Results sent to Android.")

        # ── Also print the raw JSON to stdout for easy copy-paste / debugging ──
        import json as _json
        print("  Android payload:")
        print("  " + _json.dumps({"cat": "results", "value": summary}, indent=2)
              .replace("\n", "\n  "))
        print()

        return summary

    # ── Helper: send JSON to Android ──────────────────────────────────────────
    def _bt_send_json(self, cat: str, value) -> None:
        """Serialise and send a {"cat": cat, "value": value} message to Android."""
        try:
            self.android.send(json.dumps({"cat": cat, "value": value}))
        except OSError:
            log.warning("Could not send to Android (connection lost?).")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = MainRunner()
    runner.start()
