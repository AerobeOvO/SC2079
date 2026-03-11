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
  5. Execute each command in order.
  6. Send final image-recognition summary back to Android.
  7. Trigger PC to display stitched image results grid.
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

BT_SERVICE_NAME = "MDP-RPi-Runner"
BT_UUID         = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

IMAGE_REC_DISTANCE = 25
ROBOT_START_X   = 1    
ROBOT_START_Y   = 1
ROBOT_START_DIR = 0    

STM_ACK_TIMEOUT = 15.0
STM_SERIAL_READ_TIMEOUT = 0.1
STM_MAX_CONSECUTIVE_TIMEOUTS = 5

MAX_SNAP_ATTEMPTS = 3
AUTO_RESTART = False
RESTART_DELAY = 3.0

ANDROID_TO_ALGO_DIR: Dict[int, int] = {0: 0, 1: 2, 2: 4, 3: 6}
ANDROID_DIR_NAMES: Dict[int, str] = {0: "North", 1: "East", 2: "South", 3: "West"}
ALGO_DIR_NAMES: Dict[int, str] = {0: "North", 2: "East", 4: "South", 6: "West", 8: "Skip"}

ARC_TURN_ANGLE: int = 90

ALGO_CMD_PREFIXES = (
    "FW", "BW",       
    "FS", "BS",       
    "FR", "FL",       
    "BR", "BL",       
    "TL", "TR",       
)

STM32_TEXT_PREFIXES = (
    "FWD:", "REV:",             
    "TURNL:", "TURNR:",         
    "REVL:", "REVR:",           
    "OLED:", "STOP", "RS",      
)

STM32_ACK_PREFIXES = ("OK", "ACK")

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
    def __init__(self) -> None:
        self._server: Optional[bluetooth.BluetoothSocket] = None
        self._client: Optional[bluetooth.BluetoothSocket] = None
        self._buf: str = ""

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
        self._client, info = self._server.accept()
        log.info(f"✅  Android connected from {info}")

    def send(self, text: str) -> None:
        msg = text.rstrip("\n") + "\n"
        try:
            self._client.send(msg.encode("utf-8"))
            log.debug(f"→ Android: {msg.strip()!r}")
        except OSError as exc:
            log.error(f"BT send error: {exc}")

    def recv(self) -> Optional[str]:
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
    def __init__(self) -> None:
        self._ser: Optional[serial.Serial] = None

    def connect(self) -> None:
        log.info(f"Opening serial port {SERIAL_PORT} @ {BAUD_RATE} baud …")
        self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=STM_SERIAL_READ_TIMEOUT)
        time.sleep(2)                       
        self._ser.reset_input_buffer()
        log.info("✅  STM32 serial connected.")

    def send(self, cmd: str) -> None:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("STM32 serial port is not open.")
        payload = (cmd + "\r\n").encode("utf-8")
        log.debug(f"→ STM32: {cmd!r}")
        self._ser.write(payload)
        self._ser.flush()

    def recv(self) -> Optional[str]:
        if self._ser is None or not self._ser.is_open:
            return None
        raw = self._ser.readline()
        if raw:
            decoded = raw.strip().decode("utf-8", errors="ignore")
            log.debug(f"← STM32: {decoded!r}")
            return decoded
        return None

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            log.info("STM32 serial closed.")


# ══════════════════════════════════════════════════════════════════════════════
# PATH PLANNING  (via API server /path endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def plan_path(obstacles_android: List[dict]) -> Optional[dict]:
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
    try:
        import picamera                       
    except ImportError:
        raise RuntimeError(
            "The 'picamera' package is not installed. "
            "Run this script on a Raspberry Pi with picamera enabled."
        )

    if attempt <= 3:
        shutter_us = min(1_000_000, 10_000 * (2 ** (attempt - 1)))
        ev_comp    = min(4, 2 * (attempt - 1))   
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
        time.sleep(0.4)                       
        cam.shutter_speed = shutter_us
        time.sleep(0.1)                       
        cam.capture(filename, format="jpeg", quality=85)

    log.info(f"📸  Captured → {filename}  (attempt {attempt}, ev={ev_comp}, shutter={shutter_us}µs)")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE RECOGNITION  (via API server /image endpoint)
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_CROP_ENABLED = False
SYMBOL_CROP_X0 = 0.15   
SYMBOL_CROP_X1 = 0.85   
SYMBOL_CROP_Y0 = 0.00   
SYMBOL_CROP_Y1 = 0.60   

def _crop_image_for_symbol(src_path: str) -> bytes:
    try:
        from PIL import Image as _Image   
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
            return buf.getvalue()

    except ImportError:
        pass
    except Exception as exc:
        log.warning(f"Crop failed ({exc}) — sending full image.")

    with open(src_path, "rb") as f:
        return f.read()


def send_image_to_api(filename: str) -> str:
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
            timeout=45,   
        )
        if resp.status_code == 200:
            data     = resp.json()
            segments = data.get("segments", [])
            if segments:
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
        log.error("Image API timed out (>45 s).")
        return "NA"
    except Exception as exc:
        log.error(f"Image API error: {exc}", exc_info=True)
        return "NA"


def recognise_obstacle(obstacle_id: int, signal: str) -> str:
    VALID_IDS = {str(i) for i in range(11, 41)}   
    img_id    = "NA"

    for attempt in range(1, MAX_SNAP_ATTEMPTS + 1):
        filename = f"{int(time.time())}_{obstacle_id}_{signal}.jpg"

        try:
            capture_image(filename, attempt)
        except RuntimeError as exc:
            log.error(f"Capture failed: {exc}")
            break                                  

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
    for pfx in STM32_TEXT_PREFIXES:
        if algo_cmd == pfx.rstrip(":") or algo_cmd.startswith(pfx):
            return algo_cmd

    for pfx in ("FW", "FS"):
        if algo_cmd.startswith(pfx):
            n = algo_cmd[len(pfx):]
            if n.isdigit():
                return f"FWD:{int(n)}"

    for pfx in ("BW", "BS"):
        if algo_cmd.startswith(pfx):
            n = algo_cmd[len(pfx):]
            if n.isdigit():
                return f"REV:{int(n)}"

    if algo_cmd.startswith("FR"):
        return f"TURNR:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("FL"):
        return f"TURNL:{ARC_TURN_ANGLE}"

    if algo_cmd.startswith("BR"):
        return f"REVR:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("BL"):
        return f"REVL:{ARC_TURN_ANGLE}"

    if algo_cmd.startswith("TL"):
        return f"TURNL:{ARC_TURN_ANGLE}"
    if algo_cmd.startswith("TR"):
        return f"TURNR:{ARC_TURN_ANGLE}"

    return None   


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class MainRunner:
    def __init__(self) -> None:
        self.android = AndroidBT()
        self.stm     = STMSerial()
        self.results: Dict[int, str] = {}

    def start(self) -> None:
        log.info("=" * 62)
        log.info("  SC2079 Main Runner  –  Navigation + Image Recognition")
        log.info("=" * 62)
        log.info(f"  API server   :  http://{API_IP}:{API_PORT}")
        log.info(f"  STM32 port   :  {SERIAL_PORT}  @  {BAUD_RATE} baud")
        log.info(f"  Image dist   :  {IMAGE_REC_DISTANCE} cm")
        log.info(f"  Robot start  :  ({ROBOT_START_X}, {ROBOT_START_Y}) facing North")
        log.info(f"  Auto-restart :  {AUTO_RESTART}")
        log.info("=" * 62)

        run_count = 0
        try:
            while True:
                run_count += 1
                if run_count > 1:
                    log.info(f"🔄  Auto-restart in {RESTART_DELAY:.0f}s (run #{run_count}) …")
                    time.sleep(RESTART_DELAY)

                should_continue = self._run_once()

                if not should_continue:
                    break
                if not AUTO_RESTART:
                    log.info("AUTO_RESTART=False — exiting after single run.")
                    break

        except KeyboardInterrupt:
            log.info("\nCtrl+C received at outer loop — shutting down.")
        finally:
            log.info("Runner stopped.")

    def _run_once(self) -> bool:
        self.results   = {}
        self.android   = AndroidBT()
        self.stm       = STMSerial()

        try:
            self.stm.connect()
            self._check_api()
            self.android.connect()
            self._bt_send_json("info", "Connected to RPi. Please send obstacle data.")

            obstacles = self._recv_obstacles()
            if not obstacles:
                log.error("No valid obstacles received — aborting this run.")
                self._bt_send_json("error", "No obstacles received.")
                return True   

            log.info(f"Received {len(obstacles)} obstacle(s):")
            for ob in obstacles:
                dir_name = ANDROID_DIR_NAMES.get(ob["d"], f"d={ob['d']}")
                algo_d   = ANDROID_TO_ALGO_DIR.get(ob["d"], "?")
                log.info(
                    f"  id={ob['id']:2d}  pos=({ob['x']:2d},{ob['y']:2d})  "
                    f"face={dir_name}  (android_d={ob['d']} → algo_d={algo_d})"
                )

            self._bt_send_json("info", f"Received {len(obstacles)} obstacle(s). Computing optimal path …")

            path_data = plan_path(obstacles)
            if path_data is None:
                log.error("Path planning failed — aborting this run.")
                self._bt_send_json("error", "Path planning failed. Check API server.")
                return True   

            commands: List[str] = path_data.get("data", {}).get("commands", [])
            distance: float     = path_data.get("data", {}).get("distance", 0.0)
            path_states: list   = path_data.get("data", {}).get("path", [])

            log.info(f"Path ready: {len(commands)} command(s), estimated distance={distance:.1f} units")

            self._bt_send_json("path", {
                "commands": commands,
                "distance": distance,
                "path":     path_states,
            })

            # Clear previous run's snap images so the tiled display is fresh
            try:
                reset_url = f"http://{API_IP}:{API_PORT}/stitch-reset"
                requests.get(reset_url, timeout=5)
                log.info("🗑️   Previous snap images cleared (stitch-reset).")
            except Exception as exc:
                log.warning(f"stitch-reset failed (non-critical): {exc}")

            self._execute_commands(commands)
            self._report_results(total_obstacles=len(obstacles))

            return True   

        except KeyboardInterrupt:
            log.info("Ctrl+C inside run — stopping.")
            return False  

        except Exception as exc:
            log.error(f"Unhandled error in run: {exc}", exc_info=True)
            try:
                self._bt_send_json("error", str(exc))
            except Exception:
                pass
            return True   

        finally:
            try:
                self.stm.send("STOP")
            except Exception:
                pass
            self.stm.close()
            self.android.close()
            log.info("Run cleanup complete.")

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
        except Exception as exc:
            log.warning(f"API check failed ({exc}) — continuing anyway.")

    def _recv_obstacles(self) -> Optional[List[dict]]:
        log.info("Waiting for obstacle JSON from Android …")

        while True:
            raw = self.android.recv()
            if raw is None:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cat = str(msg.get("cat", "")).lower()
            if cat != "obstacles":
                continue

            value = msg.get("value")
            if not isinstance(value, list) or len(value) == 0:
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
                    except (ValueError, TypeError):
                        pass

            if valid_obs:
                return valid_obs

    def _execute_commands(self, commands: List[str]) -> None:
        total = len(commands)
        self._stm_timeout_count = 0          
        log.info(f"Starting execution of {total} command(s) …")

        for idx, cmd in enumerate(commands, 1):
            log.info(f"[{idx:3d}/{total}]  {cmd}")

            if cmd.startswith("SNAP"):
                self._handle_snap(cmd)

            elif cmd == "FIN":
                log.info("FIN — navigation complete.")
                try:
                    self.stm.send("STOP")
                except Exception:
                    pass
                break

            elif (any(cmd.startswith(pfx) for pfx in ALGO_CMD_PREFIXES)
                  or any(cmd.startswith(pfx) for pfx in STM32_TEXT_PREFIXES)
                  or cmd == "STOP"):
                stm_cmd = convert_algo_to_stm32(cmd)
                if stm_cmd:
                    time.sleep(0.5)   
                    acked = self._send_stm_and_wait_ack(stm_cmd)
                    if not acked:
                        self._stm_timeout_count += 1
                        if (STM_MAX_CONSECUTIVE_TIMEOUTS > 0
                                and self._stm_timeout_count
                                    >= STM_MAX_CONSECUTIVE_TIMEOUTS):
                            raise RuntimeError(
                                f"STM32 stopped responding after "
                                f"{self._stm_timeout_count} consecutive timeouts."
                            )
                    else:
                        self._stm_timeout_count = 0   
                else:
                    log.warning(f"Conversion returned None for {cmd!r} — skipped.")

            else:
                log.warning(f"Unknown command skipped: {cmd!r}")

        log.info("Command execution finished.")

    def _handle_snap(self, snap_cmd: str) -> None:
        body = snap_cmd[4:]        

        if "_" in body:
            parts, signal = body.split("_", 1), body.split("_", 1)[1]
            obstacle_id_str = body.split("_", 1)[0]
        else:
            obstacle_id_str = body
            signal          = "C"    

        try:
            obstacle_id = int(obstacle_id_str)
        except ValueError:
            return

        log.info(f"📷  SNAP  obstacle_id={obstacle_id}  signal={signal}")
        self._bt_send_json("snap", f"Capturing image for obstacle {obstacle_id} (signal={signal}) …")

        img_id   = recognise_obstacle(obstacle_id, signal)
        sym_name = SYMBOL_MAP.get(img_id, img_id)

        self.results[obstacle_id] = img_id
        log.info(f"🏷️   Obstacle {obstacle_id}  →  {sym_name} (class_id={img_id})")

        self._bt_send_json("result", {
            "obstacle_id": obstacle_id,
            "image_id":    img_id,
            "image_name":  sym_name,
            "signal":      signal,
        })

        target_msg = f"TARGET,{obstacle_id},{img_id}"
        log.info(f"📡  Sending TARGET → {target_msg!r}")
        self.android.send(target_msg)

    def _send_stm_and_wait_ack(self, cmd: str) -> bool:
        ser = self.stm._ser
        if ser is None or not ser.is_open:
            return False

        ser.reset_input_buffer()

        try:
            self.stm.send(cmd)
        except Exception as exc:
            return False

        deadline    = time.time() + STM_ACK_TIMEOUT
        line_buf    = b""            

        while time.time() < deadline:
            chunk = ser.readline()   

            if chunk:
                line_buf += chunk
                while b"\n" in line_buf:
                    raw_line, line_buf = line_buf.split(b"\n", 1)
                    line = raw_line.strip().decode("utf-8", errors="ignore")
                    if not line:
                        continue
                    if any(line.upper().startswith(pfx) for pfx in STM32_ACK_PREFIXES):
                        return True
                    
        return False

    def _report_results(self, total_obstacles: int = 0) -> List[dict]:
        div = "═" * 52

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

        self._bt_send_json("results", summary)
        log.info("✅  Results sent to Android.")

        import json as _json
        print("  Android payload:")
        print("  " + _json.dumps({"cat": "results", "value": summary}, indent=2)
              .replace("\n", "\n  "))
        print()

        # --- Trigger PC tiled display (Point 7) ---
        try:
            stitch_url = f"http://{API_IP}:{API_PORT}/stitch?total={total_obstacles}"
            log.info(f"Triggering PC tiled display: {stitch_url} …")
            requests.get(stitch_url, timeout=5)
        except Exception as exc:
            log.warning(f"⚠️  Could not trigger PC stitch display: {exc}")
        # -------------------------------------------

        return summary

    def _bt_send_json(self, cat: str, value) -> None:
        try:
            self.android.send(json.dumps({"cat": cat, "value": value}))
        except OSError:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = MainRunner()
    runner.start()