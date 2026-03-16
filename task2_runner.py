#!/usr/bin/env python3
"""
task2_runner.py  –  SC2079 Task 2: Fastest Car Task
=====================================================

ARCHITECTURE
────────────
  The STM32 drives the robot autonomously using its ultrasonic sensor.
  The RPi does NOT send movement commands — it acts as a vision service.

  Protocol between STM32 ↔ RPi (serial):
    STM32 → RPi : "Capture 1"   robot has reached Obstacle 1
    RPi   → STM32: "RIGHT" | "LEFT"   detected arrow direction
    STM32 → RPi : "Capture 2"   robot has reached Obstacle 2
    RPi   → STM32: "RIGHT" | "LEFT"
    STM32 → RPi : "FIN"          robot back in carpark, task complete

FLOW
────
  1. Connect to STM32 (serial) and Android (Bluetooth).
  2. Wait for START from Android → send "START" to STM32.
  3. Wait for "Capture 1" from STM32.
       → SNAP → detect arrow → reply "RIGHT" or "LEFT" to STM32.
       → Report result to Android.
  4. Wait for "Capture 2" from STM32.
       → SNAP → detect arrow → reply "RIGHT" or "LEFT" to STM32.
       → Report result to Android.
  5. Wait for "FIN" from STM32.
       → Trigger PC tiled display (Point 8).
       → Send final summary to Android.
"""

import bluetooth
import json
import os
import requests
import serial
import sys
import time
from typing import Dict, List, Optional, Tuple

# ─── Project root importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import API_IP, API_PORT, SERIAL_PORT, BAUD_RATE
from logger import prepare_logger

log = prepare_logger()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Arrow class IDs in best.pt ──────────────────────────────────────────── 
ARROW_RIGHT = "1"
ARROW_LEFT  = "0"

# ── What STM32 sends over serial when it reaches an obstacle ─────────────────
# Matching is case-insensitive; partial prefix match is used.
CAPTURE_OBS1_TRIGGERS = ("capture 1", "cap1", "capture1", "snap1", "obs1")
CAPTURE_OBS2_TRIGGERS = ("capture 2", "cap2", "capture2", "snap2", "obs2")
TASK_DONE_TRIGGERS    = ("fin", "done", "complete", "finish")

# ── What RPi replies to STM32 with the detected direction ────────────────────
REPLY_RIGHT = "RIGHT"
REPLY_LEFT  = "LEFT"

# ── Bluetooth ─────────────────────────────────────────────────────────────────
BT_SERVICE_NAME = "MDP-RPi-T2"
BT_UUID         = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

# ── Serial / STM32 ───────────────────────────────────────────────────────────
STM_SERIAL_READ_TIMEOUT = 0.05   # seconds — keep low for fast polling

# ── Image recognition ─────────────────────────────────────────────────────────
MAX_SNAP_ATTEMPTS = 3   # retries if no valid arrow detected

SYMBOL_MAP: Dict[str, str] = {
    ARROW_RIGHT: "Right Arrow",
    ARROW_LEFT:  "Left Arrow",
    "36": "Up Arrow",
    "37": "Down Arrow",
}


# ══════════════════════════════════════════════════════════════════════════════
# BLUETOOTH
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
            self._server, BT_SERVICE_NAME,
            service_id=BT_UUID,
            service_classes=[BT_UUID, bluetooth.SERIAL_PORT_CLASS],
            profiles=[bluetooth.SERIAL_PORT_PROFILE],
        )
        log.info(f"📡  Waiting for Android on RFCOMM channel {port} …")
        self._client, info = self._server.accept()
        log.info(f"✅  Android connected from {info}")

    def send(self, text: str) -> None:
        try:
            self._client.send((text.rstrip("\n") + "\n").encode("utf-8"))
        except OSError as exc:
            log.error(f"BT send error: {exc}")

    def send_json(self, cat: str, value) -> None:
        try:
            self.send(json.dumps({"cat": cat, "value": value}))
        except OSError:
            pass

    def recv_nonblocking(self) -> Optional[str]:
        """Return a complete line if available, else None (non-blocking)."""
        try:
            self._client.setblocking(False)
            chunk = self._client.recv(4096)
            if chunk:
                self._buf += chunk.decode("utf-8", errors="ignore")
        except BlockingIOError:
            pass
        except OSError:
            pass
        finally:
            try:
                self._client.setblocking(True)
            except OSError:
                pass
        if "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            return line.strip() or None
        return None

    def recv_blocking(self) -> Optional[str]:
        """Block until a full line arrives."""
        while True:
            if "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    return line.strip()
                continue
            chunk = self._client.recv(4096)
            if not chunk:
                raise OSError("Android socket closed")
            self._buf += chunk.decode("utf-8", errors="ignore")

    def close(self) -> None:
        for s in (self._client, self._server):
            try:
                if s: s.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# STM32 SERIAL
# ══════════════════════════════════════════════════════════════════════════════

class STMSerial:
    def __init__(self) -> None:
        self._ser: Optional[serial.Serial] = None
        self._buf: str = ""

    def connect(self) -> None:
        log.info(f"Opening serial {SERIAL_PORT} @ {BAUD_RATE} baud …")
        self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE,
                                   timeout=STM_SERIAL_READ_TIMEOUT)
        time.sleep(2)
        self._ser.reset_input_buffer()
        log.info("✅  STM32 connected.")

    def send(self, msg: str) -> None:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("STM32 serial not open")
        payload = (msg.rstrip("\r\n") + "\r\n").encode("utf-8")
        self._ser.write(payload)
        self._ser.flush()
        log.info(f"→ STM32: {msg!r}")

    def readline(self) -> Optional[str]:
        """Return one complete line from STM32, or None if none yet."""
        if self._ser is None or not self._ser.is_open:
            return None
        raw = self._ser.readline()
        if raw:
            chunk = raw.decode("utf-8", errors="ignore")
            self._buf += chunk
        if "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            decoded = line.strip()
            if decoded:
                log.debug(f"← STM32: {decoded!r}")
                return decoded
        return None

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE CAPTURE + RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def capture_image(filename: str, attempt: int = 1) -> None:
    try:
        import picamera
    except ImportError:
        raise RuntimeError("picamera not installed — run on Raspberry Pi.")
    shutter_us = min(1_000_000, 10_000 * (2 ** max(0, attempt - 1)))
    ev_comp    = min(4, 2 * max(0, attempt - 1))
    with picamera.PiCamera() as cam:
        cam.resolution            = (1280, 960)
        cam.framerate             = 30
        cam.iso                   = 200
        cam.exposure_mode         = "auto"
        cam.awb_mode              = "auto"
        cam.exposure_compensation = ev_comp
        time.sleep(0.4)
        cam.shutter_speed = shutter_us
        time.sleep(0.1)
        cam.capture(filename, format="jpeg", quality=85)
    log.info(f"📸  {filename}  (attempt {attempt})")


def snap_and_detect(obstacle_id: int) -> Tuple[str, str]:
    """
    Capture image(s) of obstacle_id until a Left/Right arrow is detected.
    Returns (arrow_id, reply_str) e.g. ("38", "RIGHT") or ("39", "LEFT").
    Defaults to RIGHT if no arrow is found after MAX_SNAP_ATTEMPTS.
    """
    url   = f"http://{API_IP}:{API_PORT}/image"
    valid = {ARROW_RIGHT, ARROW_LEFT}

    for attempt in range(1, MAX_SNAP_ATTEMPTS + 1):
        fname = f"{int(time.time())}_{obstacle_id}_C.jpg"
        try:
            capture_image(fname, attempt)
        except RuntimeError as exc:
            log.error(f"Capture failed: {exc}")
            break

        try:
            with open(fname, "rb") as f:
                img_bytes = f.read()

            resp = requests.post(
                url,
                files={"file": (os.path.basename(fname), img_bytes, "image/jpeg")},
                timeout=45,
            )
            if resp.status_code != 200:
                log.error(f"Image API HTTP {resp.status_code}")
                continue

            data     = resp.json()
            segments = data.get("segments", [])

            if segments:
                segments.sort(key=lambda s: s.get("confidence", 0), reverse=True)
                top    = segments[0]
                img_id = str(top.get("class_id", "NA"))
                conf   = top.get("confidence", 0)
                name   = SYMBOL_MAP.get(img_id, img_id)
                log.info(f"🔍  Obs {obstacle_id}: {name}  "
                         f"(id={img_id}, conf={conf*100:.1f}%)")
                if img_id in valid:
                    reply = REPLY_RIGHT if img_id == ARROW_RIGHT else REPLY_LEFT
                    return img_id, reply
                log.info(f"   ↩ id={img_id} is not Left/Right — retrying …")
            else:
                log.info(f"   ↩ No detection — retrying …")

        except Exception as exc:
            log.error(f"API error: {exc}")

    log.warning(f"⚠️  Obs {obstacle_id}: no arrow after {MAX_SNAP_ATTEMPTS} "
                "attempts — defaulting to RIGHT")
    return ARROW_RIGHT, REPLY_RIGHT


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _matches(line: str, triggers: Tuple[str, ...]) -> bool:
    """Case-insensitive check: does line start with any trigger string?"""
    lower = line.strip().lower()
    return any(lower.startswith(t) for t in triggers)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class Task2Runner:
    def __init__(self) -> None:
        self.android  = AndroidBT()
        self.stm      = STMSerial()
        self.obs1_dir: Optional[str] = None   # "RIGHT" or "LEFT"
        self.obs2_dir: Optional[str] = None

    # ── Entry point ──────────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("=" * 60)
        log.info("  SC2079 Task 2 Runner  –  Fastest Car Task")
        log.info("=" * 60)
        log.info(f"  API   : http://{API_IP}:{API_PORT}")
        log.info(f"  Serial: {SERIAL_PORT}  @  {BAUD_RATE} baud")
        log.info("  (STM32 drives autonomously via ultrasonic sensor)")
        log.info("  Waiting for Capture signals: 'Capture 1', 'Capture 2'")
        log.info("=" * 60)

        try:
            self.stm.connect()
            self._check_api()
            self.android.connect()
            self.android.send_json("info",
                "Task 2 ready. Press START when supervisor approves.")

            self._wait_for_start()
            self._run()

        except KeyboardInterrupt:
            log.info("Ctrl+C — stopping.")
        except Exception as exc:
            log.error(f"Fatal: {exc}", exc_info=True)
            try:
                self.android.send_json("error", str(exc))
            except Exception:
                pass
        finally:
            try:
                self.stm.send("STOP")
            except Exception:
                pass
            self.stm.close()
            self.android.close()
            log.info("Task 2 runner stopped.")

    # ── Wait for Android START ────────────────────────────────────────────────

    def _wait_for_start(self) -> None:
        log.info("Waiting for START from Android …")
        while True:
            raw = self.android.recv_blocking()
            if raw is None:
                continue
            try:
                msg = json.loads(raw)
                cat = str(msg.get("cat", "")).lower()
                val = str(msg.get("value", "")).lower()
                if cat in ("start", "task2_start", "task2:start") or val in ("start", "go"):
                    break
            except json.JSONDecodeError:
                if raw.strip().lower() in ("start", "go", "task2:start"):
                    break

        log.info("✅  START received — signalling STM32 …")
        self.stm.send("START")

        # Reset tiled display for this run
        try:
            requests.get(f"http://{API_IP}:{API_PORT}/stitch-reset", timeout=5)
        except Exception:
            pass

        self.android.send_json("info", "Task 2 started — robot driving …")

    # ── Main event loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        t_start   = time.time()
        obs_done  = 0          # how many obstacles have been processed (0, 1, 2)
        task_over = False

        log.info("Listening for 'Capture 1' / 'Capture 2' / 'FIN' from STM32 …")

        while not task_over:
            line = self.stm.readline()
            if line is None:
                time.sleep(0.01)
                continue

            log.info(f"← STM32: {line!r}")

            # ── Obstacle 1 reached ───────────────────────────────────────────
            if obs_done == 0 and _matches(line, CAPTURE_OBS1_TRIGGERS):
                log.info("📷  Obstacle 1 reached — snapping …")
                self.android.send_json("info", "Obstacle 1 — capturing image …")
                img_id, reply = snap_and_detect(1)
                self.obs1_dir = reply
                name = SYMBOL_MAP.get(img_id, img_id)
                log.info(f"🏹  Obstacle 1: {name}  →  replying {reply!r}")
                self.stm.send(reply)
                self.android.send_json("result", {
                    "obstacle": 1,
                    "image_id": img_id,
                    "image_name": name,
                    "direction": reply,
                })
                obs_done = 1

            # ── Obstacle 2 reached ───────────────────────────────────────────
            elif obs_done == 1 and _matches(line, CAPTURE_OBS2_TRIGGERS):
                log.info("📷  Obstacle 2 reached — snapping …")
                self.android.send_json("info", "Obstacle 2 — capturing image …")
                img_id, reply = snap_and_detect(2)
                self.obs2_dir = reply
                name = SYMBOL_MAP.get(img_id, img_id)
                log.info(f"🏹  Obstacle 2: {name}  →  replying {reply!r}")
                self.stm.send(reply)
                self.android.send_json("result", {
                    "obstacle": 2,
                    "image_id": img_id,
                    "image_name": name,
                    "direction": reply,
                })
                obs_done = 2

            # ── Task complete (robot parked) ──────────────────────────────────
            elif _matches(line, TASK_DONE_TRIGGERS):
                task_over = True

            # ── Unexpected capture signals (safety) ───────────────────────────
            elif (_matches(line, CAPTURE_OBS1_TRIGGERS) or
                  _matches(line, CAPTURE_OBS2_TRIGGERS)):
                log.warning(f"Unexpected capture signal at obs_done={obs_done}: "
                            f"{line!r} — ignored.")

        # ── Post-run ─────────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        log.info(f"🏁  Task 2 complete in {elapsed:.1f} s  "
                 f"(obs1={self.obs1_dir}, obs2={self.obs2_dir})")

        self.android.send_json("finish", {
            "message":   "Task 2 complete — robot parked.",
            "elapsed_s": round(elapsed, 1),
            "obs1":      self.obs1_dir,
            "obs2":      self.obs2_dir,
        })

        # Trigger PC tiled display (Point 8 requirement)
        try:
            requests.get(f"http://{API_IP}:{API_PORT}/stitch?total=2", timeout=5)
            log.info("🖼️   Tiled display triggered.")
        except Exception as exc:
            log.warning(f"Stitch trigger failed: {exc}")

    # ── API health check ─────────────────────────────────────────────────────

    def _check_api(self) -> None:
        url = f"http://{API_IP}:{API_PORT}/status"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                st = resp.json()
                log.info(f"✅  API OK — model={st.get('model')}")
        except Exception as exc:
            log.warning(f"API check failed ({exc}) — continuing anyway.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Task2Runner().start()
