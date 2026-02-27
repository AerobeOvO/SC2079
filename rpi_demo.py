#!/usr/bin/env python3
"""
rpi_demo.py  –  Android → RPi → STM32 Manual-Control Bridge
=============================================================

DATA FLOW
─────────
  Android tablet
      │  Bluetooth RFCOMM (plain text)
      ▼
  Raspberry Pi  (this script)
      │  Serial UART  (plain text command)
      ▼
  STM32 board  →  executes motion
      │  Serial UART  ("ACK")
      ▼
  Raspberry Pi  →  relays ACK back to Android

ANDROID → RPI  –  plain-text commands (newline-terminated)
────────────────────────────────────────────────────────────
  f      → FWD:10   (forward  10 cm)
  r      → REV:10   (reverse  10 cm)
  tl     → TURNL:90 (turn left  90°)
  tr     → TURNR:90 (turn right 90°)
  s      → STOP     (stop motors)

  You may also send the full STM32 command directly:
  FWD:20 / REV:5 / TURNL:45 / TURNR:45 / STOP / OLED:Hi

RPI → ANDROID  –  plain-text replies (newline-terminated)
───────────────────────────────────────────────────────────
  "Connected to RPi! Ready to receive commands."
  "Sent to STM32: FWD:10"
  "ACK received for: FWD:10"
  "ERROR: Unknown command: XYZ"
  "ERROR: STM32 did not ACK: FWD:10"

STM32 COMMAND FORMAT  (rpi.py style, sent verbatim over serial)
───────────────────────────────────────────────────────────────
  FWD:<dist>    Move forward  (dist in cm, e.g. FWD:10)
  REV:<dist>    Move backward (dist in cm, e.g. REV:20)
  TURNL:<deg>   Turn left     (degrees,    e.g. TURNL:90)
  TURNR:<deg>   Turn right    (degrees,    e.g. TURNR:90)
  STOP          Stop all motors immediately
  OLED:<msg>    Display message on the STM32 OLED screen

USAGE (run on Raspberry Pi)
───────────────────────────
  cd /path/to/SC2079
  python3 Main/rpi_demo.py

Press Ctrl+C to quit cleanly.
"""

import bluetooth
import logging
import os
import queue
import serial
import sys
import threading
import time
from typing import Optional

# ─── Make project root importable (for settings.py etc.) ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from settings import SERIAL_PORT, BAUD_RATE

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rpi_demo.log"),
    ],
)
log = logging.getLogger("rpi_demo")

# ─── Bluetooth configuration ──────────────────────────────────────────────────
BT_SERVICE_NAME = "MDP-RPi-Demo"
BT_UUID         = "94f39d29-7d6d-437d-973b-fba39e49d4ee"   # same UUID as Week_9

# ─── STM32 timing ─────────────────────────────────────────────────────────────
STM_ACK_TIMEOUT = 15.0      # seconds to wait for ACK before flagging a warning

# ─── Short-code → STM32 command mapping (Android plain-text buttons) ─────────
#
#   Android sends  │  RPi forwards to STM32
#   ───────────────┼──────────────────────
#      f           │  FWD:10
#      r           │  REV:10
#      tl          │  TURNL:90
#      tr          │  TURNR:90
#      s           │  STOP
#
#   Adjust the default values (10 cm / 90°) to match your robot's tuning.
#
SHORTCODE_MAP = {
    "f":  "FWD:10",
    "r":  "REV:10",
    "tl": "TURNL:90",
    "tr": "TURNR:90",
    "s":  "STOP",
}

# Full-command prefixes that are forwarded verbatim (case-insensitive)
VALID_PREFIXES = (
    "FWD:",
    "REV:",
    "TURNL:",
    "TURNR:",
    "STOP",
    "OLED:",
)


def parse_android_command(raw: str) -> Optional[str]:
    """
    Accept either a short-code (f / r / tl / tr / s) or a full STM32 command
    (FWD:10, TURNL:90 …).  Returns the STM32 command string, or None if unknown.
    """
    token = raw.strip().lower()

    # Short-code lookup first
    if token in SHORTCODE_MAP:
        return SHORTCODE_MAP[token]

    # Full command — check against known prefixes
    upper = token.upper()
    if any(upper.startswith(p) for p in VALID_PREFIXES):
        return upper   # normalise to upper-case before sending

    return None   # unrecognised


# ══════════════════════════════════════════════════════════════════════════════
# STM32 SERIAL LINK
# ══════════════════════════════════════════════════════════════════════════════
class STMSerial:
    """Thin wrapper around pyserial for talking to the STM32."""

    def __init__(self):
        self._ser: Optional[serial.Serial] = None

    # ── Connect ───────────────────────────────────────────────────────────────
    def connect(self) -> None:
        log.info(f"Opening serial port  {SERIAL_PORT}  @ {BAUD_RATE} baud …")
        self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=STM_ACK_TIMEOUT)
        time.sleep(2)                    # let the STM32 finish its reset
        self._ser.reset_input_buffer()
        log.info("✅  STM32 serial connected")

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, cmd: str) -> None:
        """Send a command string followed by CR+LF."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("STM32 serial port is not open")
        payload = (cmd + "\r\n").encode("utf-8")
        log.debug(f"  →STM32  {repr(cmd)}")
        self._ser.write(payload)
        self._ser.flush()

    # ── Receive ───────────────────────────────────────────────────────────────
    def recv(self) -> Optional[str]:
        """
        Block until a line arrives (or STM_ACK_TIMEOUT elapses).
        Returns the decoded string, or None on timeout / empty line.
        """
        if self._ser is None or not self._ser.is_open:
            return None
        raw = self._ser.readline()
        if raw:
            decoded = raw.strip().decode("utf-8", errors="ignore")
            log.debug(f"  ←STM32  {repr(decoded)}")
            return decoded
        return None

    # ── Disconnect ────────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            log.info("STM32 serial closed")


# ══════════════════════════════════════════════════════════════════════════════
# ANDROID BLUETOOTH LINK
# ══════════════════════════════════════════════════════════════════════════════
class AndroidBT:
    """Bluetooth RFCOMM server that waits for the Android tablet to connect."""

    def __init__(self):
        self._server: Optional[bluetooth.BluetoothSocket] = None
        self._client: Optional[bluetooth.BluetoothSocket] = None
        self._buf = ""    # partial-line buffer for incoming data

    # ── Connect ───────────────────────────────────────────────────────────────
    def connect(self) -> None:
        log.info("Setting RPi to be Bluetooth-discoverable …")
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
        log.info(f"     Service name : {BT_SERVICE_NAME}")
        log.info(f"     UUID         : {BT_UUID}")

        self._client, info = self._server.accept()
        log.info(f"✅  Android connected  {info}")

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, text: str) -> None:
        """Send a plain-text line to Android, newline-terminated."""
        msg = text.rstrip("\n") + "\n"
        try:
            self._client.send(msg.encode("utf-8"))
            log.debug(f"  →Android  {msg.strip()!r}")
        except OSError as exc:
            log.error(f"BT send error: {exc}")
            raise

    # ── Receive ───────────────────────────────────────────────────────────────
    def recv(self) -> Optional[str]:
        """
        Return the next non-empty line received from Android (plain text).
        Buffers internally to handle partial / multi-line TCP reads.
        Raises OSError on connection loss.
        """
        while True:
            # If there's already a complete line in the buffer, use it
            if "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                log.debug(f"  ←Android  {line!r}")
                return line

            # Read more bytes from the socket
            chunk = self._client.recv(1024)
            if not chunk:
                raise OSError("Android socket closed (recv returned empty)")
            self._buf += chunk.decode("utf-8", errors="ignore")

    # ── Disconnect ────────────────────────────────────────────────────────────
    def close(self) -> None:
        try:
            if self._client:
                self._client.close()
            if self._server:
                self._server.close()
            log.info("Bluetooth closed")
        except Exception as exc:
            log.error(f"BT close error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class DemoController:
    """
    Orchestrates two threads:
      • _thread_recv_android  – reads plain-text from Android, maps to STM32 command, enqueues
      • _thread_send_stm      – dequeues commands, sends to STM32, waits for ACK
    """

    def __init__(self):
        self.android   = AndroidBT()
        self.stm       = STMSerial()
        self.cmd_queue: "queue.Queue[str]" = queue.Queue()
        self._running  = False

    # ── Start ─────────────────────────────────────────────────────────────────
    def start(self) -> None:
        log.info("=" * 58)
        log.info("  RPi Demo  –  Android  →  RPi  →  STM32")
        log.info("=" * 58)

        # 1. Open serial to STM32
        self.stm.connect()

        # 2. Wait for Android Bluetooth connection
        self.android.connect()

        # 3. Welcome message
        self.android.send("Connected to RPi! Ready to receive commands.")

        self._running = True

        # 4. Spawn worker threads
        t_recv = threading.Thread(target=self._thread_recv_android, daemon=True, name="recv_android")
        t_send = threading.Thread(target=self._thread_send_stm,     daemon=True, name="send_stm")
        t_recv.start()
        t_send.start()

        log.info("Threads started. Listening for commands from Android …\n")

        # 5. Main thread just waits so Ctrl+C works cleanly
        try:
            t_recv.join()   # recv thread exits when Android disconnects
        except KeyboardInterrupt:
            log.info("\nCtrl+C received – shutting down …")
        finally:
            self._running = False
            self.android.close()
            self.stm.close()
            log.info("Shutdown complete.")

    # ── Thread: Receive from Android ──────────────────────────────────────────
    def _thread_recv_android(self) -> None:
        """
        Continuously reads plain-text lines from the Android tablet.

        Accepted inputs (case-insensitive)
        ────────────────────────────────────
          f      → FWD:10   (forward)
          r      → REV:10   (reverse)
          tl     → TURNL:90 (turn left)
          tr     → TURNR:90 (turn right)
          s      → STOP

          Full commands also accepted verbatim:
          FWD:20 / REV:5 / TURNL:45 / TURNR:45 / STOP / OLED:Hi
        """
        while self._running:
            try:
                raw = self.android.recv()
            except OSError:
                log.error("Android disconnected – stopping recv thread")
                self._running = False
                break

            if raw is None:
                continue

            cmd = parse_android_command(raw)

            if cmd is not None:
                log.info(f"📥  Android → {raw!r}  maps to  {cmd}")
                self.cmd_queue.put(cmd)
            else:
                log.warning(f"⚠️  Unrecognised input from Android: {raw!r}")
                self._safe_send_android(f"ERROR: Unknown command: {raw}")

    # ── Thread: Send to STM32 and wait for ACK ────────────────────────────────
    def _thread_send_stm(self) -> None:
        """
        Dequeues commands one by one, sends them to STM32 over serial,
        then waits for the ACK.  Sends the result back to Android.
        """
        while self._running:
            try:
                cmd = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            log.info(f"🚗  RPi → STM32:  {cmd}")
            try:
                # Inform Android that the command is being sent
                self._safe_send_android(f"Sent to STM32: {cmd}")

                # Send command to STM32
                self.stm.send(cmd)

                # Wait for ACK from STM32 (blocking, up to STM_ACK_TIMEOUT s)
                ack = self.stm.recv()

                if ack and ack.upper().startswith("ACK"):
                    log.info(f"✅  STM32 ACK for: {cmd}")
                    self._safe_send_android(f"ACK received for: {cmd}")
                else:
                    # No ACK within timeout – warn but keep going
                    log.warning(f"⚠️  No ACK (got {ack!r}) for: {cmd}")
                    self._safe_send_android(f"ERROR: STM32 did not ACK: {cmd}")

            except Exception as exc:
                log.error(f"Error while sending {cmd!r} to STM32: {exc}")
                self._safe_send_android(f"ERROR: Serial error for {cmd}: {exc}")

    # ── Helper: send to Android without crashing on disconnect ───────────────
    def _safe_send_android(self, text: str) -> None:
        try:
            self.android.send(text)
        except OSError:
            log.warning("Could not send to Android (connection lost?)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = DemoController()
    demo.start()
