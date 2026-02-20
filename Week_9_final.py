#!/usr/bin/env python3
import json
import queue
import time
from multiprocessing import Process, Manager
from typing import Optional
import os
import requests
from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from consts import SYMBOL_MAP
from logger import prepare_logger
from settings import API_IP, API_PORT


class PiAction:
    def __init__(self, cat, value):
        self._cat = cat
        self._value = value

    @property
    def cat(self):
        return self._cat

    @property
    def value(self):
        return self._value


class RaspberryPi:
    def __init__(self):
        # Initialize logger and communication objects with Android and STM
        self.logger = prepare_logger()
        self.android_link = AndroidLink()
        self.stm_link = STMLink()

        # For sharing information between child processes
        self.manager = Manager()

        # Set robot mode to be 1 (Path mode)
        self.robot_mode = self.manager.Value('i', 1)

        # Events
        self.android_dropped = self.manager.Event()  # Set when the android link drops
        # commands will be retrieved from commands queue when this event is set
        self.unpause = self.manager.Event()

        # Movement Lock
        self.movement_lock = self.manager.Lock()

        # Queues
        self.android_queue = self.manager.Queue() # Messages to send to Android
        self.rpi_action_queue = self.manager.Queue() # Messages that need to be processed by RPi
        self.command_queue = self.manager.Queue() # Messages that need to be processed by STM32, as well as snap commands

        # Define empty processes
        self.proc_recv_android = None
        self.proc_recv_stm32 = None
        self.proc_android_sender = None
        self.proc_command_follower = None
        self.proc_rpi_action = None

        # Counters shared across processes
        self.ack_count = self.manager.Value('i', 0)
        self.clk_count = self.manager.Value('i', 0)  # Track number of CLK commands received

    def start(self):
        """Starts the RPi orchestrator"""
        try:
            # Establish Bluetooth connection with Android
            self.android_link.connect()
            self.android_queue.put(AndroidMessage('info', 'You are connected to the RPi!'))

            # Establish connection with STM32
            self.stm_link.connect()

            # Check Image Recognition and Algorithm API status
            self.check_api()
            
            #self.small_direction = self.snap_and_rec("Small")
            #self.logger.info(f"PREINFER small direction is: {self.small_direction}")

            # Define child processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_recv_stm32 = Process(target=self.recv_stm)
            self.proc_android_sender = Process(target=self.android_sender)
            self.proc_command_follower = Process(target=self.command_follower)
            self.proc_rpi_action = Process(target=self.rpi_action)

            # Start child processes
            self.proc_recv_android.start()
            self.proc_recv_stm32.start()
            self.proc_android_sender.start()
            self.proc_command_follower.start()
            self.proc_rpi_action.start()

            self.logger.info("Child Processes started")

            ### Start up complete ###

            # Send success message to Android
            self.android_queue.put(AndroidMessage('info', 'Robot is ready!'))
            self.android_queue.put(AndroidMessage('mode', 'path' if self.robot_mode.value == 1 else 'manual'))
            
            
            
            # Handover control to the Reconnect Handler to watch over Android connection
            self.reconnect_android()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops all processes on the RPi and disconnects gracefully with Android and STM32"""
        self.android_link.disconnect()
        self.stm_link.disconnect()
        self.logger.info("Program exited!")

    def reconnect_android(self):
        """Handles the reconnection to Android in the event of a lost connection."""
        self.logger.info("Reconnection handler is watching...")

        while True:
            # Wait for android connection to drop
            self.android_dropped.wait()

            self.logger.error("Android link is down!")

            # Kill child processes
            self.logger.debug("Killing android child processes")
            self.proc_android_sender.kill()
            self.proc_recv_android.kill()

            # Wait for the child processes to finish
            self.proc_android_sender.join()
            self.proc_recv_android.join()
            assert self.proc_android_sender.is_alive() is False
            assert self.proc_recv_android.is_alive() is False
            self.logger.debug("Android child processes killed")

            # Clean up old sockets
            self.android_link.disconnect()

            # Reconnect
            self.android_link.connect()

            # Recreate Android processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_android_sender = Process(target=self.android_sender)

            # Start previously killed processes
            self.proc_recv_android.start()
            self.proc_android_sender.start()

            self.logger.info("Android child processes restarted")
            self.android_queue.put(AndroidMessage("info", "You are reconnected!"))
            self.android_queue.put(AndroidMessage('mode', 'path' if self.robot_mode.value == 1 else 'manual'))

            self.android_dropped.clear()
            
        
    def recv_android(self) -> None:
        """
        [Child Process] Processes the messages received from Android
        """
       
        while True:
            msg_str: Optional[str] = None
            try:
                msg_str = self.android_link.recv()
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android connection dropped")

            # If an error occurred in recv()
            if msg_str is None:
                continue

            # Handle multiple JSON objects concatenated in a single message
            messages = []
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(msg_str):
                try:
                    message, end_idx = decoder.raw_decode(msg_str, idx)
                    messages.append(message)
                    idx = end_idx
                    # Skip whitespace between JSON objects
                    while idx < len(msg_str) and msg_str[idx].isspace():
                        idx += 1
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON at position {idx}: {e}")
                    break

            # Process each message
            for message in messages:
                ## Command: Start Moving ##
                if message['cat'] == "control":
                    if message['value'] == "start":
            
                        if not self.check_api():
                            self.logger.error("API is down! Start command aborted.")

                        self.clear_queues()
                        
                        # Reset counters
                        self.ack_count.value = 0
                        self.clk_count.value = 0
                        
                        # Send RS00 to start the car moving
                        self.command_queue.put("RS00") # ack_count = 1
                        
                        self.logger.info("Start command received, starting robot on Week 9 task!")
                        self.logger.info("Car will start moving, waiting for CLK command to capture images...")
                        self.android_queue.put(AndroidMessage('status', 'running'))

                        # Commencing path following | Main trigger to start movement #
                        self.unpause.set()
                    
    def recv_stm(self) -> None:
        """
        [Child Process] Receive acknowledgement messages from STM32, and release the movement lock
        """
        while True:

            message: str = self.stm_link.recv()
            
            # Acknowledgement from STM32
            if message.startswith("ACK"):

                self.ack_count.value += 1

                # Release movement lock
                try:
                    self.movement_lock.release()
                except Exception:
                    self.logger.warning("Tried to release a released lock!")

                self.logger.debug(f"ACK from STM32 received, ACK count now:{self.ack_count.value}")
                self.logger.info(f"self.ack_count: {self.ack_count.value}")
                
                # Check if task is complete (after both obstacles handled)
                if self.ack_count.value == 6:
                    self.logger.debug("Task complete - both obstacles handled!")
                    self.android_queue.put(AndroidMessage("status", "finished"))
                    self.command_queue.put("FIN")
            
            # CLK command from STM32 to capture image
            elif message.startswith("CLK"):
                self.clk_count.value += 1
                self.logger.info(f"CLK command received from STM32: {message} (CLK #{self.clk_count.value})")
                
                # Extract obstacle ID from CLK command (e.g., CLK01, CLK_Small, CLK_Large, etc.)
                obstacle_id = message[3:] if len(message) > 3 else "Obstacle"
                
                # Determine obstacle type based on CLK count
                # First CLK = Small obstacle, Second CLK = Large obstacle
                is_small_obstacle = (self.clk_count.value == 1)
                obstacle_type = "Small" if is_small_obstacle else "Large"
                
                self.logger.info(f"CLK: Processing {obstacle_type} obstacle")
                
                # Capture image and get direction
                direction = self.snap_and_rec(f"{obstacle_type}_{obstacle_id}")
                self.logger.info(f"CLK: Detected direction: {direction}")
                
                # Send turn command based on detected direction and obstacle type
                if direction == "Left Arrow":
                    if is_small_obstacle:
                        self.command_queue.put("TL00")
                        self.logger.debug("CLK: Queueing TL00 for small obstacle left turn")
                    else:
                        self.command_queue.put("PL01")
                        self.logger.debug("CLK: Queueing PL01 for large obstacle left turn")
                elif direction == "Right Arrow":
                    if is_small_obstacle:
                        self.command_queue.put("TR00")
                        self.logger.debug("CLK: Queueing TR00 for small obstacle right turn")
                    else:
                        self.command_queue.put("PR01")
                        self.logger.debug("CLK: Queueing PR01 for large obstacle right turn")
                else:
                    # Default to left/right based on obstacle type if detection failed
                    if is_small_obstacle:
                        self.command_queue.put("TL00")
                        self.logger.warning("CLK: Detection failed for small obstacle, defaulting to TL00")
                    else:
                        self.command_queue.put("PR01")
                        self.logger.warning("CLK: Detection failed for large obstacle, defaulting to PR01")
            
            else:
                self.logger.warning(
                    f"Ignored unknown message from STM: {message}")

    def android_sender(self) -> None:
        while True:
            try:
                message: AndroidMessage = self.android_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.android_link.send(message)
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android dropped")

    def command_follower(self) -> None:
        while True:
            command: str = self.command_queue.get()
            self.unpause.wait()
            self.movement_lock.acquire()
            stm32_prefixes = ("STOP", "ZZ", "TL", "TR", "PL", "PR", "RS", "OB")
            if command.startswith(stm32_prefixes):
                self.stm_link.send(command)
            elif command == "FIN":
                self.unpause.clear()
                self.movement_lock.release()
                self.logger.info("Commands queue finished.")
                self.android_queue.put(AndroidMessage("info", "Commands queue finished."))
                self.android_queue.put(AndroidMessage("status", "finished"))
                self.rpi_action_queue.put(PiAction(cat="stitch", value=""))
            else:
                raise Exception(f"Unknown command: {command}")

    def rpi_action(self):
        while True:
            action: PiAction = self.rpi_action_queue.get()
            self.logger.debug(f"PiAction retrieved from queue: {action.cat} {action.value}")
            if action.cat == "snap": self.snap_and_rec(obstacle_id=action.value)
            elif action.cat == "stitch": self.request_stitch()

    def _capture_image_picamera(self, filename: str, attempt_index: int) -> None:
        """Capture an image using the legacy picamera module with retry-based exposure tweaks."""
        try:
            import picamera
        except ImportError:
            self.logger.error("picamera module not found. Please install 'picamera' on the Raspberry Pi.")
            raise

        if attempt_index <= 3:
            shutter_us = min(1000000, 10000 * (2 ** (attempt_index - 1)))
            ev_comp = min(6, 2 * attempt_index)
            iso_value = 200
        else:
            factor = max(1, 2 ** (attempt_index - 3))
            shutter_us = max(1000, 10000 // factor)
            ev_comp = -min(6, 2 * (attempt_index - 3))
            iso_value = 100

        with picamera.PiCamera() as camera:
            camera.resolution = (1280, 960)
            camera.framerate = 30
            camera.iso = iso_value
            camera.exposure_mode = 'auto'
            camera.awb_mode = 'auto'
            camera.exposure_compensation = ev_comp
            camera.brightness = 50
            camera.contrast = 0
            camera.saturation = 0
            camera.sharpness = 0

            time.sleep(0.4)

            camera.shutter_speed = shutter_us
            time.sleep(0.1)

            camera.capture(filename, format='jpeg', quality=85)

    def snap_and_rec(self, obstacle_id: str) -> Optional[str]:
        """Capture an image, call the image recognition API, and return the interpreted symbol."""

        self.logger.info(f"Capturing image for obstacle id: {obstacle_id}")
        self.android_queue.put(AndroidMessage("info", f"Capturing image for obstacle id: {obstacle_id}"))

        url = f"http://{API_IP}:{API_PORT}/image"
        filename = f"{int(time.time())}_{obstacle_id}_C.jpg"

        retry_count = 0
        results = None

        while True:
            retry_count += 1

            try:
                self._capture_image_picamera(filename, retry_count)
            except Exception as exc:
                self.logger.error(f"Failed to capture image with picamera: {exc}")
                return None

            self.logger.debug("Requesting from image API")

            try:
                with open(filename, 'rb') as image_file:
                    response = requests.post(url, files={"file": (filename, image_file)})
            except Exception as exc:
                self.logger.error(f"Exception when calling image-rec API: {exc}")
                return None

            if response.status_code != 200:
                self.logger.error("Something went wrong when requesting path from image-rec API. Please try again.")
                return None

            try:
                results = json.loads(response.content)
            except json.JSONDecodeError as exc:
                self.logger.error(f"Invalid JSON received from image-rec API: {exc}")
                return None

            # Check if we got valid results (either old format with image_id or new format with segments)
            has_valid_result = False
            if results.get('image_id') and results.get('image_id') != 'NA':
                has_valid_result = True
            elif results.get('segments') and len(results.get('segments', [])) > 0:
                has_valid_result = True
            
            if has_valid_result or retry_count > 6:
                break
            elif retry_count > 3:
                self.logger.info(f"Image recognition results: {results}")
                self.logger.info("Recapturing with lower exposure...")
            else:
                self.logger.info(f"Image recognition results: {results}")
                self.logger.info("Recapturing with higher exposure...")

        if results is None:
            return None

        # Handle new API format with segments
        symbol = None
        if results.get('segments') and len(results.get('segments', [])) > 0:
            # Get the segment with highest confidence
            segments = results.get('segments', [])
            best_segment = max(segments, key=lambda s: s.get('confidence', 0))
            
            class_name = best_segment.get('class_name', '')
            class_id = best_segment.get('class_id')
            
            # Try to get symbol from class_name directly (e.g., "Arrow_Left" -> "Left Arrow")
            if 'Arrow_Left' in class_name or class_name == 'Arrow_Left':
                symbol = "Left Arrow"
            elif 'Arrow_Right' in class_name or class_name == 'Arrow_Right':
                symbol = "Right Arrow"
            elif 'Arrow_Up' in class_name or class_name == 'Arrow_Up':
                symbol = "Up Arrow"
            elif 'Arrow_Down' in class_name or class_name == 'Arrow_Down':
                symbol = "Down Arrow"
            elif class_id is not None:
                # Fallback to using class_id with SYMBOL_MAP
                symbol = SYMBOL_MAP.get(str(class_id))
        else:
            # Handle old API format with image_id
            symbol = SYMBOL_MAP.get(results.get('image_id'))
        
        self.logger.info(f"Image recognition results: {results} ({symbol})")
        self.android_queue.put(AndroidMessage("image-rec", results))

        return symbol

    def request_stitch(self):
        url = f"http://{API_IP}:{API_PORT}/stitch"
        response = requests.get(url)
        if response.status_code != 200:
            self.logger.error("Something went wrong when requesting stitch from the API.")
            return
        self.logger.info("Images stitched!")

    def clear_queues(self):
        while not self.command_queue.empty():
            self.command_queue.get()

    def check_api(self) -> bool:
        url = f"http://{API_IP}:{API_PORT}/status"
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                self.logger.debug("API is up!")
                return True
        except ConnectionError:
            self.logger.warning("API Connection Error")
            return False
        except requests.Timeout:
            self.logger.warning("API Timeout")
            return False
        except Exception as e:
            self.logger.warning(f"API Exception: {e}")
            return False

if __name__ == "__main__":
    rpi = RaspberryPi()
    rpi.start()
