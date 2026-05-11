"""
processor.py
============
PC-side processing pipeline for the Orange Juice Detection Machine (OJDM).

Features
--------
* Receives frames from the Pi via VideoReceiver.
* Runs YOLO detection via Detector.
* Sends random Opus audio clips from:
    - assets/sounds/detected/*.opus (when detection present)
    - assets/sounds/idle/*.opus (periodically when idle)
* Sends OLED display frames based on state:
    - assets/faces/love (when detection present)
    - assets/faces/idle (when idle)
* Optional debug GUI (Tkinter) with detection boxes.

Expected asset layout
---------------------
assets/
  sounds/
    idle/*.opus
    detected/*.opus
  faces/
    idle        (1024 bytes raw framebuffer)
    love        (1024 bytes raw framebuffer)

Usage
-----
python3 processor.py --pi-host 192.168.1.50 --model ./OJ_model_v1.pt --debug-gui

"""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from data_transfer import VideoConfig, FeedbackSender, VideoReceiver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISPLAY_BYTES = 128 * 64 // 8  # 1024
BASE_DIR = Path(__file__).resolve().parent  # ../src
ASSETS_DIR = BASE_DIR.parent / "assets"


@dataclass
class Detection:
    """
    A single object detection result produced by the YOLO model.

    Attributes
    ----------
    class_id : int
        Integer class index as defined in the model's ``names`` mapping.
    class_name : str
        Human-readable class label (e.g. ``"orange_juice"``).
    confidence : float
        Detection confidence score in the range ``[0, 1]``.
    bbox : tuple[int, int, int, int]
        Bounding box in pixel coordinates ``(x1, y1, x2, y2)`` where
        ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the
        bottom-right corner of the detected object.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class ActionSet:
    """
    The set of feedback actions decided by a :class:`Decider`.

    Attributes
    ----------
    play_audio : bool
        Whether to play an audio alert on the Pi's speaker.
    audio_label : str
        Short label identifying *which* audio clip to play
        (e.g. ``"alert"``).  The caller maps this to an actual Opus file.
    update_display : bool
        Whether to push a new image to the Pi's OLED display.
    display_frame : bytes or None
        1024-byte raw framebuffer to send, or ``None`` when
        ``update_display`` is ``False``.
    detections : list[Detection]
        The detections that triggered this action set (for logging/debug).
    """

    play_audio: bool = False
    audio_label: str = ""
    update_display: bool = False
    display_frame: Optional[bytes] = None
    detections: list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class IDetector(ABC):
    """
    Interface for detecting objects in a single video frame.

    Implementors hold any stateful resources (e.g. a loaded model) at
    construction time; each call to ``detect()`` is otherwise stateless.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run object detection on a single BGR video frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image of shape ``(H, W, 3)`` and dtype ``uint8``, as
            delivered by ``VideoReceiver.get_frame()``.

        Returns
        -------
        list[Detection]
            Possibly-empty list of detections found in the frame,
            ordered by confidence (highest first).
        """

# ---------------------------------------------------------------------------
# Concrete: Detector
# ---------------------------------------------------------------------------

class Detector(IDetector):
    """
    YOLO-based object detector backed by the Ultralytics library.

    Parameters
    ----------
    model_path : str
        Path to the ``.pt`` weights file, e.g.
        ``"runs/detect/train/weights/best.pt"``.
    confidence_threshold : float
        Minimum confidence score required to include a detection.
        Detections below this threshold are silently discarded.
        Default is ``0.5``.
    device : str
        Torch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        Use ``"cpu"`` for PC setups without a GPU.

    Raises
    ------
    ImportError
        If the ``ultralytics`` package is not installed.

    Example
    -------
    >>> detector = Detector("runs/detect/train/weights/best.pt")
    >>> detections = detector.detect(frame)
    >>> for det in detections:
    ...     print(det.class_name, f"{det.confidence:.2f}", det.bbox)

    Notes
    -----
    * The YOLO model is loaded once at construction time and reused for
      every ``detect()`` call.
    * A threading lock protects the model during inference so that
      ``detect()`` is safe to call from multiple threads (e.g. when frames
      arrive faster than they can be processed).
    * ``detect()`` always returns an empty list rather than raising when a
      frame cannot be processed; errors are logged at ERROR level.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc

        self._model = YOLO(model_path)
        self._conf = confidence_threshold
        self._device = device
        self._lock = threading.Lock()

        logger.info(
            "Detector loaded model '%s' on device '%s' (conf ≥ %.2f)",
            model_path, device, confidence_threshold,
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLO inference on *frame* and return filtered detections.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        list[Detection]
            Detections with confidence ≥ ``confidence_threshold``,
            sorted by confidence descending.  Returns ``[]`` on error.
        """
        try:
            with self._lock:
                results = self._model(
                    frame,
                    conf=self._conf,
                    device=self._device,
                    verbose=False,
                )
        except Exception:
            logger.exception("Detector.detect() raised an unexpected error")
            return []

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=result.names[cls_id],
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# Frame buffer (thread-safe)
# ---------------------------------------------------------------------------

class FrameBuffer:
    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()


# ---------------------------------------------------------------------------
# Asset loading helpers
# ---------------------------------------------------------------------------

@dataclass
class AudioPools:
    idle: list[Path]
    detected: list[Path]

    def random_idle(self) -> Optional[Path]:
        return random.choice(self.idle) if self.idle else None

    def random_detected(self) -> Optional[Path]:
        return random.choice(self.detected) if self.detected else None


@dataclass
class FaceFrames:
    idle: bytes
    love: bytes


def _load_audio_pools(base: Path) -> AudioPools:
    idle_dir = base / "idle"
    detected_dir = base / "detected"

    idle = sorted(p for p in idle_dir.glob("*.opus") if p.is_file())
    detected = sorted(p for p in detected_dir.glob("*.opus") if p.is_file())

    logger.info("Loaded %d idle audio clips, %d detected clips", len(idle), len(detected))
    return AudioPools(idle=idle, detected=detected)

def _bitstring_to_display_bytes(bit_string: str) -> bytes:
    cleaned = "".join(ch for ch in bit_string.strip() if ch in ("0", "1"))
    expected_bits = DISPLAY_BYTES * 8  # 8192
    if len(cleaned) != expected_bits:
        raise ValueError(
            f"Display bit string must contain exactly {expected_bits} bits, got {len(cleaned)}"
        )

    out = bytearray(DISPLAY_BYTES)
    for byte_idx, i in enumerate(range(0, len(cleaned), 8)):
        out[byte_idx] = int(cleaned[i:i + 8], 2)
    return bytes(out)


#TODO: more logic on face thingies
def _load_face_frames(faces_dir: Path) -> FaceFrames:
    idle_path = faces_dir / "idle"
    love_path = faces_dir / "love"

    idle_raw = idle_path.read_bytes()
    love_raw = love_path.read_bytes()

    if len(idle_raw) == DISPLAY_BYTES and len(love_raw) == DISPLAY_BYTES:
        return FaceFrames(idle=idle_raw, love=love_raw)

    # Treat as ASCII bitstrings
    idle_text = idle_raw.decode("utf-8")
    love_text = love_raw.decode("utf-8")
    idle = _bitstring_to_display_bytes(idle_text)
    love = _bitstring_to_display_bytes(love_text)
    return FaceFrames(idle=idle, love=love)


# ---------------------------------------------------------------------------
# Debug GUI (optional)
# ---------------------------------------------------------------------------

class DebugGUI:
    def __init__(self, frame_buffer: FrameBuffer, detector: Detector) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk, ImageDraw, ImageFont

        self._tk = tk
        self._ttk = ttk
        self._Image = Image
        self._ImageTk = ImageTk
        self._ImageDraw = ImageDraw
        self._ImageFont = ImageFont

        self._frame_buffer = frame_buffer
        self._detector = detector
        self._running = True

        self.root = tk.Tk()
        self.root.title("Detector Debug")
        self.root.geometry("900x700")

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(control_frame, text="Waiting for frames...")
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self._on_stop)
        self._poll_and_display()

    def _poll_and_display(self) -> None:
        frame = self._frame_buffer.get()
        if frame is not None:
            detections = self._detector.detect(frame)

            rgb = frame[..., ::-1]
            img = self._Image.fromarray(rgb)
            draw = self._ImageDraw.Draw(img)

            try:
                font = self._ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = self._ImageFont.load_default()

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                label = f"{det.class_name} {det.confidence:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                text_box = draw.textbbox((0, 0), label, font=font)
                tw = text_box[2] - text_box[0]
                th = text_box[3] - text_box[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="red")
                draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)

            img.thumbnail((900, 700), self._Image.Resampling.LANCZOS)
            photo = self._ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
            self.status_label.config(text=f"Detections: {len(detections)}")

        if self._running:
            self.root.after(33, self._poll_and_display)

    def _on_stop(self) -> None:
        self._running = False
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

class Processor:
    def __init__(
        self,
        pi_host: str,
        model_path: Path,
        confidence_threshold: float,
        device: str,
        video_cfg: VideoConfig,
        audio_pools: AudioPools,
        face_frames: FaceFrames,
        idle_interval_seconds: float,
        debug_gui: bool,
    ) -> None:
        self._pi_host = pi_host
        self._video_cfg = video_cfg
        self._idle_interval = idle_interval_seconds

        self._audio_pools = audio_pools
        self._faces = face_frames

        self._frame_buffer = FrameBuffer()
        self._detector = Detector(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            device=device,
        )

        self._receiver = VideoReceiver(
            pi_host=self._pi_host,
            cfg=self._video_cfg,
            frame_callback=self._frame_buffer.update,
        )
        self._feedback = FeedbackSender(pi_host=self._pi_host)

        self._last_idle_audio_time = 0.0
        self._last_state_detected = False
        self._debug_gui_enabled = debug_gui
        self._debug_gui: Optional[DebugGUI] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._receiver.start()
        self._feedback.start()

    def stop(self) -> None:
        self._receiver.stop()
        self._feedback.stop()

    def _send_detected_feedback(self) -> None:
        audio = self._audio_pools.random_detected()
        if audio is not None:
            self._feedback.send_audio(audio)
        self._feedback.send_display(self._faces.love)

    def _send_idle_feedback(self, now: float) -> None:
        if now - self._last_idle_audio_time >= self._idle_interval:
            audio = self._audio_pools.random_idle()
            if audio is not None:
                self._feedback.send_audio(audio)
            self._last_idle_audio_time = now

        if self._last_state_detected:
            self._feedback.send_display(self._faces.idle)

    def run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._frame_buffer.get()
                if frame is None:
                    time.sleep(0.01)
                    continue

                detections = self._detector.detect(frame)
                detected = len(detections) > 0

                now = time.monotonic()
                if detected:
                    if not self._last_state_detected:
                        self._send_detected_feedback()
                    self._last_state_detected = True
                else:
                    self._send_idle_feedback(now)
                    self._last_state_detected = False

                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Processor interrupted by user")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="PC-side processor for OJDM")
    parser.add_argument("--pi-host", required=True, help="Pi hostname or IP")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda/mps)")
    parser.add_argument("--video-port", type=int, default=5000, help="Video TCP port")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--idle-interval", type=float, default=40.0, help="Idle audio interval (s)")
    parser.add_argument("--assets", default="assets", help="Assets base directory")
    parser.add_argument("--debug-gui", action="store_true", help="Show debug GUI with boxes")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    assets = Path(args.assets)
    audio_pools = _load_audio_pools(ASSETS_DIR / "sounds")
    face_frames = _load_face_frames(ASSETS_DIR / "faces")

    video_cfg = VideoConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.video_port,
    )

    processor = Processor(
        pi_host=args.pi_host,
        model_path=Path(args.model),
        confidence_threshold=args.conf,
        device=args.device,
        video_cfg=video_cfg,
        audio_pools=audio_pools,
        face_frames=face_frames,
        idle_interval_seconds=args.idle_interval,
        debug_gui=args.debug_gui,
    )

    processor.start()

    if args.debug_gui:
        worker = threading.Thread(target=processor.run_loop, daemon=True)
        worker.start()

        gui = DebugGUI(processor._frame_buffer, processor._detector)
        try:
            gui.run()  # MAIN THREAD
        finally:
            processor.stop()
    else:
        try:
            processor.run_loop()
        finally:
            processor.stop()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
