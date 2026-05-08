"""
processor.py
============
PC-side processing pipeline for the Orange Juice Detection Machine (OJDM).

Two main components
-------------------
* **Detector** — wraps an Ultralytics YOLO model and runs inference on
  incoming BGR video frames received from the Pi.
* **Decider** — interprets detection results and decides which feedback
  actions to trigger (audio alert, OLED display update, etc.).

Typical data flow on the PC side
---------------------------------
::

    VideoReceiver.frame_callback
           │
           ▼
    Detector.detect(frame)         → list[Detection]
           │
           ▼
    Decider.evaluate(detections)   → ActionSet
           │
           ├──► FeedbackSender.send_audio(opus_bytes)   (if play_audio)
           └──► FeedbackSender.send_display(frame_bytes) (if update_display)
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

# Byte count of the 128×64 1-bit OLED framebuffer used by Decider.
_DISPLAY_BYTES: int = 128 * 64 // 8  # 1024


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


class IDecider(ABC):
    """
    Interface for deciding which feedback actions to perform based on
    detection results.
    """

    @abstractmethod
    def evaluate(self, detections: list[Detection]) -> ActionSet:
        """
        Determine what actions to take given the current detections.

        Parameters
        ----------
        detections : list[Detection]
            Output from ``IDetector.detect()``.

        Returns
        -------
        ActionSet
            The actions that should be executed this cycle.
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
# Concrete: Decider
# ---------------------------------------------------------------------------

class Decider(IDecider):
    """
    Rule-based action decider for orange-juice detections.

    Decision logic
    --------------
    * If any detection matches ``target_class`` **and** its confidence
      is ≥ ``alert_confidence``, ``ActionSet.play_audio`` is set to
      ``True`` with ``audio_label = "alert"``.
    * ``ActionSet.update_display`` is ``True`` whenever there are any
      detections at all, so the OLED reflects the live detection state.
    * The display framebuffer is a minimal 128×64 1-bit image whose filled
      rows are proportional to the number of target-class detections
      (see ``_build_display_frame()``).

    Parameters
    ----------
    target_class : str
        The detection class name to monitor (e.g. ``"orange_juice"``).
    alert_confidence : float
        Minimum confidence required to trigger an audio alert.
        Must be in the range ``[0, 1]``.  Default is ``0.6``.

    Example
    -------
    >>> decider = Decider(target_class="orange_juice")
    >>> actions = decider.evaluate(detections)
    >>> if actions.play_audio:
    ...     feedback_sender.send_audio(audio_clips["alert"])
    >>> if actions.update_display:
    ...     feedback_sender.send_display(actions.display_frame)
    """

    def __init__(
        self,
        target_class: str = "orange_juice",
        alert_confidence: float = 0.6,
    ) -> None:
        self._target_class = target_class
        self._alert_confidence = alert_confidence

    # ------------------------------------------------------------------
    def evaluate(self, detections: list[Detection]) -> ActionSet:
        """
        Map a list of detections to a concrete set of feedback actions.

        Parameters
        ----------
        detections : list[Detection]
            Output from ``Detector.detect()``.

        Returns
        -------
        ActionSet
            * ``play_audio`` is ``True`` iff at least one detection has
              ``class_name == target_class`` and
              ``confidence >= alert_confidence``.
            * ``update_display`` is ``True`` iff ``detections`` is
              non-empty.
            * ``display_frame`` contains the 1024-byte framebuffer when
              ``update_display`` is ``True``, otherwise ``None``.
        """
        actions = ActionSet(detections=detections)

        target_hits = [
            d for d in detections
            if d.class_name == self._target_class
            and d.confidence >= self._alert_confidence
        ]

        if target_hits:
            actions.play_audio = True
            actions.audio_label = "alert"
            logger.info(
                "Decider: %d high-confidence '%s' detection(s) → audio alert",
                len(target_hits), self._target_class,
            )

        if detections:
            actions.update_display = True
            actions.display_frame = self._build_display_frame(detections)

        return actions

    # ------------------------------------------------------------------
    def _build_display_frame(self, detections: list[Detection]) -> bytes:
        """
        Build a 1024-byte 128×64 monochrome framebuffer summarising the
        current detections.

        The framebuffer uses a simple bar-chart metaphor: for each target-
        class detection, 8 additional rows at the top of the display are
        filled white (max 8 detections × 8 rows = 64 rows = full display).
        All other pixels are black.

        Parameters
        ----------
        detections : list[Detection]
            Current detections (any class).

        Returns
        -------
        bytes
            Exactly ``_DISPLAY_BYTES`` (1024) bytes of packed 1-bit pixel
            data, row-major, MSB-first within each byte.

        Notes
        -----
        This is a placeholder visualisation.  A proper bitmap font
        renderer (see TODO list) should replace or augment this once a
        font library is integrated.
        """
        # 128 columns × 64 rows; each byte encodes 8 horizontal pixels.
        buf = bytearray(_DISPLAY_BYTES)  # all pixels off (black)

        target_count = sum(
            1 for d in detections if d.class_name == self._target_class
        )

        # Fill top N rows (8 rows per detected target, capped at 64 rows).
        fill_rows = min(target_count * 8, 64)
        for row in range(fill_rows):
            byte_offset = row * (128 // 8)  # 16 bytes per row
            for col_byte in range(128 // 8):
                buf[byte_offset + col_byte] = 0xFF  # all 8 pixels on

        return bytes(buf)


# ---------------------------------------------------------------------------
# TODO — next steps
# ---------------------------------------------------------------------------
# 1. [Detector]  Collect orange-juice images, annotate bounding boxes, and
#    train/fine-tune a YOLO model (replace the placeholder in detector.py).
# 2. [Detector]  Benchmark inference time on the PC; if GPU is available,
#    switch device to "cuda" for faster throughput.
# 3. [Decider]   Integrate a bitmap font library (e.g. Pillow) so that
#    _build_display_frame() renders readable text (count, confidence %).
# 4. [Decider]   Load audio clips (Opus format) from a config file rather
#    than hard-coding the "alert" label.
# 5. [General]   Implement a main PC loop that wires VideoReceiver,
#    Detector, Decider, and FeedbackSender together in a single process.
# 6. [General]   Add logging configuration (rotating file handler) for
#    production deployment.
# ---------------------------------------------------------------------------
