"""
machine.py
==========
Raspberry Pi Zero 2 W node implementation for the OJDM project.

The Pi node has two concurrent responsibilities managed by daemon threads:

1. **Video uplink** (Thread 1, ``stream_video``) — continuously capture
   frames from the Pi camera and stream them to the PC via
   ``VideoSender``.
2. **Feedback downlink** (Thread 2, ``process_feedback``) — listen for
   audio and display data from the PC and drive the speaker and OLED
   display via ``FeedbackReceiver``.

Typical usage
-------------
::

    node = PiNode()
    node.run()        # starts both threads (non-blocking)
    try:
        signal.pause()    # keep the main thread alive
    except KeyboardInterrupt:
        node.stop()       # clean shutdown

Dependencies (Pi-side only)
---------------------------
* ``python3-gi`` + GStreamer 1.0 plugins (``gstreamer1.0-plugins-*``).
* ``luma.oled`` (or equivalent) for writing to the SSD1306 OLED panel;
  wire it into ``PiNode._on_display_frame()`` — see docstring.
* A V4L2-compatible camera accessible at ``/dev/video0``.
"""

from __future__ import annotations
import sys
import argparse
import signal
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, ssd1325, ssd1331, sh1106
from time import sleep

import logging
import threading
from abc import ABC, abstractmethod

from data_transfer import VideoConfig, FeedbackConfig, VideoSender, FeedbackReceiver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class IPiNode(ABC):
    """
    Abstract interface for the Raspberry Pi Zero 2 W node.

    Concrete implementations must provide two long-running methods intended
    to run on separate threads: video capture/streaming and feedback
    reception.
    """

    @abstractmethod
    def stream_video(self) -> None:
        """Thread 1: Capture and stream video frames to the PC."""

    @abstractmethod
    def process_feedback(self) -> None:
        """Thread 2: Receive audio and display feedback from the PC."""


# ---------------------------------------------------------------------------
# Concrete: PiNode
# ---------------------------------------------------------------------------

class PiNode(IPiNode):
    """
    Concrete Raspberry Pi Zero 2 W node.

    Manages a :class:`~data_transfer.VideoSender` for the camera uplink
    and a :class:`~data_transfer.FeedbackReceiver` for the audio/display
    downlink.  Both channels run on independent daemon threads so neither
    blocks the other.

    Parameters
    ----------
    video_cfg : VideoConfig
        Video uplink configuration (resolution, FPS, TCP port).
    feedback_cfg : FeedbackConfig
        Feedback downlink configuration (audio/display ports).
    video_bind_host : str
        TCP bind host for the video server (default: "0.0.0.0").
    display_bind_host : str
        UDP bind host for display feedback (default: "0.0.0.0").

    Raises
    ------
    RuntimeError
        From ``run()`` if the node is already running.

    Example
    -------
    >>> node = PiNode(
    ...     video_cfg=VideoConfig(width=640, height=480, fps=10, port=5000),
    ...     feedback_cfg=FeedbackConfig(),
    ...     video_bind_host="0.0.0.0",
    ...     display_bind_host="0.0.0.0",
    ... )
    >>> node.run()          # starts both threads, non-blocking
    >>> # … application logic or signal.pause() …
    >>> node.stop()         # graceful shutdown, joins both threads

    Notes
    -----
    * ``stream_video()`` and ``process_feedback()`` are the thread
      *targets*; call ``run()`` to launch them — do not call them
      directly.
    * OLED display updates arrive via ``_on_display_frame()`` callback.
      Replace the placeholder body with a real SSD1306 driver call (see
      the inline docstring).
    * The stop event is used as the blocking mechanism inside each thread
      target so that ``stop()`` can signal both threads simultaneously
      and then join them.
    """

    def __init__(
        self,
        video_cfg: VideoConfig = VideoConfig(),
        feedback_cfg: FeedbackConfig = FeedbackConfig(),
        video_bind_host: str = "0.0.0.0",
        display_bind_host: str = "0.0.0.0",
    ) -> None:
        self._video_cfg = video_cfg
        self._feedback_cfg = feedback_cfg
        self._video_bind_host = video_bind_host
        self._display_bind_host = display_bind_host

        self._video_sender = VideoSender(
            cfg=self._video_cfg,
            bind_host=self._video_bind_host,
        )
        self._feedback_receiver = FeedbackReceiver(
            cfg=self._feedback_cfg,
            bind_host=self._display_bind_host,
            display_callback=self._on_display_frame,
        )

        self._video_thread: threading.Thread | None = None
        self._feedback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._oled_device = None

    def stream_video(self) -> None:
        """
        Thread-1 target: start the video uplink and block until stopped.

        Calls ``VideoSender.start()`` to initialise the GStreamer pipeline,
        then blocks on the stop event.  When ``stop()`` is called,
        ``VideoSender.stop()`` is invoked to tear down the pipeline before
        this method returns.

        Expected behaviour
        ------------------
        * Runs indefinitely after ``run()`` is called.
        * Returns (thread exits) only when ``stop()`` sets the stop event.
        * Any fatal GStreamer error is logged; the thread exits cleanly
          rather than crashing the whole process.
        """
        try:
            self._video_sender.start()
            logger.info(
                "PiNode: video uplink active on %s:%d",
                self._video_bind_host,
                self._video_cfg.port,
            )
            self._stop_event.wait()  # Block until stop() is called.
        except Exception:
            logger.exception("PiNode: video uplink encountered a fatal error")
        finally:
            self._video_sender.stop()
            logger.info("PiNode: video uplink stopped")

    def process_feedback(self) -> None:
        """
        Thread-2 target: start the feedback downlink and block until stopped.

        Calls ``FeedbackReceiver.start()`` to initialise the GStreamer audio
        pipeline and the display UDP socket, then blocks on the stop event.
        When ``stop()`` is called, ``FeedbackReceiver.stop()`` tears down
        both channels before this method returns.

        Expected behaviour
        ------------------
        * Audio plays automatically as data arrives from the PC.
        * OLED is updated via the ``_on_display_frame`` callback.
        * Returns only when ``stop()`` sets the stop event.
        """
        try:
            self._feedback_receiver.start()
            logger.info("PiNode: feedback downlink active")
            self._stop_event.wait()
        except Exception:
            logger.exception("PiNode: feedback downlink encountered a fatal error")
        finally:
            self._feedback_receiver.stop()
            logger.info("PiNode: feedback downlink stopped")

    def _frame_bytes_to_points(self, frame_bytes: bytes) -> list[tuple[int, int]]:
        """
        Convert 1024 bytes of packed 1-bit monochrome pixel data to a list
        of (x, y) coordinates where pixels are ON.

        The input frame is 128×64 pixels, packed row-major with MSB-first
        bit ordering within each byte. Bit value 1 = pixel ON.

        Parameters
        ----------
        frame_bytes : bytes
            Exactly 1024 bytes of packed 1-bit monochrome pixel data.

        Returns
        -------
        list[tuple[int, int]]
            List of (x, y) coordinates where pixels are ON.
            0 ≤ x ≤ 127, 0 ≤ y ≤ 63.
        """
        points = []

        for byte_index, byte_val in enumerate(frame_bytes):
            # Calculate row (y) and column position within row (x_base)
            # Each row has 128 pixels = 16 bytes
            y = byte_index // 16
            x_base = (byte_index % 16) * 8

            # Check each bit in the byte (MSB-first)
            for bit_pos in range(8):
                if byte_val & (0x80 >> bit_pos):  # MSB-first: check bit 7, 6, 5, ...
                    x = x_base + bit_pos
                    points.append((x, y))

        return points

    def _on_display_frame(self, frame_bytes: bytes) -> None:
        """
        Callback invoked by ``FeedbackReceiver`` for each incoming display
        framebuffer (``FeedbackConfig.DISPLAY_BYTES`` = 1024 bytes).

        Parameters
        ----------
        frame_bytes : bytes
            Exactly ``FeedbackConfig.DISPLAY_BYTES`` (1024) bytes of
            packed 1-bit monochrome pixel data, row-major, MSB-first.
        """
        if len(frame_bytes) != FeedbackConfig.DISPLAY_BYTES:
            logger.warning(
                "PiNode: display frame size mismatch (%d bytes)", len(frame_bytes)
            )
            return

        try:
            if self._oled_device is None:
                serial = i2c(port=1, address=0x3C)
                self._oled_device = ssd1306(serial, rotate=2)

            points = self._frame_bytes_to_points(frame_bytes)
            with canvas(self._oled_device) as draw:
                draw.point(points, fill="white")
        except Exception:
            logger.exception("PiNode: failed to render display frame")

    def run(self) -> None:
        """
        Launch the video-uplink and feedback-downlink threads.

        Returns immediately after starting both threads; the threads run
        as daemons so the process can exit without explicitly calling
        ``stop()``.  For a clean shutdown, call ``stop()``.

        Raises
        ------
        RuntimeError
            If the node is already running (``stop()`` must be called
            first).
        """
        if self._video_thread and self._video_thread.is_alive():
            raise RuntimeError("PiNode is already running; call stop() first")

        self._stop_event.clear()

        self._video_thread = threading.Thread(
            target=self.stream_video,
            name="PiNode-video",
            daemon=True,
        )
        self._feedback_thread = threading.Thread(
            target=self.process_feedback,
            name="PiNode-feedback",
            daemon=True,
        )

        self._video_thread.start()
        self._feedback_thread.start()
        logger.info(
            "PiNode running (video %s:%d, display %s:%d)",
            self._video_bind_host,
            self._video_cfg.port,
            self._display_bind_host,
            self._feedback_cfg.DISPLAY_PORT,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Signal both threads to stop and wait for them to exit.

        Sets the internal stop event (causing ``stream_video`` and
        ``process_feedback`` to unblock from their ``wait()`` calls), then
        joins both threads with a 5-second timeout each.  Logs a warning
        if a thread does not exit within the timeout.
        """
        logger.info("PiNode: stopping …")
        self._stop_event.set()
        if self._oled_device is not None:
            self._oled_device.clear

        for thread, name in [
            (self._video_thread, "video"),
            (self._feedback_thread, "feedback"),
        ]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(
                        "PiNode: %s thread did not exit within timeout", name
                    )


        logger.info("PiNode stopped")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PiNode (video uplink + feedback downlink)"
    )
    parser.add_argument(
        "--video-host",
        default="0.0.0.0",
        help="TCP bind host for VideoSender (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--video-port",
        type=int,
        default=5000,
        help="TCP port for VideoSender (default: 5000)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video FPS (default: 10)",
    )
    parser.add_argument(
        "--display-host",
        default="0.0.0.0",
        help="UDP bind host for display feedback (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--audio-port",
        type=int,
        default=5001,
        help="UDP port for audio feedback (default: 5001)",
    )
    parser.add_argument(
        "--display-port",
        type=int,
        default=5002,
        help="UDP port for display feedback (default: 5002)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    video_cfg = VideoConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.video_port,
    )
    feedback_cfg = FeedbackConfig()
    feedback_cfg.AUDIO_PORT = args.audio_port
    feedback_cfg.DISPLAY_PORT = args.display_port

    node = PiNode(
        video_cfg=video_cfg,
        feedback_cfg=feedback_cfg,
        video_bind_host=args.video_host,
        display_bind_host=args.display_host,
    )

    node.run()

    try:
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        node.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

