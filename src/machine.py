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

    node = PiNode(pc_host="192.168.1.50")
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

import logging
import threading
from abc import ABC, abstractmethod

from data_transfer import VideoSender, FeedbackReceiver, FeedbackConfig

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
    pc_host : str
        IP address (or resolvable hostname) of the PC running the
        processor (``processor.py``).
    camera_device : str
        V4L2 device node for the camera.  Default ``"/dev/video0"``.
    bind_host : str
        Network interface to bind the display receive socket to.
        Defaults to ``"0.0.0.0"`` (all interfaces).  Set to the Pi's own
        WLAN IP to restrict display updates to the local subnet.

    Raises
    ------
    RuntimeError
        From ``run()`` if the node is already running.

    Example
    -------
    >>> node = PiNode(pc_host="192.168.1.50", bind_host="192.168.1.100")
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
        pc_host: str,
        camera_device: str = "/dev/video0",
        bind_host: str = "0.0.0.0",
    ) -> None:
        self._pc_host = pc_host
        self._camera_device = camera_device

        self._video_sender = VideoSender(
            pc_host=pc_host,
            device=camera_device,
        )
        self._feedback_receiver = FeedbackReceiver(
            bind_host=bind_host,
            display_callback=self._on_display_frame,
        )

        self._video_thread: threading.Thread | None = None
        self._feedback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
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
            logger.info("PiNode: video uplink active → %s", self._pc_host)
            self._stop_event.wait()  # Block until stop() is called.
        except Exception:
            logger.exception("PiNode: video uplink encountered a fatal error")
        finally:
            self._video_sender.stop()
            logger.info("PiNode: video uplink stopped")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _on_display_frame(self, frame_bytes: bytes) -> None:
        """
        Callback invoked by ``FeedbackReceiver`` for each incoming display
        framebuffer (``FeedbackConfig.DISPLAY_BYTES`` = 1024 bytes).

        **Placeholder implementation** — logs receipt and stores the frame.
        Replace (or subclass and override) to drive a real SSD1306 OLED,
        for example::

            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            from PIL import Image

            serial = i2c(port=1, address=0x3C)
            device = ssd1306(serial)
            img = Image.frombytes('1', (128, 64), frame_bytes)
            device.display(img)

        Parameters
        ----------
        frame_bytes : bytes
            Exactly ``FeedbackConfig.DISPLAY_BYTES`` (1024) bytes of
            packed 1-bit monochrome pixel data, row-major, MSB-first.
        """
        logger.debug("PiNode: display frame received (%d bytes)", len(frame_bytes))
        # TODO: replace with real SSD1306 driver call (see docstring above).

    # ------------------------------------------------------------------
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
        logger.info("PiNode running (PC host: %s)", self._pc_host)

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


# ---------------------------------------------------------------------------
# TODO — next steps
# ---------------------------------------------------------------------------
# 1. [PiNode._on_display_frame]  Wire in the luma.oled SSD1306 driver so
#    that incoming framebuffers are rendered on the physical OLED panel.
# 2. [PiNode]  Add a health-check / watchdog: if stream_video or
#    process_feedback exits unexpectedly, restart the affected thread.
# 3. [PiNode]  Expose a simple status API (e.g. is_streaming property)
#    for monitoring from the main application.
# 4. [VideoSender]  Test the v4l2src pipeline with the Pi Camera Module v3
#    (libcamera stack); may require the ``libcamerasrc`` GStreamer element
#    instead of ``v4l2src``.
# 5. [General]  Create a systemd service file that launches PiNode on boot
#    and restarts it on failure.
# ---------------------------------------------------------------------------
