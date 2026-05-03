"""
data_transfer.py
================
Shared data-transfer primitives used by **both** sides of the OJDM system:

* **Raspberry Pi Zero 2 W** (``machine.py``) — sender of camera video,
  receiver of audio/display feedback.
* **PC processing node** (``processor.py``) — receiver of camera video,
  sender of audio/display feedback.

All GStreamer pipelines are built as strings and parsed at runtime, so they
can be tuned in one place without touching application logic.  The GLib main
loop is run on a dedicated daemon thread; user code only calls
``start()`` / ``stop()``.

"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoConfig:
    """
    Video transport config (Pi → PC) using TCP + multipart JPEG.

    NOTE: TCP bc i havent figured out how to use UDP. tehe:3
    Pi:
      libcamerasrc ! ... ! jpegenc ! multipartmux ! tcpserversink host=0.0.0.0 port=PORT

    PC:
      tcpclientsrc host=PI_HOST port=PORT ! multipartdemux ! jpegdec ! ... ! appsink

    Attributes
    ----------
    WIDTH : int
        Capture width in pixels.  Must match the camera's supported resolution.
    HEIGHT : int
        Capture height in pixels.
    FPS : int
        Target frames per second.
    PORT : int
        Destination port on the PC that ``VideoReceiver`` will listen on.
    """
    width: int = 640
    height: int = 480
    fps: int = 10
    port: int = 5000

    raw_format: str = "NV12"
    colorimetry: str = "bt709"


# TODO: check feedback pipeline FIRST, then implement later
class FeedbackConfig:
    """
    Parameters for the feedback downlink (PC → Pi).

    Attributes
    ----------
    DISPLAY_WIDTH : int
        Width of the SSD1306-compatible OLED panel in pixels.
    DISPLAY_HEIGHT : int
        Height of the OLED panel in pixels.
    DISPLAY_BPP : int
        Bits per pixel — 1 for monochrome (black/white only).
    DISPLAY_BYTES : int
        Byte count of one raw framebuffer snapshot
        (``DISPLAY_WIDTH × DISPLAY_HEIGHT / 8 = 1024``).
    AUDIO_FORMAT : str
        Audio codec identifier understood by both sides (``"opus"``).
    AUDIO_PORT : int
        UDP destination port on the Pi that ``FeedbackReceiver`` listens on
        for RTP/Opus audio.
    DISPLAY_PORT : int
        UDP destination port on the Pi that ``FeedbackReceiver`` listens on
        for raw display-framebuffer datagrams.
    """

    # Display is a 128x64 monochrome (1-bit) screen.
    DISPLAY_WIDTH: int = 128
    DISPLAY_HEIGHT: int = 64
    DISPLAY_BPP: int = 1  # 1-bit black/white
    DISPLAY_BYTES: int = 128 * 64 // 8  # 1024 bytes per frame

    # Audio feedback is an Opus-encoded file/segment.
    AUDIO_FORMAT: str = "opus"
    AUDIO_PORT: int = 5001
    DISPLAY_PORT: int = 5002


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class VideoTransferer(ABC):
    """
    Minimal contract for video transport components.

    Rules
    -----
    - `start()` / `stop()` must be safe to call from any thread.
    - Implementations own their GStreamer pipeline lifecycle.
    """

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...

class FeedbackTransferer(ABC):
    """
    Abstract base for feedback downlink components (PC → Pi).

    Notes
    -----
    Implementations must be safe to call ``start()`` / ``stop()`` from any
    thread.
    """

    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Common GStreamer runner mixin (shared implementation detail)
# ---------------------------------------------------------------------------

class _GstRunner:
    def __init__(self) -> None:
        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop) -> bool:
        if msg.type == Gst.MessageType.EOS:
            logger.info("%s: pipeline EOS", self.__class__.__name__)
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("%s pipeline error: %s — %s", self.__class__.__name__, err, debug)
            loop.quit()
        return True

    def _start_pipeline(self, pipeline_str: str) -> None:
        with self._lock:
            if self._pipeline is not None:
                return  # already started

            logger.debug("%s pipeline: %s", self.__class__.__name__, pipeline_str)
            pipeline = Gst.parse_launch(pipeline_str)
            if pipeline is None:
                raise RuntimeError(f"{self.__class__.__name__}: Gst.parse_launch() returned None")

            loop = GLib.MainLoop()
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message, loop)

            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                raise RuntimeError(f"{self.__class__.__name__}: pipeline failed to reach PLAYING state")

            t = threading.Thread(target=loop.run, name=f"{self.__class__.__name__}-loop", daemon=True)
            t.start()

            self._pipeline = pipeline
            self._loop = loop
            self._loop_thread = t

    def _stop_pipeline(self) -> None:
        with self._lock:
            if self._loop and self._loop.is_running():
                self._loop.quit()

            if self._pipeline:
                self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None

            if self._loop_thread:
                self._loop_thread.join(timeout=5)
                self._loop_thread = None

            self._loop = None


# ---------------------------------------------------------------------------
# VideoSender (Pi): TCP server, multipart JPEG
# ---------------------------------------------------------------------------

class VideoSender(_GstRunner, VideoTransferer):
    """
    Pi-side sender: camera → JPEG → multipartmux → tcpserversink.

    Parameters
    ----------
    cfg : VideoConfig
        Video caps + TCP port.
    bind_host : str
        Usually "0.0.0.0" to listen on all interfaces.
    """

    def __init__(self, cfg: VideoConfig = VideoConfig(), bind_host: str = "0.0.0.0") -> None:
        super().__init__()
        self._cfg = cfg
        self._bind_host = bind_host

    def _build_pipeline_str(self) -> str:
        c = self._cfg
        return (
            "libcamerasrc ! "
            f"video/x-raw,colorimetry={c.colorimetry},format={c.raw_format},"
            f"width={c.width},height={c.height},framerate={c.fps}/1 ! "
            "jpegenc ! multipartmux ! "
            f"tcpserversink host={self._bind_host} port={c.port}"
        )

    def start(self) -> None:
        self._start_pipeline(self._build_pipeline_str())
        logger.info("VideoSender started (TCP server) on %s:%d", self._bind_host, self._cfg.port)

    def stop(self) -> None:
        self._stop_pipeline()
        logger.info("VideoSender stopped")


# ---------------------------------------------------------------------------
# VideoReceiver (PC): TCP client → demux/dec → appsink (NumPy frames)
# ---------------------------------------------------------------------------

class VideoReceiver(_GstRunner, VideoTransferer):
    """
    PC-side receiver: tcpclientsrc → multipartdemux → jpegdec → videoconvert → appsink.

    Parameters
    ----------
    pi_host : str
        IP/hostname of the Pi running VideoSender.
    cfg : VideoConfig
        Must match sender's port.
    frame_callback : callable, optional
        Called with each decoded BGR frame (H, W, 3) uint8.
    """

    def __init__(
        self,
        pi_host: str,
        cfg: VideoConfig = VideoConfig(),
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        super().__init__()
        self._pi_host = pi_host
        self._cfg = cfg
        self._frame_callback = frame_callback

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def _build_pipeline_str(self) -> str:
        c = self._cfg
        # appsink produces BGR frames for OpenCV-style consumers.
        return (
            f"tcpclientsrc host={self._pi_host} port={c.port} ! "
            "multipartdemux ! jpegdec ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )

    def _on_new_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        sample: Gst.Sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf: Gst.Buffer = sample.get_buffer()
        caps_struct: Gst.Structure = sample.get_caps().get_structure(0)
        width: int = caps_struct.get_value("width")
        height: int = caps_struct.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(mapinfo)

        with self._frame_lock:
            self._latest_frame = frame

        if self._frame_callback is not None:
            try:
                self._frame_callback(frame)
            except Exception:
                logger.exception("VideoReceiver: frame_callback raised an exception")

        return Gst.FlowReturn.OK

    def start(self) -> None:
        pipeline_str = self._build_pipeline_str()
        self._start_pipeline(pipeline_str)

        # Connect appsink after pipeline is created.
        # [Inference] In practice, get_by_name("sink") is available immediately after parse_launch.
        with self._lock:
            if not self._pipeline:
                raise RuntimeError("VideoReceiver: internal error: pipeline missing after start")

            sink: Gst.Element = self._pipeline.get_by_name("sink")
            sink.connect("new-sample", self._on_new_sample)

        logger.info("VideoReceiver started (TCP client) connecting to %s:%d", self._pi_host, self._cfg.port)

    def stop(self) -> None:
        self._stop_pipeline()
        logger.info("VideoReceiver stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

# TODO: FeedbackTransferer
