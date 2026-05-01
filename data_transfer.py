"""
data_transfer.py
================
Shared data-transfer primitives used by **both** sides of the OJDM system:

* **Raspberry Pi Zero 2 W** (``machine.py``) — sender of camera video,
  receiver of audio/display feedback.
* **PC processing node** (``processor.py``) — receiver of camera video,
  sender of audio/display feedback.

Architecture overview
---------------------
::

    ┌────────────────────────┐             ┌──────────────────────────┐
    │  Raspberry Pi Zero 2 W │             │  PC (processor.py)       │
    │                        │             │                          │
    │  VideoSender           │──RTP/H.264─►│  VideoReceiver           │
    │  FeedbackReceiver      │◄──RTP/Opus──│  FeedbackSender          │
    │  (audio + OLED)        │◄──UDP raw───│  (audio + display bytes) │
    └────────────────────────┘             └──────────────────────────┘

Transport
---------
* **Video uplink**     : RTP/UDP, H.264 encoded, port ``VideoConfig.PORT``
* **Audio downlink**   : RTP/UDP, Opus encoded, port ``FeedbackConfig.AUDIO_PORT``
* **Display downlink** : Raw UDP datagram (1024 bytes/frame),
  port ``FeedbackConfig.DISPLAY_PORT``

All GStreamer pipelines are built as strings and parsed at runtime, so they
can be tuned in one place without touching application logic.  The GLib main
loop is run on a dedicated daemon thread; user code only calls
``start()`` / ``stop()``.

Usage example (Pi side)
-----------------------
>>> sender   = VideoSender(pc_host="192.168.1.50")
>>> receiver = FeedbackReceiver(display_callback=oled_driver.update)
>>> sender.start()
>>> receiver.start()
>>> ...
>>> sender.stop()
>>> receiver.stop()

Usage example (PC side)
-----------------------
>>> receiver = VideoReceiver(frame_callback=detector.detect)
>>> sender   = FeedbackSender(pi_host="192.168.1.100")
>>> receiver.start()
>>> sender.start()
>>> ...
>>> sender.send_audio(opus_bytes)
>>> sender.send_display(framebuffer_bytes)
>>> ...
>>> receiver.stop()
>>> sender.stop()
"""

from __future__ import annotations

import logging
import socket
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialise GStreamer once at import time.
Gst.init(None)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------------

class VideoConfig:
    """
    Parameters for the camera video uplink (Pi → PC).

    Attributes
    ----------
    WIDTH : int
        Capture width in pixels.  Must match the camera's supported resolution.
    HEIGHT : int
        Capture height in pixels.
    FPS : int
        Target frames per second.  Pi Zero 2 W typically manages 10–15 fps
        with software H.264 encoding at 640×480.
    PORT : int
        UDP destination port on the PC that ``VideoReceiver`` will listen on.
    """

    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 10
    PORT: int = 5000


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
# Abstract base classes  (public interface contract)
# ---------------------------------------------------------------------------

class VideoTransferer(ABC):
    """
    Abstract base for video uplink components (Pi → PC).

    Both the sending side (Pi, ``VideoSender``) and the receiving side
    (PC, ``VideoReceiver``) share this interface, so the rest of the
    codebase depends only on the abstraction rather than a specific
    GStreamer implementation.

    Notes
    -----
    Concrete subclasses must be safe to call ``start()`` / ``stop()`` from
    any thread.
    """

    CONFIG: VideoConfig = VideoConfig()

    @abstractmethod
    def start(self) -> None:
        """Start GStreamer pipeline(s) and any background threads."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Transition pipeline to NULL state and join background threads."""
        ...


class FeedbackTransferer(ABC):
    """
    Abstract base for feedback downlink components (PC → Pi).

    Covers both the audio channel (Opus over RTP/UDP) and the display
    channel (1024-byte raw UDP datagram per frame).

    Notes
    -----
    Implementations must be safe to call ``start()`` / ``stop()`` from any
    thread.
    """

    CONFIG: FeedbackConfig = FeedbackConfig()

    @abstractmethod
    def start(self) -> None:
        """Start pipeline(s) and/or background threads."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Shut down pipeline(s) and background threads cleanly."""
        ...


# ---------------------------------------------------------------------------
# VideoSender  (runs on the Pi)
# ---------------------------------------------------------------------------

class VideoSender(VideoTransferer):
    """
    Captures video from the Pi camera and streams it to the PC via H.264/RTP.

    GStreamer pipeline
    ------------------
    ::

        v4l2src → videoconvert → x264enc (ultrafast / zerolatency)
                → rtph264pay  → udpsink  (PC_HOST:VideoConfig.PORT)

    Parameters
    ----------
    pc_host : str
        IP address (or resolvable hostname) of the PC running
        ``VideoReceiver``.
    device : str
        V4L2 device node of the camera, e.g. ``"/dev/video0"``.

    Raises
    ------
    RuntimeError
        From ``start()`` if the GStreamer pipeline cannot be parsed or
        transitions to PLAYING state fail.

    Example
    -------
    >>> sender = VideoSender(pc_host="192.168.1.50")
    >>> sender.start()
    >>> # … camera is streaming …
    >>> sender.stop()

    Notes
    -----
    * H.264 is software-encoded on the Pi Zero 2 W.  The ``ultrafast``
      speed-preset and ``zerolatency`` tune keep CPU usage manageable at
      640×480 / 10 fps.
    * The pipeline string is built by ``_build_pipeline_str()`` so that
      operators can swap elements (e.g. ``omxh264enc`` on a Pi 4) without
      modifying other code.
    * A daemon GLib main-loop thread handles all GStreamer bus messages;
      ``start()`` and ``stop()`` are the only public entry points.
    """

    def __init__(self, pc_host: str, device: str = "/dev/video0") -> None:
        self._pc_host = pc_host
        self._device = device
        self._cfg = self.CONFIG

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def _build_pipeline_str(self) -> str:
        """
        Build the GStreamer launch-string for the video uplink.

        Returns
        -------
        str
            A ``gst-launch``-compatible pipeline description that captures
            from the V4L2 device, encodes to H.264, wraps in RTP, and
            sends over UDP to the PC.
        """
        return (
            f"v4l2src device={self._device} ! "
            f"video/x-raw,width={self._cfg.WIDTH},"
            f"height={self._cfg.HEIGHT},"
            f"framerate={self._cfg.FPS}/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency speed-preset=ultrafast "
            f"key-int-max={self._cfg.FPS * 2} ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self._pc_host} port={self._cfg.PORT}"
        )

    # ------------------------------------------------------------------
    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop
    ) -> bool:
        """
        Handle GStreamer bus messages (EOS and ERROR).

        Stops the GLib main-loop on any terminal event so that ``stop()``
        remains consistent with pipeline state.

        Parameters
        ----------
        bus : Gst.Bus
            The pipeline bus (injected by GStreamer's signal machinery).
        msg : Gst.Message
            The incoming bus message.
        loop : GLib.MainLoop
            The event loop to quit on terminal events.

        Returns
        -------
        bool
            ``True`` to keep the signal handler registered.
        """
        if msg.type == Gst.MessageType.EOS:
            logger.info("VideoSender: pipeline EOS")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("VideoSender pipeline error: %s — %s", err, debug)
            loop.quit()
        return True

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Build and start the GStreamer capture/send pipeline.

        Spins up a daemon thread that runs the GLib event loop to keep
        the pipeline alive until ``stop()`` is called.

        Raises
        ------
        RuntimeError
            If GStreamer fails to parse the pipeline string or if the
            pipeline cannot transition to PLAYING state.
        """
        pipeline_str = self._build_pipeline_str()
        logger.debug("VideoSender pipeline: %s", pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        if self._pipeline is None:
            raise RuntimeError("VideoSender: Gst.parse_launch() returned None")

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self._loop)

        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("VideoSender: pipeline failed to reach PLAYING state")

        self._loop_thread = threading.Thread(
            target=self._loop.run, name="VideoSender-loop", daemon=True
        )
        self._loop_thread.start()
        logger.info("VideoSender started → %s:%d", self._pc_host, self._cfg.PORT)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Stop the capture/send pipeline and release all GStreamer resources.

        Safe to call even when ``start()`` has not been called.  Blocks
        until the GLib event-loop thread has exited (up to 5 seconds).
        """
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
        logger.info("VideoSender stopped")


# ---------------------------------------------------------------------------
# VideoReceiver  (runs on the PC)
# ---------------------------------------------------------------------------

class VideoReceiver(VideoTransferer):
    """
    Receives the H.264/RTP video stream from the Pi and exposes decoded
    frames to application code.

    GStreamer pipeline
    ------------------
    ::

        udpsrc → rtph264depay → avdec_h264 → videoconvert
               → video/x-raw,format=BGR → appsink

    Parameters
    ----------
    listen_port : int
        UDP port to bind.  Must match ``VideoConfig.PORT`` used by the Pi.
        Defaults to ``VideoConfig.PORT``.
    frame_callback : callable, optional
        If provided, called with each new ``numpy.ndarray`` frame
        (shape ``(H, W, 3)``, dtype ``uint8``, BGR channel order) as it
        arrives from the pipeline.  Useful for feeding frames directly
        into a ``Detector`` without polling.

    Raises
    ------
    RuntimeError
        From ``start()`` if the pipeline cannot be parsed or started.

    Example
    -------
    >>> recv = VideoReceiver(frame_callback=detector.detect)
    >>> recv.start()
    >>> # frames flow into detector.detect() automatically
    >>> recv.stop()

    Notes
    -----
    * ``appsink`` is configured with ``max-buffers=1`` and ``drop=True``
      to prevent unbounded memory growth when the detector is slower than
      the incoming stream rate.
    * ``get_frame()`` is thread-safe and always returns the **most
      recently decoded** frame, or ``None`` if no frame has arrived yet.
    """

    def __init__(
        self,
        listen_port: int = VideoConfig.PORT,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._port = listen_port
        self._frame_callback = frame_callback
        self._cfg = self.CONFIG

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _build_pipeline_str(self) -> str:
        """
        Build the GStreamer receive/decode pipeline string.

        Returns
        -------
        str
            Pipeline description that binds a UDP socket and produces
            BGR-formatted frames at the appsink.
        """
        return (
            f"udpsrc port={self._port} "
            "caps=\"application/x-rtp,media=video,clock-rate=90000,"
            "encoding-name=H264,payload=96\" ! "
            "rtph264depay ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )

    # ------------------------------------------------------------------
    def _on_new_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        GStreamer ``new-sample`` callback: convert a decoded buffer to NumPy.

        Called by the GStreamer runtime each time the appsink has a fully
        decoded video frame available.  The frame is stored as the latest
        frame and, if a ``frame_callback`` was provided, forwarded to it.

        Parameters
        ----------
        appsink : Gst.Element
            The appsink element that emitted the signal.

        Returns
        -------
        Gst.FlowReturn
            ``GST_FLOW_OK`` on success; ``GST_FLOW_ERROR`` if the sample
            or its buffer cannot be mapped.
        """
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
            # Copy to avoid holding a reference to the GStreamer buffer.
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                (height, width, 3)
            ).copy()
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

    # ------------------------------------------------------------------
    def _on_bus_message(
        self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop
    ) -> bool:
        """Handle GStreamer bus messages (EOS and ERROR)."""
        if msg.type == Gst.MessageType.EOS:
            logger.info("VideoReceiver: pipeline EOS")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("VideoReceiver pipeline error: %s — %s", err, debug)
            loop.quit()
        return True

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Bind the UDP socket, start the decode pipeline, and begin
        delivering frames to the callback / ``get_frame()``.

        Raises
        ------
        RuntimeError
            If the pipeline cannot be parsed or started.
        """
        pipeline_str = self._build_pipeline_str()
        logger.debug("VideoReceiver pipeline: %s", pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        if self._pipeline is None:
            raise RuntimeError("VideoReceiver: Gst.parse_launch() returned None")

        sink: Gst.Element = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self._loop)

        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("VideoReceiver: pipeline failed to reach PLAYING state")

        self._loop_thread = threading.Thread(
            target=self._loop.run, name="VideoReceiver-loop", daemon=True
        )
        self._loop_thread.start()
        logger.info("VideoReceiver listening on UDP port %d", self._port)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop the receive pipeline and release resources."""
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
        logger.info("VideoReceiver stopped")

    # ------------------------------------------------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recently decoded video frame (thread-safe).

        Returns
        -------
        numpy.ndarray or None
            BGR image array of shape ``(HEIGHT, WIDTH, 3)``, dtype
            ``uint8``, or ``None`` if no frame has arrived yet.
        """
        with self._frame_lock:
            return self._latest_frame


# ---------------------------------------------------------------------------
# FeedbackSender  (runs on the PC)
# ---------------------------------------------------------------------------

class FeedbackSender(FeedbackTransferer):
    """
    Sends audio (Opus/RTP) and display framebuffer data (raw UDP) from the
    PC to the Pi.

    Two channels
    ------------
    * **Audio** — Raw Opus frames are pushed into a GStreamer ``appsrc``,
      packetised as RTP (payload type 111), and sent over UDP to the Pi.
    * **Display** — The 1024-byte raw framebuffer is sent as a single UDP
      datagram.  No GStreamer overhead is needed for such small payloads.

    Parameters
    ----------
    pi_host : str
        IP address (or hostname) of the Raspberry Pi Zero 2 W.
    audio_port : int
        UDP port that ``FeedbackReceiver`` on the Pi listens on for audio.
        Defaults to ``FeedbackConfig.AUDIO_PORT``.
    display_port : int
        UDP port that ``FeedbackReceiver`` on the Pi listens on for display
        data.  Defaults to ``FeedbackConfig.DISPLAY_PORT``.

    Raises
    ------
    RuntimeError
        From ``start()`` if the GStreamer audio pipeline cannot be started.
    ValueError
        From ``send_display()`` if the supplied framebuffer is not exactly
        ``FeedbackConfig.DISPLAY_BYTES`` bytes.

    Example
    -------
    >>> sender = FeedbackSender(pi_host="192.168.1.100")
    >>> sender.start()
    >>> sender.send_audio(opus_bytes)      # alert sound to Pi speaker
    >>> sender.send_display(frame_bytes)   # update Pi OLED
    >>> sender.stop()

    Notes
    -----
    * ``send_audio`` pushes an Opus frame directly into the GStreamer
      pipeline appsrc with a monotonically increasing presentation
      timestamp (20 ms per call by default).
    * ``send_display`` is fire-and-forget: if the UDP packet is lost, the
      Pi display retains the previous image, which is acceptable for this
      application.
    """

    def __init__(
        self,
        pi_host: str,
        audio_port: int = FeedbackConfig.AUDIO_PORT,
        display_port: int = FeedbackConfig.DISPLAY_PORT,
    ) -> None:
        self._pi_host = pi_host
        self._audio_port = audio_port
        self._display_port = display_port
        self._cfg = self.CONFIG

        self._audio_pipeline: Optional[Gst.Pipeline] = None
        self._audio_appsrc: Optional[Gst.Element] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Lightweight UDP socket for display frames (no GStreamer needed).
        self._display_sock: Optional[socket.socket] = None

        # Monotonically increasing presentation timestamp for audio buffers.
        self._audio_pts: int = 0

    # ------------------------------------------------------------------
    def _build_audio_pipeline_str(self) -> str:
        """
        Build the GStreamer pipeline string for the audio downlink.

        Returns
        -------
        str
            Pipeline that reads raw Opus frames from an appsrc, wraps them
            in RTP, and sends them over UDP to the Pi.
        """
        return (
            "appsrc name=audio_src is-live=true format=time "
            "caps=\"audio/x-opus,rate=48000,channels=1\" ! "
            "rtpopuspay pt=111 ! "
            f"udpsink host={self._pi_host} port={self._audio_port}"
        )

    # ------------------------------------------------------------------
    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop) -> bool:
        """Handle GStreamer bus messages for the audio pipeline."""
        if msg.type == Gst.MessageType.EOS:
            logger.info("FeedbackSender audio pipeline: EOS")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("FeedbackSender audio error: %s — %s", err, debug)
            loop.quit()
        return True

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Open the display UDP socket and start the audio GStreamer pipeline.

        Must be called before ``send_audio()`` or ``send_display()``.

        Raises
        ------
        RuntimeError
            If the GStreamer audio pipeline cannot be initialised.
        """
        # -- Audio pipeline --
        pipeline_str = self._build_audio_pipeline_str()
        logger.debug("FeedbackSender audio pipeline: %s", pipeline_str)

        self._audio_pipeline = Gst.parse_launch(pipeline_str)
        if self._audio_pipeline is None:
            raise RuntimeError("FeedbackSender: Gst.parse_launch() returned None")

        self._audio_appsrc = self._audio_pipeline.get_by_name("audio_src")

        self._loop = GLib.MainLoop()
        bus = self._audio_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self._loop)

        if self._audio_pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("FeedbackSender: audio pipeline failed to reach PLAYING state")

        self._loop_thread = threading.Thread(
            target=self._loop.run, name="FeedbackSender-loop", daemon=True
        )
        self._loop_thread.start()

        # -- Display socket --
        self._display_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        logger.info(
            "FeedbackSender started → %s (audio:%d, display:%d)",
            self._pi_host, self._audio_port, self._display_port,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Shut down the audio pipeline and close the display socket."""
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._audio_pipeline:
            self._audio_pipeline.set_state(Gst.State.NULL)
            self._audio_pipeline = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
        if self._display_sock:
            self._display_sock.close()
            self._display_sock = None
        logger.info("FeedbackSender stopped")

    # ------------------------------------------------------------------
    def send_audio(self, opus_bytes: bytes) -> None:
        """
        Push a block of raw Opus audio to the Pi's speaker.

        The bytes are wrapped in a GStreamer buffer with a monotonically
        increasing presentation timestamp (20 ms per call) and pushed into
        the ``appsrc`` element of the audio pipeline.

        Parameters
        ----------
        opus_bytes : bytes
            One Opus frame or a contiguous sequence of Opus frames in
            **raw Opus format** (not Ogg-encapsulated).  Each standard
            Opus frame is 20 ms at 48 kHz.

        Raises
        ------
        RuntimeError
            If ``start()`` has not been called yet.

        Notes
        -----
        This method is non-blocking.  The GStreamer pipeline drains the
        data asynchronously on its own thread.
        """
        if self._audio_appsrc is None:
            raise RuntimeError("FeedbackSender: call start() before send_audio()")

        # Standard Opus frame duration: 20 ms expressed as GStreamer nanoseconds.
        duration_ns = 20 * Gst.MSECOND
        buf = Gst.Buffer.new_wrapped(opus_bytes)
        buf.pts = self._audio_pts
        buf.duration = duration_ns
        self._audio_pts += duration_ns

        ret = self._audio_appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.warning("FeedbackSender: push-buffer returned %s", ret)

    # ------------------------------------------------------------------
    def send_display(self, frame_bytes: bytes) -> None:
        """
        Send a 128×64 1-bit framebuffer to the Pi's OLED display.

        The entire framebuffer is transmitted as a single UDP datagram.
        Because UDP is unreliable, a missed packet means the Pi display
        simply retains the previous image — acceptable for this use case.

        Parameters
        ----------
        frame_bytes : bytes
            Exactly ``FeedbackConfig.DISPLAY_BYTES`` (1024) bytes of
            packed monochrome pixel data in row-major order, MSB-first
            within each byte.

        Raises
        ------
        ValueError
            If ``len(frame_bytes) != FeedbackConfig.DISPLAY_BYTES``.
        RuntimeError
            If ``start()`` has not been called yet.
        """
        if len(frame_bytes) != self._cfg.DISPLAY_BYTES:
            raise ValueError(
                f"send_display expects exactly {self._cfg.DISPLAY_BYTES} bytes, "
                f"got {len(frame_bytes)}"
            )
        if self._display_sock is None:
            raise RuntimeError("FeedbackSender: call start() before send_display()")

        self._display_sock.sendto(frame_bytes, (self._pi_host, self._display_port))


# ---------------------------------------------------------------------------
# FeedbackReceiver  (runs on the Pi)
# ---------------------------------------------------------------------------

class FeedbackReceiver(FeedbackTransferer):
    """
    Receives audio (Opus/RTP) and display framebuffer (raw UDP) on the Pi
    and drives the speaker and OLED accordingly.

    Two channels
    ------------
    * **Audio** — A GStreamer pipeline receives RTP/Opus from the PC,
      decodes it, and plays it through the system audio output
      (``autoaudiosink`` → ALSA on the Pi).
    * **Display** — A background thread receives 1024-byte UDP datagrams
      and delivers them to an optional callback (e.g. an SSD1306 driver).

    Parameters
    ----------
    audio_port : int
        UDP port to bind for incoming RTP/Opus audio.
        Defaults to ``FeedbackConfig.AUDIO_PORT``.
    display_port : int
        UDP port to bind for incoming raw display framebuffer data.
        Defaults to ``FeedbackConfig.DISPLAY_PORT``.
    bind_host : str
        Network interface address to bind the display UDP socket to.
        Use ``"127.0.0.1"`` for loopback-only (testing) or the Pi's WLAN
        IP address (e.g. ``"192.168.1.100"``) to accept traffic only from
        the local subnet.  Defaults to ``"0.0.0.0"`` (all interfaces) which
        is appropriate when the Pi's IP is dynamic or unknown at start time.
    display_callback : callable, optional
        Called with each received ``bytes`` framebuffer
        (length == ``FeedbackConfig.DISPLAY_BYTES``).  Intended to write
        the image to the physical SSD1306 OLED driver.

    Raises
    ------
    RuntimeError
        From ``start()`` if the GStreamer audio pipeline cannot be started.

    Example
    -------
    >>> def update_oled(data: bytes) -> None:
    ...     oled_driver.image(data)
    ...     oled_driver.show()
    ...
    >>> receiver = FeedbackReceiver(
    ...     bind_host="192.168.1.100",  # Pi's own IP — restrict to LAN
    ...     display_callback=update_oled,
    ... )
    >>> receiver.start()
    >>> # audio plays automatically; OLED updated via callback
    >>> receiver.stop()

    Notes
    -----
    * The display receive thread is a plain Python daemon thread; no
      GStreamer is involved in receiving the 1 KB/frame display channel.
    * ``get_latest_display()`` returns the most recently received
      framebuffer in a thread-safe manner.
    * Set ``bind_host`` to the Pi's own WLAN IP to limit who can send
      display updates (basic network-level access control).
    """

    def __init__(
        self,
        audio_port: int = FeedbackConfig.AUDIO_PORT,
        display_port: int = FeedbackConfig.DISPLAY_PORT,
        bind_host: str = "0.0.0.0",
        display_callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        self._audio_port = audio_port
        self._display_port = display_port
        self._bind_host = bind_host
        self._display_callback = display_callback
        self._cfg = self.CONFIG

        self._audio_pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._display_sock: Optional[socket.socket] = None
        self._display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_display: Optional[bytes] = None
        self._display_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _build_audio_pipeline_str(self) -> str:
        """
        Build the GStreamer pipeline string for receiving and playing audio.

        Returns
        -------
        str
            Pipeline that receives RTP/Opus over UDP, decodes it, and
            plays it through the system audio output.
        """
        return (
            f"udpsrc port={self._audio_port} "
            "caps=\"application/x-rtp,media=audio,clock-rate=48000,"
            "encoding-name=OPUS,payload=111\" ! "
            "rtpopusdepay ! "
            "opusdec ! "
            "audioconvert ! "
            "autoaudiosink"
        )

    # ------------------------------------------------------------------
    def _display_receive_loop(self) -> None:
        """
        Background thread target: receive display framebuffer datagrams.

        Blocks on the UDP socket waiting for 1024-byte datagrams from the
        PC's ``FeedbackSender``.  On receipt, the bytes are stored as the
        latest display frame and the optional ``display_callback`` is
        invoked.

        This method exits cleanly when ``_stop_event`` is set (by
        ``stop()``), or when the socket is closed.
        """
        assert self._display_sock is not None

        while not self._stop_event.is_set():
            try:
                # Buffer slightly larger than DISPLAY_BYTES to detect oversized packets.
                data, _ = self._display_sock.recvfrom(self._cfg.DISPLAY_BYTES + 16)
            except OSError:
                # Socket was closed by stop(); exit the loop.
                break

            if len(data) != self._cfg.DISPLAY_BYTES:
                logger.warning(
                    "FeedbackReceiver: unexpected display packet length %d (expected %d)",
                    len(data), self._cfg.DISPLAY_BYTES,
                )
                continue

            with self._display_lock:
                self._latest_display = data

            if self._display_callback is not None:
                try:
                    self._display_callback(data)
                except Exception:
                    logger.exception("FeedbackReceiver: display_callback raised an exception")

    # ------------------------------------------------------------------
    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop) -> bool:
        """Handle GStreamer bus messages for the audio pipeline."""
        if msg.type == Gst.MessageType.EOS:
            logger.info("FeedbackReceiver audio pipeline: EOS")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("FeedbackReceiver audio error: %s — %s", err, debug)
            loop.quit()
        return True

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Bind UDP sockets and start the audio playback pipeline and the
        display receive thread.

        Raises
        ------
        RuntimeError
            If the GStreamer audio pipeline cannot be started.
        """
        # -- Audio pipeline --
        pipeline_str = self._build_audio_pipeline_str()
        logger.debug("FeedbackReceiver audio pipeline: %s", pipeline_str)

        self._audio_pipeline = Gst.parse_launch(pipeline_str)
        if self._audio_pipeline is None:
            raise RuntimeError("FeedbackReceiver: Gst.parse_launch() returned None")

        self._loop = GLib.MainLoop()
        bus = self._audio_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self._loop)

        if self._audio_pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("FeedbackReceiver: audio pipeline failed to reach PLAYING state")

        self._loop_thread = threading.Thread(
            target=self._loop.run, name="FeedbackReceiver-audio-loop", daemon=True
        )
        self._loop_thread.start()

        # -- Display UDP socket --
        self._stop_event.clear()
        self._display_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to _bind_host (default "0.0.0.0" = all interfaces).
        # Set bind_host to the Pi's own WLAN IP to restrict incoming senders.
        self._display_sock.bind((self._bind_host, self._display_port))
        # Short timeout so the loop can check _stop_event periodically.
        self._display_sock.settimeout(1.0)

        self._display_thread = threading.Thread(
            target=self._display_receive_loop,
            name="FeedbackReceiver-display",
            daemon=True,
        )
        self._display_thread.start()

        logger.info(
            "FeedbackReceiver started (audio port %d, display port %d)",
            self._audio_port, self._display_port,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """
        Stop audio playback, close the display socket, and join all threads.

        Blocks until both the audio event-loop thread and the display
        receive thread have exited.
        """
        # Stop GLib loop and audio pipeline.
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._audio_pipeline:
            self._audio_pipeline.set_state(Gst.State.NULL)
            self._audio_pipeline = None
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None

        # Stop display receive thread.
        self._stop_event.set()
        if self._display_sock:
            self._display_sock.close()
            self._display_sock = None
        if self._display_thread:
            self._display_thread.join(timeout=2)
            self._display_thread = None

        logger.info("FeedbackReceiver stopped")

    # ------------------------------------------------------------------
    def get_latest_display(self) -> Optional[bytes]:
        """
        Return the most recently received display framebuffer (thread-safe).

        Returns
        -------
        bytes or None
            1024-byte raw monochrome framebuffer, or ``None`` if no frame
            has been received yet.
        """
        with self._display_lock:
            return self._latest_display


# ---------------------------------------------------------------------------
# TODO — next steps
# ---------------------------------------------------------------------------
# 1. [VideoSender]  Investigate hardware H.264 encoding on Pi Zero 2 W
#    (e.g. V4L2 M2M encoder via ``v4l2h264enc``) to reduce CPU load.
# 2. [VideoSender]  Add an appsrc-based fallback path for testing on
#    systems without /dev/video0 (feed synthetic or file-based frames).
# 3. [FeedbackSender / FeedbackReceiver]  Agree on a framing protocol for
#    the display channel (e.g. add a 2-byte sequence number header) to
#    detect dropped or reordered packets.
# 4. [FeedbackSender]  Support sending multi-frame Opus payloads (longer
#    audio clips) via a queue + background push thread.
# 5. [General]  Add reconnect / retry logic for all UDP pipelines so the
#    system recovers from a brief network outage without restarting.
# 6. [General]  Write unit tests using GStreamer's videotestsrc /
#    audiotestsrc to exercise VideoSender + VideoReceiver end-to-end on a
#    loopback interface.
# ---------------------------------------------------------------------------

