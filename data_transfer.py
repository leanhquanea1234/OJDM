from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoConfig:
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 10


class FeedbackConfig:
    # Display is a 128x64 monochrome (1-bit) screen.
    DISPLAY_WIDTH: int = 128
    DISPLAY_HEIGHT: int = 64
    DISPLAY_BPP: int = 1  # 1-bit black/white

    # Audio feedback is an Opus-encoded file/segment.
    AUDIO_FORMAT: str = "opus"

# TODO: Implement both ways: send(Pi), receive(PC)
class VideoTransferer(ABC):
    """
    Video uplink: Pi -> PC.
    Implementations are expected to be backed by GStreamer (e.g., appsink/appsrc).
    """
    CONFIG: VideoConfig = VideoConfig()

    @abstractmethod
    def start(self) -> None:
        """Start GStreamer pipeline(s) and any background threads."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop pipelines and release resources."""
        ...

# TODO: Implement both ways: send(PC), receive(Pi)
class FeedbackTransferer(ABC):
    """
    Feedback downlink: PC -> Pi.
    - Audio: Opus bytes (file/segment format must match pipeline contract).
    - Display: 128x64 1-bit framebuffer (exactly 1024 bytes when present).
    Implementations are expected to be backed by GStreamer (audio) and a display driver (or GStreamer if applicable).
    """
    CONFIG: FeedbackConfig = FeedbackConfig()

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

