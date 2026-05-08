#!/usr/bin/env python3
"""
Test script for Pi Zero 2 W: VideoSender (TCP server + multipart JPEG).

This script starts the video sender pipeline and keeps it running.
Run on the Pi to stream video to the PC.

Usage:
    python3 test_video_pi.py --host 0.0.0.0 --port 5000
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

# Assume data_transfer module is in the same directory or PYTHONPATH
from data_transfer import VideoConfig, VideoSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Context manager to handle SIGINT / SIGTERM cleanly."""

    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d; shutting down...", signum)
        self.interrupted = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Test VideoSender (Pi) — TCP server + multipart JPEG"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="TCP bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="TCP port (default: 5000)",
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
        help="Frames per second (default: 10)",
    )

    args = parser.parse_args()

    # Build config from CLI args
    cfg = VideoConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port,
    )

    logger.info("VideoSender Configuration:")
    logger.info("  Resolution: %dx%d", cfg.width, cfg.height)
    logger.info("  FPS: %d", cfg.fps)
    logger.info("  Raw format: %s", cfg.raw_format)
    logger.info("  Colorimetry: %s", cfg.colorimetry)
    logger.info("  TCP bind: %s:%d", args.host, cfg.port)

    sender = VideoSender(cfg=cfg, bind_host=args.host)

    with GracefulShutdown() as shutdown:
        try:
            sender.start()
            logger.info("VideoSender is running. Press Ctrl+C to stop.")

            # Keep the main thread alive
            while not shutdown.interrupted:
                signal.pause()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            return 1
        finally:
            sender.stop()
            logger.info("Test complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())

