#!/usr/bin/env python3
"""
Test script for PC: VideoReceiver with native GStreamer window (autovideosink).

No manual display code needed — GStreamer handles the window directly.

Usage:
    python3 test_video_pc_native.py --pi-host <pi ip addr> --port 5000
"""

import argparse
import logging
import signal
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


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
        description="Test VideoReceiver with native GStreamer window (autovideosink)"
    )
    parser.add_argument(
        "--pi-host",
        type=str,
        default="192.168.11.8", # my own pi's addr
        help="IP/hostname of the Pi running VideoSender",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="TCP port (default: 5000, must match Pi's port)",
    )

    args = parser.parse_args()

    logger.info("GStreamer native window test")
    logger.info("  Pi host: %s", args.pi_host)
    logger.info("  TCP port: %d", args.port)

    # Build pipeline: tcpclientsrc → demux → decode → autovideosink
    pipeline_str = (
        f"tcpclientsrc host={args.pi_host} port={args.port} ! "
        "multipartdemux ! jpegdec ! "
        "autovideosink"
    )

    logger.debug("Pipeline: %s", pipeline_str)

    try:
        pipeline = Gst.parse_launch(pipeline_str)
        if pipeline is None:
            raise RuntimeError("Gst.parse_launch() returned None")

        # Create GLib main loop to handle window events
        loop = GLib.MainLoop()

        # Attach bus message handler (EOS, ERROR)
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(bus, msg):
            if msg.type == Gst.MessageType.EOS:
                logger.info("Pipeline EOS")
                loop.quit()
            elif msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error("Pipeline error: %s — %s", err, debug)
                loop.quit()
            return True

        bus.connect("message", on_message)

        # Start pipeline
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline failed to reach PLAYING state")

        logger.info("Pipeline running. Window should open. Close window or Ctrl+C to stop.")

        with GracefulShutdown() as shutdown:
            # Run GLib main loop (handles window events)
            # Will quit when user closes window or signal received
            loop.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1
    finally:
        if 'pipeline' in locals():
            pipeline.set_state(Gst.State.NULL)
        logger.info("Test complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
