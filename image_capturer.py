#!/usr/bin/env python3
"""
NOTE: run with pi_video_sender_test.py on pi
Test script for PC: VideoReceiver with frame capture GUI.

Uses appsink to extract frames, displays latest frame via PIL in Tkinter,
and provides a button to capture frames for YOLO dataset.

Only displays the newest frame; older frames are discarded to prevent lag.

Usage:
    python3 test_video_pc_capture.py --pi-host <pi host's ip address> --port 5000 --output-dir ./<some folder>

Controls:
    Click "Capture" button to save current frame
    Click "Stop" button or close window to exit
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class FrameCapture:
    """Manages frame capture and auto-naming."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.frame_count = 0
        self.current_frame: np.ndarray | None = None
        self.frame_lock = Lock()
        
        logger.info("Frame capture initialized: %s", self.output_dir)

    def update_frame(self, frame: np.ndarray) -> None:
        """Store latest frame (discard older ones)."""
        with self.frame_lock:
            self.current_frame = frame.copy()

    def capture_current_frame(self) -> Path | None:
        """
        Save the current frame to disk with auto-generated filename.
        
        Returns
        -------
        Path or None
            Path to saved file, or None if no frame available.
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None
            
            self.frame_count += 1
            frame = self.current_frame.copy()
            count_id = self.frame_count
        
        # BGR to RGB
        frame_rgb = frame[..., ::-1]
        
        # Save as JPEG
        filename = f"{self.start_time.strftime('%Y%m%d_%H%M%S')}_{count_id:05d}.jpg"
        filepath = self.output_dir / filename
        
        img = Image.fromarray(frame_rgb, mode='RGB')
        img.save(filepath, quality=95)
        
        logger.info("Captured frame #%d → %s", count_id, filepath)
        return filepath

    def get_current_frame(self) -> np.ndarray | None:
        """Get a copy of the current frame."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()


class VideoGUI:
    """Tkinter GUI for video display + capture controls."""

    def __init__(self, root, frame_capture: FrameCapture):
        self.root = root
        self.root.title("Video Capture for YOLO")
        self.root.geometry("800x600")
        
        self.frame_capture = frame_capture
        self.running = True
        
        # Video label
        self.video_label = tk.Label(root, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Capture button
        self.capture_btn = ttk.Button(
            control_frame,
            text="Capture Frame",
            command=self._on_capture
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_btn = ttk.Button(
            control_frame,
            text="Stop",
            command=self._on_stop
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Waiting for frames...")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Handle window close
        root.protocol("WM_DELETE_WINDOW", self._on_stop)
        
        # Start polling (only display latest frame, discard rest)
        self._poll_and_display()

    def _poll_and_display(self) -> None:
        """Poll for latest frame and display it (skip queued frames)."""
        frame = self.frame_capture.get_current_frame()
        
        if frame is not None:
            # Resize and display
            frame_rgb = frame[..., ::-1]  # BGR → RGB
            img = Image.fromarray(frame_rgb)
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep reference
            
            self.status_label.config(
                text=f"Frames displayed | Captured: {self.frame_capture.frame_count}"
            )
        
        if self.running:
            self.root.after(33, self._poll_and_display)  # ~30 fps display refresh

    def _on_capture(self) -> None:
        """Capture button callback."""
        filepath = self.frame_capture.capture_current_frame()
        if filepath:
            self.status_label.config(text=f"Saved: {filepath.name}")
        else:
            self.status_label.config(text="No frame to capture yet")

    def _on_stop(self) -> None:
        """Stop button / window close callback."""
        self.running = False
        self.root.quit()


def main():
    parser = argparse.ArgumentParser(
        description="VideoReceiver with GUI + frame capture for YOLO dataset"
    )
    parser.add_argument("--pi-host", type=str, default="192.168.11.8") # default of my pi's
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="./captured_images")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    from data_transfer import VideoConfig, VideoReceiver

    cfg = VideoConfig(width=args.width, height=args.height, fps=args.fps, port=args.port)
    frame_capture = FrameCapture(args.output_dir)

    receiver = VideoReceiver(
        pi_host=args.pi_host,
        cfg=cfg,
        frame_callback=frame_capture.update_frame,  # Just update; don't queue
    )

    try:
        receiver.start()
        logger.info("VideoReceiver started")

        # Launch Tkinter GUI
        root = tk.Tk()
        gui = VideoGUI(root, frame_capture)
        root.mainloop()

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1
    finally:
        receiver.stop()
        logger.info("Total frames captured: %d", frame_capture.frame_count)

    return 0


if __name__ == "__main__":
