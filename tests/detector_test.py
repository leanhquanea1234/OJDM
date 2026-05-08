#!/usr/bin/env python3
"""
Simple test: show VideoReceiver frames in Tkinter and draw YOLO detections.
"""

import logging
import tkinter as tk
from tkinter import ttk
from threading import Lock

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

from data_transfer import VideoConfig, VideoReceiver
from processor import Detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: reset when chaning model
MODEL_PATH = "./OJ_model_v1.pt"
CONFIDENCE_THRESHOLD = 0.3


class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = Lock()

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()


class DetectorGUI:
    def __init__(self, root, frame_buffer: FrameBuffer, detector: Detector):
        self.root = root
        self.root.title("Detector Test")
        self.root.geometry("900x700")

        self.frame_buffer = frame_buffer
        self.detector = detector
        self.running = True

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(control_frame, text="Waiting for frames...")
        self.status_label.pack(side=tk.LEFT, padx=5)

        root.protocol("WM_DELETE_WINDOW", self._on_stop)
        self._poll_and_display()

    def _poll_and_display(self) -> None:
        frame = self.frame_buffer.get()
        if frame is not None:
            detections = self.detector.detect(frame)

            # BGR -> RGB
            rgb = frame[..., ::-1]
            img = Image.fromarray(rgb)

            draw = ImageDraw.Draw(img)

            # Optional font; fallback if not available
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                label = f"{det.class_name} {det.confidence:.2f}"

                # box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # label background + text
                text_size = draw.textbbox((0, 0), label, font=font)
                tw = text_size[2] - text_size[0]
                th = text_size[3] - text_size[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="red")
                draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)

            # resize for display
            img.thumbnail((900, 700), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo

            self.status_label.config(text=f"Detections: {len(detections)}")

        if self.running:
            self.root.after(33, self._poll_and_display)  # ~30 fps UI refresh

    def _on_stop(self) -> None:
        self.running = False
        self.root.quit()


def main():
    pi_host = "192.168.11.8"
    cfg = VideoConfig(width=640, height=480, fps=10, port=5000)

    frame_buffer = FrameBuffer()
    detector = Detector(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device="cpu") 

    receiver = VideoReceiver(
        pi_host=pi_host,
        cfg=cfg,
        frame_callback=frame_buffer.update,
    )

    try:
        receiver.start()
        root = tk.Tk()
        gui = DetectorGUI(root, frame_buffer, detector)
        root.mainloop()
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
