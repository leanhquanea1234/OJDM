# OJDM — Project TODO List

A prioritised list of next steps for the Orange Juice Detection Machine project.
Items are grouped by area and ordered roughly by dependency.

---

## 1  Model & Training  (`detector.py`)

- [ ] **Collect dataset** — photograph orange juice bottles/cartons from multiple
  angles, distances, and lighting conditions (target: ≥ 200 labelled images).
- [ ] **Annotate bounding boxes** — use a tool such as [Label Studio](https://labelstud.io/)
  or [Roboflow](https://roboflow.com/) and export in YOLO format.
- [ ] **Train model** — fine-tune a `yolov8n.pt` (nano) base on the annotated
  dataset.  Suggested starting point:
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  model.train(data="oj_dataset.yaml", epochs=100, imgsz=640)
  ```
- [ ] **Evaluate model** — check mAP@50 and mAP@50-95; iterate on data and
  hyperparameters until accuracy is acceptable.
- [ ] **Export weights** — save `runs/detect/train/weights/best.pt` and update
  `Detector(model_path=...)` in the main PC script.

---

## 2  PC Main Loop  (`processor.py` + `data_transfer.py`)

- [ ] **Write `main_pc.py`** — a script that:
  1. Instantiates `VideoReceiver`, `FeedbackSender`, `Detector`, `Decider`.
  2. Passes `detector.detect` as `VideoReceiver.frame_callback`.
  3. In the callback, calls `decider.evaluate()` and dispatches actions via
     `FeedbackSender`.
- [ ] **Load audio clips** — store Opus-encoded alert sounds in an `assets/`
  directory; map `ActionSet.audio_label` keys to file paths.
- [ ] **Display rendering** — integrate [Pillow](https://pillow.readthedocs.io/)
  to render text (count, confidence %) and bounding-box metadata into the
  128×64 framebuffer rather than the current solid-bar placeholder in
  `Decider._build_display_frame()`.
- [ ] **Show video on PC** — optionally display the incoming video with
  bounding-box overlays using `cv2.imshow()` for debugging.
- [ ] **Graceful shutdown** — register `SIGINT` / `SIGTERM` handlers to call
  `stop()` on all components cleanly.

---

## 3  Pi Zero 2 W Setup  (`machine.py`)

- [ ] **Wire OLED driver** — install `luma.oled` and implement
  `PiNode._on_display_frame()` to render the received framebuffer on a
  physical SSD1306 128×64 display via I²C.
- [ ] **Camera compatibility** — test `VideoSender` with the Pi Camera Module v3;
  this may require replacing `v4l2src` with `libcamerasrc` in the
  `_build_pipeline_str()` method.
- [ ] **Speaker output** — verify that `autoaudiosink` routes to the correct ALSA
  device; if not, replace it with `alsasink device=hw:0,0` (or the correct
  card/device index).
- [ ] **Write `main_pi.py`** — a minimal entry-point that creates a `PiNode`,
  calls `node.run()`, and keeps the process alive with `signal.pause()`.
- [ ] **Systemd service** — create `/etc/systemd/system/ojdm.service` so the Pi
  node starts automatically on boot and restarts on failure.
- [ ] **Watchdog** — if `stream_video` or `process_feedback` exits unexpectedly,
  restart the affected thread automatically.

---

## 4  Data Transfer  (`data_transfer.py`)

- [ ] **H.264 hardware encoding** — investigate the V4L2 M2M encoder
  (`v4l2h264enc`) on Pi Zero 2 W to reduce CPU load from `x264enc`.
- [ ] **Reconnect / retry** — add logic to re-establish GStreamer pipelines after
  a brief network outage without requiring a full process restart.
- [ ] **Display packet framing** — add a 2-byte sequence number to the 1024-byte
  display UDP datagram so the Pi can detect dropped or reordered packets.
- [ ] **Multi-frame audio** — support queuing and streaming longer Opus audio
  clips via a background push-thread rather than a single 20 ms call.
- [ ] **Network discovery** — consider mDNS (e.g. `avahi`) so neither side
  needs a hard-coded IP address.

---

## 5  Testing & CI

- [ ] **Unit tests** — use `videotestsrc` / `audiotestsrc` GStreamer elements to
  exercise `VideoSender` + `VideoReceiver` end-to-end on a loopback interface.
- [ ] **Mock detector tests** — verify `Decider.evaluate()` logic with synthetic
  `Detection` objects (no GPU or model required).
- [ ] **Integration test script** — run both `main_pc.py` and `main_pi.py` on the
  same machine using `127.0.0.1` as the host to confirm the full pipeline.
- [ ] **CI workflow** — add a GitHub Actions workflow that runs linting (`ruff`)
  and unit tests on every push.

---

## 6  Documentation

- [ ] **Architecture diagram** — create an `architecture.md` with an ASCII or
  vector diagram of the full data-flow (camera → Pi → PC → detector → Pi).
- [ ] **Hardware wiring guide** — document the I²C OLED wiring and camera module
  connection for the Pi Zero 2 W.
- [ ] **Setup guide** — step-by-step instructions for installing GStreamer
  plugins, Python dependencies, and configuring WLAN on both devices.
