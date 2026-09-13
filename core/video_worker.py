import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.config import CameraConfig

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


class VideoWorker(QThread):
    """
    Owns the video capture for the whole session (video file or RTSP
    stream). It is created once per "Connect" and opens the capture
    exactly once - "Start Tracking"/"Stop Tracking" only flip a flag on
    the already running loop, so tracking always resumes from whatever
    frame is currently playing instead of restarting the source from
    frame 0.

    While tracking is on, detection runs synchronously on every frame that
    gets displayed: the box drawn is always the real detector output for
    that *exact* frame, never a cached or extrapolated guess, so there is
    no synchronization gap between what is shown and what was detected.

    To keep this fast without decoupling display from detection (which
    would reintroduce a stale/lagging box), frames between detections are
    discarded cheaply with cv2.VideoCapture.grab() - which advances the
    stream without the costly decode step - instead of being fully read,
    processed and shown. So raising "detect every N frames" trades a lower
    displayed frame rate while tracking for a precise, up-to-the-instant
    box on every frame that IS shown, rather than showing every frame at
    full rate with a box that belongs to an older one.
    """

    frame_ready = pyqtSignal(np.ndarray)
    angles_ready = pyqtSignal(float, float)
    fps_ready = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, video_source, display_width=800,
                 horizontal_fov=60.0, vertical_fov=45.0,
                 max_reconnect_attempts=5, reconnect_delay_sec=2.0):
        super().__init__()
        self.video_source = video_source
        self.display_width = display_width
        self.camera_config = CameraConfig(horizontal_fov, vertical_fov)
        self.running = False

        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec

        # Tracking is toggled live from the GUI thread. Plain attribute
        # reads/writes are atomic under CPython's GIL, so no lock is
        # needed for this simple on/off flag.
        self.tracking_enabled = False
        self.detect_every_n_frames = 1

        self._model = None
        self._model_key = None
        self._model_path = None
        self._device = None
        self._imgsz = 640
        self._half = None
        self._resolved_device = None
        self._effective_half = None

        self._last_saved_frame_id = -1

        # Dataset capture (save frame + YOLO-format label above a
        # confidence threshold, into a user-chosen folder).
        self.dataset_capture_enabled = False
        self.dataset_dir = None
        self.dataset_conf_threshold = 0.5

    def configure_tracking(self, model_path, device, imgsz, half, detect_every_n_frames):
        self._model_path = model_path
        self._device = device
        self._imgsz = imgsz
        self._half = half
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))

    def configure_dataset_capture(self, enabled, output_dir, conf_threshold_pct):
        self.dataset_capture_enabled = enabled
        self.dataset_dir = output_dir
        self.dataset_conf_threshold = max(0.0, min(100.0, conf_threshold_pct)) / 100.0

    def set_tracking_enabled(self, enabled):
        if enabled and not self._model_path:
            self.error_signal.emit("Please provide a model path before starting tracking")
            return

        self.tracking_enabled = enabled
        if not enabled:
            self.angles_ready.emit(0.0, 0.0)

    def _ensure_model_loaded(self):
        """(Re)loads the model on this same thread, on demand, only when
        the model/device/imgsz/half configuration actually changed."""
        key = (self._model_path, self._device, self._imgsz, self._half)
        if self._model is not None and self._model_key == key:
            return True

        try:
            self._resolved_device, self._effective_half = self._resolve_device()
            self._model = YOLO(self._model_path)
            self._model_key = key
            print(f"[VideoWorker] model={self._model_path} device={self._resolved_device} "
                  f"half={self._effective_half} imgsz={self._imgsz}")
            return True
        except Exception as e:
            self.error_signal.emit(f"Failed to load model: {e}")
            self._model = None
            self._model_key = None
            self.tracking_enabled = False
            return False

    def _resolve_device(self):
        cuda_available = bool(torch and torch.cuda.is_available())

        if self._device in (None, "auto", "Auto"):
            resolved = "cuda:0" if cuda_available else "cpu"
        elif str(self._device).lower() == "cpu":
            resolved = "cpu"
        else:
            if not cuda_available:
                raise RuntimeError(
                    "GPU (CUDA) was requested but PyTorch did not detect a CUDA-capable "
                    "GPU/driver on this machine. Install a CUDA-enabled PyTorch build, or "
                    "set Device to 'Auto' or 'CPU'."
                )
            resolved = self._device

        effective_half = self._half if self._half is not None else cuda_available
        # Half precision on CPU (in particular for ONNX-on-CPU) is not
        # supported/useful - always force it off there.
        if resolved == "cpu":
            effective_half = False

        return resolved, effective_half

    def _detect_best_box(self, frame):
        results = self._model(frame, device=self._resolved_device, half=self._effective_half,
                               imgsz=self._imgsz, verbose=False)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = f"{self._model.names[cls]} {conf:.2f}"
                return (float(x1), float(y1), float(x2), float(y2), conf, cls, label)
        return None

    def _maybe_save_dataset_sample(self, frame, box, frame_id):
        if not (self.dataset_capture_enabled and self.dataset_dir):
            return
        if frame_id == self._last_saved_frame_id:
            return

        x1, y1, x2, y2, conf, cls, _ = box
        if conf < self.dataset_conf_threshold:
            return

        self._last_saved_frame_id = frame_id

        h, w = frame.shape[:2]
        images_dir = os.path.join(self.dataset_dir, "images")
        labels_dir = os.path.join(self.dataset_dir, "labels")
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            stamp = f"{int(time.time() * 1000)}_{frame_id}"
            cv2.imwrite(os.path.join(images_dir, f"{stamp}.jpg"), frame)

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            with open(os.path.join(labels_dir, f"{stamp}.txt"), "w") as f:
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        except Exception as e:
            self.error_signal.emit(f"Dataset capture failed: {e}")

    def run(self):
        try:
            is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith('rtsp')

            if is_rtsp:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'

            cap = cv2.VideoCapture(self.video_source)

            if not cap.isOpened():
                if is_rtsp:
                    self.error_signal.emit(
                        f"Cannot open camera stream at {self.video_source}.\n\n"
                        "Check that:\n"
                        "- the camera's IP is reachable from this PC (try 'ping <camera IP>')\n"
                        "- this PC's network adapter is on the same subnet as the camera\n"
                        "- the RTSP IP/port are correct"
                    )
                else:
                    self.error_signal.emit(f"Cannot open video source: {self.video_source}")
                return

            self.running = True

            frame_id = 0
            consecutive_failures = 0
            fps_smoothed = 0.0
            last_time = time.time()

            while self.running:
                is_tracking = self.tracking_enabled

                if is_tracking:
                    # Discard the in-between frames cheaply (grab() skips
                    # the decode step) so the frame we actually read below
                    # is as fresh as possible when we hand it to the model.
                    for _ in range(self.detect_every_n_frames - 1):
                        if not cap.grab():
                            break

                ret, frame = cap.read()

                if not ret:
                    if is_rtsp:
                        consecutive_failures += 1
                        if consecutive_failures > self.max_reconnect_attempts:
                            self.error_signal.emit(
                                f"Lost connection to the camera at {self.video_source} after "
                                f"{self.max_reconnect_attempts} reconnect attempts.\n\n"
                                "Verify the camera and PC are on the same subnet and that "
                                "the camera is reachable, then try again."
                            )
                            break
                        cap.release()
                        self.msleep(int(self.reconnect_delay_sec * 1000))
                        cap = cv2.VideoCapture(self.video_source)
                        continue
                    else:
                        break

                consecutive_failures = 0
                frame_id += 1
                h, w = frame.shape[:2]

                if is_tracking and self._ensure_model_loaded():
                    try:
                        box = self._detect_best_box(frame)
                    except Exception as e:
                        self.error_signal.emit(f"Detection error: {e}")
                        box = None

                    if box is not None:
                        self._draw_box_and_emit_angles(frame, box)
                        self._maybe_save_dataset_sample(frame, box, frame_id)
                    else:
                        self.angles_ready.emit(0.0, 0.0)

                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps_smoothed = instant_fps if fps_smoothed == 0 else (0.9 * fps_smoothed + 0.1 * instant_fps)

                status = "TRACKING" if is_tracking else "STREAMING (no detection)"
                color = (0, 255, 0) if is_tracking else (60, 170, 220)
                cv2.putText(frame, f"FPS: {fps_smoothed:.1f}  [{status}]", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if frame_id % 10 == 0:
                    self.fps_ready.emit(fps_smoothed)

                display_frame = self._resize_for_display(frame, self.display_width)
                self.frame_ready.emit(display_frame)

            cap.release()

        except Exception as e:
            self.error_signal.emit(f"Streaming error: {str(e)}")

    def _draw_box_and_emit_angles(self, frame, box):
        x1, y1, x2, y2, conf, cls, label = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        h, w = frame.shape[:2]
        azimuth, elevation = self.camera_config.calculate_angles(center_x, center_y, w, h)
        self.angles_ready.emit(azimuth, elevation)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        angle_text = f"Az: {azimuth:+.2f}° El: {elevation:+.2f}°"
        cv2.putText(frame, angle_text, (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        frame_center_x = w // 2
        frame_center_y = h // 2
        cv2.line(frame, (frame_center_x, frame_center_y),
                 (int(center_x), int(center_y)), (255, 0, 0), 2)
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

    def _resize_for_display(self, frame, target_width):
        h, w = frame.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def stop(self):
        self.running = False
        self.wait()
