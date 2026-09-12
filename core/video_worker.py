import os
import time
import threading

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


class _InferenceWorker(threading.Thread):
    """
    Runs YOLO inference on its own plain (non-Qt) thread so a slow model
    never blocks frame capture/display.

    Only the single most recently submitted frame is ever processed (a
    one-slot "mailbox", not a queue): if the model is slower than the
    camera, older frames are simply dropped instead of backing up, which is
    what keeps the displayed video real-time while tracking is on.
    """

    def __init__(self, model_path, device, imgsz, half, on_result, on_error):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.on_result = on_result
        self.on_error = on_error

        self.resolved_device = None
        self.effective_half = None

        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_id = -1
        self._new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._model = None

    def submit(self, frame, frame_id):
        with self._lock:
            self._pending_frame = frame
            self._pending_id = frame_id
        self._new_frame_event.set()

    def wait_until_ready(self, timeout=None):
        return self._ready_event.wait(timeout)

    def stop(self):
        self._stop_event.set()
        self._new_frame_event.set()

    def _resolve_device(self):
        cuda_available = bool(torch and torch.cuda.is_available())

        if self.device in (None, "auto", "Auto"):
            resolved = "cuda:0" if cuda_available else "cpu"
        elif str(self.device).lower() == "cpu":
            resolved = "cpu"
        else:
            if not cuda_available:
                raise RuntimeError(
                    "GPU (CUDA) was requested but PyTorch did not detect a CUDA-capable "
                    "GPU/driver on this machine. Install a CUDA-enabled PyTorch build, or "
                    "set Device to 'Auto' or 'CPU'."
                )
            resolved = self.device

        effective_half = self.half if self.half is not None else cuda_available
        # Half precision on CPU (in particular for ONNX-on-CPU) is not
        # supported/useful - always force it off there.
        if resolved == "cpu":
            effective_half = False

        return resolved, effective_half

    def run(self):
        try:
            self.resolved_device, self.effective_half = self._resolve_device()
            self._model = YOLO(self.model_path)
            print(f"[InferenceWorker] model={self.model_path} device={self.resolved_device} "
                  f"half={self.effective_half} imgsz={self.imgsz}")
            self._ready_event.set()
        except Exception as e:
            self.on_error(f"Failed to load model: {e}")
            self._ready_event.set()
            return

        while not self._stop_event.is_set():
            self._new_frame_event.wait()
            self._new_frame_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                frame = self._pending_frame
                frame_id = self._pending_id
                self._pending_frame = None

            if frame is None:
                continue

            try:
                results = self._model(frame, device=self.resolved_device, half=self.effective_half,
                                       imgsz=self.imgsz, verbose=False)
                box = self._extract_best_box(results, self._model)
                self.on_result(frame_id, frame, box)
            except Exception as e:
                self.on_error(f"Detection error: {e}")

    @staticmethod
    def _extract_best_box(results, model):
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = f"{model.names[cls]} {conf:.2f}"
                return (float(x1), float(y1), float(x2), float(y2), conf, cls, label)
        return None


class VideoWorker(QThread):
    """
    Owns the video capture for the whole session (video file or RTSP
    stream). It is created once per "Connect" and opens the capture exactly
    once - "Start Tracking"/"Stop Tracking" only flip a flag on the already
    running loop, so tracking always resumes from whatever frame is
    currently playing instead of restarting the source from frame 0.

    Detection runs on a second, decoupled thread (see _InferenceWorker) so
    a slow model degrades detection latency, not the displayed frame rate.
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

        self._inference = None
        self._inference_key = None
        self._model_path = None
        self._device = None
        self._imgsz = 640
        self._half = None

        self._last_box = None
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
        if enabled:
            if not self._model_path:
                self.error_signal.emit("Please provide a model path before starting tracking")
                return

            key = (self._model_path, self._device, self._imgsz, self._half)
            if self._inference is not None and self._inference_key != key:
                self._inference.stop()
                self._inference.join(timeout=1.0)
                self._inference = None
                self._last_box = None

            if self._inference is None:
                self._inference = _InferenceWorker(
                    self._model_path, self._device, self._imgsz, self._half,
                    on_result=self._on_inference_result,
                    on_error=self.error_signal.emit,
                )
                self._inference_key = key
                self._inference.start()

        self.tracking_enabled = enabled
        if not enabled:
            self._last_box = None
            self.angles_ready.emit(0.0, 0.0)

    def _on_inference_result(self, frame_id, frame, box):
        self._last_box = box
        if box is not None:
            self._maybe_save_dataset_sample(frame, box, frame_id)

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

                if self.tracking_enabled and self._inference is not None:
                    if frame_id % self.detect_every_n_frames == 0:
                        self._inference.submit(frame.copy(), frame_id)

                    box = self._last_box
                    if box is not None:
                        self._draw_box_and_emit_angles(frame, box)

                h, w = frame.shape[:2]
                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps_smoothed = instant_fps if fps_smoothed == 0 else (0.9 * fps_smoothed + 0.1 * instant_fps)

                status = "TRACKING" if self.tracking_enabled else "STREAMING (no detection)"
                color = (0, 255, 0) if self.tracking_enabled else (60, 170, 220)
                cv2.putText(frame, f"FPS: {fps_smoothed:.1f}  [{status}]", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if frame_id % 10 == 0:
                    self.fps_ready.emit(fps_smoothed)

                display_frame = self._resize_for_display(frame, self.display_width)
                self.frame_ready.emit(display_frame)

            cap.release()
            if self._inference is not None:
                self._inference.stop()
                self._inference.join(timeout=2.0)

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
