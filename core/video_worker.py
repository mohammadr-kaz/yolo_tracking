import os
import time
import queue
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


class _FrameReader(threading.Thread):
    """
    Reads (and, per the current frame-skip setting, cheaply discards)
    frames on its own thread, so decoding frame N+1 overlaps with the main
    thread's detection on frame N instead of adding to it. Decode and
    inference are genuinely independent costs that can run concurrently on
    a multi-core CPU, which is the one place extra threading is worth it
    here.

    This does NOT change which frame gets analyzed or displayed: the main
    loop still runs detection synchronously, on this thread's own output,
    before showing it. It only hides I/O latency behind compute time - it
    can never reintroduce a stale/lagging box.
    """

    def __init__(self, video_source, is_rtsp, max_reconnect_attempts,
                 reconnect_delay_sec, get_skip_count, on_error):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.is_rtsp = is_rtsp
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self.get_skip_count = get_skip_count
        self.on_error = on_error

        # Holds at most the single freshest frame: if the main thread
        # hasn't consumed the previous one yet, it gets dropped rather
        # than letting a backlog build up.
        self.frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.opened_event = threading.Event()
        self.failed_to_open = False

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            self.failed_to_open = True
            self.opened_event.set()
            self.frame_queue.put(None)
            return

        self.opened_event.set()
        consecutive_failures = 0

        while not self._stop_event.is_set():
            for _ in range(self.get_skip_count()):
                if not cap.grab():
                    break

            ret, frame = cap.read()

            if not ret:
                if self.is_rtsp:
                    consecutive_failures += 1
                    if consecutive_failures > self.max_reconnect_attempts:
                        self.on_error(
                            f"Lost connection to the camera at {self.video_source} after "
                            f"{self.max_reconnect_attempts} reconnect attempts.\n\n"
                            "Verify the camera and PC are on the same subnet and that "
                            "the camera is reachable, then try again."
                        )
                        break
                    cap.release()
                    time.sleep(self.reconnect_delay_sec)
                    cap = cv2.VideoCapture(self.video_source)
                    continue
                else:
                    break

            consecutive_failures = 0

            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put(frame)

        cap.release()
        self.frame_queue.put(None)


class VideoWorker(QThread):
    """
    Owns the video capture for the whole session (video file or RTSP
    stream). It is created once per "Connect"; frame reading happens on a
    _FrameReader helper thread that is started exactly once and kept
    alive for the whole session - "Start Tracking"/"Stop Tracking" only
    flip a flag read by this loop, so tracking always resumes from
    whatever frame is currently playing instead of restarting the source
    from frame 0.

    While tracking is on, detection runs synchronously on every frame this
    loop pulls off the reader: the box drawn is always the real detector
    output for that *exact* frame, never a cached or extrapolated guess,
    so there is no synchronization gap between what is shown and what was
    detected. Frame decoding for the *next* frame overlaps with detection
    on the current one (see _FrameReader) to keep the frame rate up
    without touching that guarantee - and "detect every N frames" skips
    the frames in between cheaply (no decode) rather than showing them
    with a stale box.
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

    def _get_skip_count(self):
        return (self.detect_every_n_frames - 1) if self.tracking_enabled else 0

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
        reader = None
        try:
            is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith('rtsp')

            if is_rtsp:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'

            reader = _FrameReader(
                self.video_source, is_rtsp,
                self.max_reconnect_attempts, self.reconnect_delay_sec,
                get_skip_count=self._get_skip_count,
                on_error=self.error_signal.emit,
            )
            reader.start()
            if not reader.opened_event.wait(timeout=15.0):
                self.error_signal.emit(f"Timed out opening video source: {self.video_source}")
                return

            if reader.failed_to_open:
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
            fps_smoothed = 0.0
            last_time = time.time()

            while self.running:
                try:
                    frame = reader.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if frame is None:
                    break  # stream ended, or a fatal error was already reported

                is_tracking = self.tracking_enabled
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

        except Exception as e:
            self.error_signal.emit(f"Streaming error: {str(e)}")
        finally:
            if reader is not None:
                reader.stop()
                reader.join(timeout=2.0)

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
