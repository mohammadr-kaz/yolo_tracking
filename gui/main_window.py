from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog,
                             QGroupBox, QComboBox, QRadioButton, QButtonGroup,
                             QMessageBox, QSpinBox, QCheckBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

from core.video_worker import VideoWorker
from core.onnx_export import OnnxExportWorker
from gui.radar_widget import RadarWidget
from gui.styles import DARK_MILITARY_STYLE, ACCENT, BORDER


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Tactical Tracker")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(DARK_MILITARY_STYLE)

        self.video_worker = None
        self.export_worker = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        display_panel = self.create_display_panel()
        main_layout.addWidget(display_panel, 3)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to YOLO model (.pt or .onnx)...")

        model_btn_layout = QHBoxLayout()
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        self.export_onnx_btn = QPushButton("Export to ONNX")
        self.export_onnx_btn.clicked.connect(self.export_to_onnx)
        model_btn_layout.addWidget(model_browse_btn)
        model_btn_layout.addWidget(self.export_onnx_btn)

        model_layout.addWidget(QLabel("Model Path:"))
        model_layout.addWidget(self.model_path_input)
        model_layout.addLayout(model_btn_layout)

        onnx_hint = QLabel("ONNX models load/run faster than .pt on CPU-only machines.")
        onnx_hint.setStyleSheet("color: #888; font-size: 9pt;")
        onnx_hint.setWordWrap(True)
        model_layout.addWidget(onnx_hint)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Video source selection
        video_group = QGroupBox("Video Source")
        video_layout = QVBoxLayout()

        self.source_type_group = QButtonGroup()
        self.file_radio = QRadioButton("Video File")
        self.rtsp_radio = QRadioButton("RTSP Stream")
        self.file_radio.setChecked(True)

        self.source_type_group.addButton(self.file_radio)
        self.source_type_group.addButton(self.rtsp_radio)

        self.file_radio.toggled.connect(self.toggle_source_type)

        video_layout.addWidget(self.file_radio)
        video_layout.addWidget(self.rtsp_radio)

        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Path to video file...")
        self.video_browse_btn = QPushButton("Browse Video")
        self.video_browse_btn.clicked.connect(self.browse_video)

        self.rtsp_container = QWidget()
        rtsp_container_layout = QVBoxLayout(self.rtsp_container)
        rtsp_container_layout.setContentsMargins(0, 0, 0, 0)

        rtsp_address_layout = QHBoxLayout()
        self.rtsp_ip = QLineEdit()
        self.rtsp_ip.setPlaceholderText("192.168.1.100")
        self.rtsp_port = QLineEdit()
        self.rtsp_port.setPlaceholderText("554")
        self.rtsp_port.setMaximumWidth(70)
        rtsp_address_layout.addWidget(QLabel("IP:"))
        rtsp_address_layout.addWidget(self.rtsp_ip)
        rtsp_address_layout.addWidget(QLabel("Port:"))
        rtsp_address_layout.addWidget(self.rtsp_port)

        rtsp_container_layout.addLayout(rtsp_address_layout)
        self.rtsp_container.setVisible(False)

        video_layout.addWidget(QLabel("Source:"))
        video_layout.addWidget(self.video_path_input)
        video_layout.addWidget(self.video_browse_btn)
        video_layout.addWidget(self.rtsp_container)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (recommended)", "CPU", "GPU (CUDA)"])
        perf_layout.addWidget(QLabel("Device:"))
        perf_layout.addWidget(self.device_combo)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "480", "640", "960", "1280"])
        self.imgsz_combo.setCurrentText("640")
        perf_layout.addWidget(QLabel("Inference size (smaller = faster):"))
        perf_layout.addWidget(self.imgsz_combo)

        self.detect_n_spin = QSpinBox()
        self.detect_n_spin.setRange(1, 30)
        self.detect_n_spin.setValue(1)
        perf_layout.addWidget(QLabel("Run detection every N frames (skip N-1 for speed):"))
        perf_layout.addWidget(self.detect_n_spin)

        perf_hint = QLabel("Every displayed frame while tracking shows its own exact, "
                            "just-computed box - never a stale or predicted one. Raising "
                            "'detect every N frames' skips the frames in between at low "
                            "cost, trading a lower frame rate while tracking for a box "
                            "that's always precisely in sync with what's shown.")
        perf_hint.setStyleSheet("color: #888; font-size: 9pt;")
        perf_hint.setWordWrap(True)
        perf_layout.addWidget(perf_hint)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Camera FOV
        fov_group = QGroupBox("Camera Field of View")
        fov_layout = QVBoxLayout()

        self.fov_x_input = QLineEdit("60")
        self.fov_y_input = QLineEdit("40")

        fov_layout.addWidget(QLabel("Horizontal FOV (degrees):"))
        fov_layout.addWidget(self.fov_x_input)
        fov_layout.addWidget(QLabel("Vertical FOV (degrees):"))
        fov_layout.addWidget(self.fov_y_input)

        fov_group.setLayout(fov_layout)
        layout.addWidget(fov_group)

        # Dataset capture
        dataset_group = QGroupBox("Dataset Capture (YOLO format)")
        dataset_layout = QVBoxLayout()

        self.dataset_enable_check = QCheckBox("Save frame + label above confidence threshold")
        dataset_layout.addWidget(self.dataset_enable_check)

        self.dataset_dir_input = QLineEdit()
        self.dataset_dir_input.setPlaceholderText("Output folder...")
        dataset_dir_btn = QPushButton("Browse Folder")
        dataset_dir_btn.clicked.connect(self.browse_dataset_dir)
        dataset_layout.addWidget(QLabel("Output folder:"))
        dataset_layout.addWidget(self.dataset_dir_input)
        dataset_layout.addWidget(dataset_dir_btn)

        self.dataset_conf_spin = QSpinBox()
        self.dataset_conf_spin.setRange(1, 100)
        self.dataset_conf_spin.setValue(50)
        self.dataset_conf_spin.setSuffix(" %")
        dataset_layout.addWidget(QLabel("Minimum confidence to save:"))
        dataset_layout.addWidget(self.dataset_conf_spin)

        dataset_hint = QLabel("Saves <folder>/images/*.jpg + <folder>/labels/*.txt while tracking runs.")
        dataset_hint.setStyleSheet("color: #888; font-size: 9pt;")
        dataset_hint.setWordWrap(True)
        dataset_layout.addWidget(dataset_hint)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Control buttons

        # Connect/disconnect the video source (file or RTSP) - no detection runs here.
        stream_btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_stream)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("dangerButton")
        self.disconnect_btn.clicked.connect(self.disconnect_stream)
        self.disconnect_btn.setEnabled(False)
        stream_btn_layout.addWidget(self.connect_btn)
        stream_btn_layout.addWidget(self.disconnect_btn)
        layout.addLayout(stream_btn_layout)

        # Start/stop tracking - toggles detection on the already-running stream,
        # so tracking always resumes from the current frame, never from frame 0.
        track_btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        track_btn_layout.addWidget(self.start_btn)
        track_btn_layout.addWidget(self.stop_btn)
        layout.addLayout(track_btn_layout)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(panel)
        return scroll

    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Video display
        video_group = QGroupBox("Video Output")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(f"border: 2px solid {BORDER};")
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group, 3)

        # Bottom section
        bottom_layout = QHBoxLayout()

        # Logo (bottom left)
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
        else:
            logo_label.setText("LOGO")
            logo_label.setStyleSheet(f"color: {BORDER}; font-size: 10pt;")
        logo_label.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        bottom_layout.addWidget(logo_label)

        # Angles display
        angles_group = QGroupBox("Target Angles")
        angles_layout = QVBoxLayout()

        self.azimuth_label = QLabel("Azimuth: 0.00°")
        self.azimuth_label.setStyleSheet(f"font-size: 16pt; color: {ACCENT};")
        self.elevation_label = QLabel("Elevation: 0.00°")
        self.elevation_label.setStyleSheet(f"font-size: 16pt; color: {ACCENT};")

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(f"font-size: 12pt; color: {ACCENT};")

        angles_layout.addWidget(self.azimuth_label)
        angles_layout.addWidget(self.elevation_label)
        angles_layout.addWidget(self.fps_label)
        angles_group.setLayout(angles_layout)
        bottom_layout.addWidget(angles_group, 1)

        # Radar display
        radar_group = QGroupBox("Radar Display")
        radar_layout = QVBoxLayout()
        self.radar_widget = RadarWidget()
        radar_layout.addWidget(self.radar_widget)
        radar_group.setLayout(radar_layout)
        bottom_layout.addWidget(radar_group, 1)

        layout.addLayout(bottom_layout, 1)

        return panel

    def toggle_source_type(self):
        if self.file_radio.isChecked():
            self.video_path_input.setVisible(True)
            self.video_browse_btn.setVisible(True)
            self.rtsp_container.setVisible(False)
        else:
            self.video_path_input.setVisible(False)
            self.video_browse_btn.setVisible(False)
            self.rtsp_container.setVisible(True)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "Model Files (*.pt *.onnx);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.video_path_input.setText(file_path)

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if dir_path:
            self.dataset_dir_input.setText(dir_path)

    def export_to_onnx(self):
        model_path = self.model_path_input.text().strip()
        if not model_path.lower().endswith(".pt"):
            QMessageBox.warning(self, "Export to ONNX", "Select a .pt model file to export first.")
            return

        try:
            imgsz = int(self.imgsz_combo.currentText())
        except ValueError:
            imgsz = 640

        self.export_onnx_btn.setEnabled(False)
        self.export_onnx_btn.setText("Exporting...")

        self.export_worker = OnnxExportWorker(model_path, imgsz)
        self.export_worker.finished_ok.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()

    def _on_export_done(self, onnx_path):
        self.export_onnx_btn.setEnabled(True)
        self.export_onnx_btn.setText("Export to ONNX")
        self.model_path_input.setText(onnx_path)
        QMessageBox.information(self, "Export complete", f"Exported ONNX model:\n{onnx_path}")

    def _on_export_failed(self, message):
        self.export_onnx_btn.setEnabled(True)
        self.export_onnx_btn.setText("Export to ONNX")
        QMessageBox.critical(self, "Export failed", message)

    def build_rtsp_url(self):
        ip = self.rtsp_ip.text().strip()
        port = self.rtsp_port.text().strip()

        if not ip or not port:
            return None

        return f"rtsp://{ip}:{port}/"

    def get_video_source(self):
        if self.file_radio.isChecked():
            video_source = self.video_path_input.text().strip()
            if not video_source:
                return None, "Please provide a video file path"
            return video_source, None
        else:
            video_source = self.build_rtsp_url()
            if not video_source:
                return None, "Please provide both the camera IP and Port"
            return video_source, None

    def connect_stream(self):
        """Open the video source (file or RTSP). No detection runs yet -
        this is a plain preview/connection step."""
        if self.video_worker is not None:
            return

        video_source, error = self.get_video_source()
        if error:
            QMessageBox.warning(self, "Input Error", error)
            return

        try:
            fov_x = float(self.fov_x_input.text())
            fov_y = float(self.fov_y_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid FOV values")
            return

        self.video_worker = VideoWorker(
            video_source=video_source,
            horizontal_fov=fov_x,
            vertical_fov=fov_y,
        )
        self.video_worker.frame_ready.connect(self.update_frame)
        self.video_worker.angles_ready.connect(self.update_angles)
        self.video_worker.error_signal.connect(self.show_error)
        self.video_worker.fps_ready.connect(self.update_fps)
        self.video_worker.finished.connect(self._on_stream_finished)
        self.video_worker.start()

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)

    def _on_stream_finished(self):
        """Called when the worker's run() loop ends on its own (e.g. end of
        a video file), so the UI doesn't stay stuck showing it as connected."""
        self.video_worker = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_label.setText("FPS: --")

    def disconnect_stream(self):
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker = None

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_label.setText("FPS: --")
        self.azimuth_label.setText("Azimuth: 0.00°")
        self.elevation_label.setText("Elevation: 0.00°")

    def start_tracking(self):
        model_path = self.model_path_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Input Error", "Please provide a model path")
            return

        # If not connected yet, connect now using the current source fields,
        # so tracking can be started directly without a separate step.
        if self.video_worker is None:
            self.connect_stream()
            if self.video_worker is None:
                return

        try:
            imgsz = int(self.imgsz_combo.currentText())
        except ValueError:
            imgsz = 640

        device_map = {
            "Auto (recommended)": None,
            "CPU": "cpu",
            "GPU (CUDA)": "cuda:0",
        }
        device = device_map.get(self.device_combo.currentText())
        detect_every_n = self.detect_n_spin.value()

        self.video_worker.configure_tracking(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            half=None,
            detect_every_n_frames=detect_every_n,
        )
        self.video_worker.configure_dataset_capture(
            enabled=self.dataset_enable_check.isChecked(),
            output_dir=self.dataset_dir_input.text().strip(),
            conf_threshold_pct=self.dataset_conf_spin.value(),
        )
        self.video_worker.set_tracking_enabled(True)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_tracking(self):
        """Stops detection but keeps the stream/connection running, so
        tracking can be resumed later from the current frame."""
        if self.video_worker:
            self.video_worker.set_tracking_enabled(False)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def update_angles(self, azimuth, elevation):
        self.azimuth_label.setText(f"Azimuth: {azimuth:.2f}°")
        self.elevation_label.setText(f"Elevation: {elevation:.2f}°")
        self.radar_widget.set_angles(azimuth, elevation)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        if self.video_worker:
            self.video_worker.stop()
        event.accept()
