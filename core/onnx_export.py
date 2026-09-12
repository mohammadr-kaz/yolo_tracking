from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class OnnxExportWorker(QThread):
    """Exports a .pt model to .onnx on a background thread so the GUI
    doesn't freeze during export (which can take several seconds)."""

    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pt_path, imgsz=640):
        super().__init__()
        self.pt_path = pt_path
        self.imgsz = imgsz

    def run(self):
        try:
            model = YOLO(self.pt_path)
            exported_path = model.export(format="onnx", imgsz=self.imgsz, simplify=True)
            self.finished_ok.emit(str(exported_path))
        except Exception as e:
            self.failed.emit(str(e))
