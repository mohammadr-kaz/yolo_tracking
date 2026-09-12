from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QRadialGradient, QConicalGradient
import math


class RadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.azimuth = 0
        self.elevation = 0
        self.targets = []  # لیست اهداف با زمان
        self.scan_angle = 0  # زاویه اسکن رادار
        self.pulse_radius = 0  # شعاع پالس رادار

        # تایمر برای انیمیشن
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # هر 50 میلی‌ثانیه

    def update_animation(self):
        # چرخش خط اسکن
        self.scan_angle = (self.scan_angle + 2) % 360

        # پالس رادار
        self.pulse_radius += 5
        if self.pulse_radius > 150:
            self.pulse_radius = 0

        # حذف اهداف قدیمی (فید)
        self.targets = [(az, el, age + 1) for az, el, age in self.targets if age < 20]

        self.update()

    def set_angles(self, azimuth, elevation):
        self.azimuth = azimuth
        self.elevation = elevation
        # اضافه کردن هدف جدید
        self.targets.append((azimuth, elevation, 0))
        if len(self.targets) > 10:
            self.targets.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        size = min(width, height)
        center_x = width / 2
        center_y = height / 2
        radius = size / 2 - 20

        # پس‌زمینه تیره (تقریبا مشکی)
        painter.fillRect(0, 0, width, height, QColor(9, 9, 8))

        # گرادیانت شعاعی برای افکت نور (کهربایی کم‌رنگ)
        gradient = QRadialGradient(center_x, center_y, radius)
        gradient.setColorAt(0, QColor(60, 42, 15, 90))
        gradient.setColorAt(0.7, QColor(30, 22, 10, 45))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

        # دایره‌های شبکه
        for i in range(1, 5):
            r = radius * i / 4
            painter.setPen(QPen(QColor(201, 138, 59, 45), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(center_x, center_y), r, r)

        # خطوط شبکه (8 جهت)
        painter.setPen(QPen(QColor(201, 138, 59, 35), 1))
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x = center_x + radius * math.cos(rad)
            y = center_y + radius * math.sin(rad)
            painter.drawLine(QPointF(center_x, center_y), QPointF(x, y))

        # پالس رادار (موج انبساطی)
        if self.pulse_radius > 0:
            alpha = int(255 * (1 - self.pulse_radius / 150))
            painter.setPen(QPen(QColor(217, 155, 80, alpha), 2))
            painter.drawEllipse(QPointF(center_x, center_y),
                                self.pulse_radius, self.pulse_radius)

        # خط اسکن چرخان (افکت رادار)
        scan_gradient = QConicalGradient(center_x, center_y, self.scan_angle)
        scan_gradient.setColorAt(0, QColor(201, 138, 59, 0))
        scan_gradient.setColorAt(0.1, QColor(201, 138, 59, 140))
        scan_gradient.setColorAt(0.3, QColor(201, 138, 59, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(scan_gradient)
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

        # نمایش اهداف قدیمی (فید)
        for az, el, age in self.targets:
            alpha = int(255 * (1 - age / 20))
            size_factor = max(0.3, 1 - age / 30)

            # محاسبه موقعیت
            distance = (90 - abs(el)) / 90 * radius
            rad = math.radians(az - 90)
            x = center_x + distance * math.cos(rad)
            y = center_y + distance * math.sin(rad)

            # رسم هدف (قرمز خاکی - برای تمایز از رنگ شبکه)
            painter.setPen(QPen(QColor(196, 74, 46, alpha), 2))
            painter.setBrush(QColor(196, 74, 46, alpha // 2))
            point_size = 8 * size_factor
            painter.drawEllipse(QPointF(x, y), point_size, point_size)

            # دایره اطراف هدف
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(x, y), point_size + 3, point_size + 3)

        # نمایش هدف فعلی (بزرگتر و روشن‌تر)
        if self.azimuth != 0 or self.elevation != 0:
            distance = (90 - abs(self.elevation)) / 90 * radius
            rad = math.radians(self.azimuth - 90)
            x = center_x + distance * math.cos(rad)
            y = center_y + distance * math.sin(rad)

            # افکت درخشش
            glow = QRadialGradient(x, y, 15)
            glow.setColorAt(0, QColor(255, 0, 0, 200))
            glow.setColorAt(0.5, QColor(255, 100, 0, 100))
            glow.setColorAt(1, QColor(255, 0, 0, 0))
            painter.setBrush(glow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), 15, 15)

            # نقطه مرکزی
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPointF(x, y), 6, 6)

            # خطوط متقاطع
            painter.setPen(QPen(QColor(255, 255, 0, 200), 2))
            painter.drawLine(QPointF(x - 12, y), QPointF(x + 12, y))
            painter.drawLine(QPointF(x, y - 12), QPointF(x, y + 12))

            # مربع هدف
            painter.setPen(QPen(QColor(255, 0, 0, 150), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(x - 10), int(y - 10), 20, 20)

        # برچسب‌های جهت
        painter.setPen(QPen(QColor(201, 138, 59), 2))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        directions = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        for label, angle in directions:
            rad = math.radians(angle - 90)
            x = center_x + (radius + 15) * math.cos(rad)
            y = center_y + (radius + 15) * math.sin(rad)
            painter.drawText(QPointF(x - 10, y + 5), label)

        # نمایش مختصات در گوشه
        painter.setPen(QColor(201, 138, 59))
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(10, 20, f"AZ: {self.azimuth:.1f}°")
        painter.drawText(10, 40, f"EL: {self.elevation:.1f}°")

        # خط مرزی
        painter.setPen(QPen(QColor(201, 138, 59, 90), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
