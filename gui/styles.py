"""
Dark / blackish military theme.

Rationale: a saturated neon-green HUD look is hard on the eyes during
prolonged night use. This palette keeps everything near-black with a low,
warm amber accent (the same idea used in cockpit/NVG-friendly lighting)
instead of a bright saturated color, so it stays comfortable to look at
for long tracking sessions while still reading as a tactical interface.
"""

# Core palette
BG_WINDOW = "#0a0a09"        # near-black window background
BG_PANEL = "#131412"         # slightly-raised panel/group background
BG_INPUT = "#0d0e0c"         # input field background
BORDER = "#3a3b35"           # muted steel/olive border
BORDER_LIGHT = "#4d4e46"

TEXT_PRIMARY = "#c9cabf"     # off-white/bone body text (low glare)
TEXT_MUTED = "#84867c"

ACCENT = "#c98a3b"           # tactical amber - primary accent / readouts
ACCENT_HOVER = "#dba055"
ACCENT_PRESSED = "#9c6b2c"

DANGER = "#a13c30"           # muted red for stop/destructive actions
DANGER_HOVER = "#bf4a3c"
DANGER_PRESSED = "#7e2f26"

DARK_MILITARY_STYLE = f"""
QMainWindow {{
    background-color: {BG_WINDOW};
}}

QWidget {{
    background-color: {BG_WINDOW};
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 10.5pt;
}}

QPushButton {{
    background-color: {BG_PANEL};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    padding: 4px 8px;
    font-weight: bold;
    min-width: 70px;
    min-height: 18px;
    font-size: 9.5pt;
}}

QPushButton:hover {{
    background-color: #201f18;
    border: 1px solid {ACCENT_HOVER};
    color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: #0d0d0a;
    color: {ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: {BG_PANEL};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
}}

QPushButton#dangerButton {{
    color: {DANGER};
    border: 1px solid {DANGER};
}}

QPushButton#dangerButton:hover {{
    border: 1px solid {DANGER_HOVER};
    color: {DANGER_HOVER};
}}

QPushButton#dangerButton:pressed {{
    color: {DANGER_PRESSED};
}}

QLineEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 3px;
    selection-background-color: {ACCENT};
    selection-color: #0a0a09;
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

QLabel {{
    color: {TEXT_PRIMARY};
    font-weight: bold;
    background-color: transparent;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 2px;
    margin-top: 10px;
    font-weight: bold;
    color: {ACCENT};
    background-color: {BG_PANEL};
    padding-top: 4px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {ACCENT};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 3px;
}}

QCheckBox {{
    color: {TEXT_PRIMARY};
    font-weight: bold;
    background-color: transparent;
    spacing: 6px;
}}

QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 3px;
}}

QComboBox:hover {{
    border: 1px solid {ACCENT};
}}

QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #0a0a09;
}}

QRadioButton {{
    color: {TEXT_PRIMARY};
    font-weight: bold;
    background-color: transparent;
    spacing: 6px;
}}

QRadioButton::indicator {{
    width: 12px;
    height: 12px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 7px;
    background-color: {BG_INPUT};
}}

QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}

QMessageBox {{
    background-color: {BG_PANEL};
}}
"""
