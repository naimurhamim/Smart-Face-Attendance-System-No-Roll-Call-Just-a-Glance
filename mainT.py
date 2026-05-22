"""
Smart Face Attendance System
────────────────────────────
All-in-one PyQt5 + InsightFace attendance system.
"""

import sys
import json
import os
import csv
import cv2
import pickle
import shutil
import logging
import zipfile
import numpy as np
import threading
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from insightface.app import FaceAnalysis

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFrame, QStackedWidget,
    QScrollArea, QSizePolicy, QMessageBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QDateEdit,
    QProgressBar, QCheckBox, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QDate
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH     = "config/config.json"
USERS_PATH      = "data/users.json"
ATTENDANCE_PATH = "data/attendance.json"
DATASET_PATH    = "dataset/faces"
OUTPUT_PATH     = "output"

DEPTS = ["IRE", "CySE", "DSE", "SWE", "EdTE", "EEE", "CSE"]

# ── Locks ─────────────────────────────────────────────────────────────────────
_FA_LOCK          = threading.Lock()
_ATTENDANCE_LOCK  = threading.Lock()
_FA_INSTANCE      = None

# ── FA Singleton ──────────────────────────────────────────────────────────────
def get_insight_app():
    global _FA_INSTANCE
    with _FA_LOCK:
        if _FA_INSTANCE is None:
            fa = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            fa.prepare(ctx_id=0, det_size=(640, 640))
            _FA_INSTANCE = fa
    return _FA_INSTANCE

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = """
QWidget {
    background-color: #0a0a0f;
    color: #ffffff;
    font-family: 'Segoe UI';
}
QLabel#page_title {
    font-size: 22px;
    font-weight: bold;
    color: #d4af37;
}
QLabel#page_sub {
    font-size: 11px;
    color: rgba(192,192,192,0.4);
    letter-spacing: 3px;
}
QLabel#field_label {
    font-size: 11px;
    color: rgba(212,175,55,0.7);
    letter-spacing: 2px;
}
QLabel#stat_value {
    font-size: 28px;
    font-weight: bold;
    color: #d4af37;
}
QLabel#stat_label {
    font-size: 11px;
    color: rgba(192,192,192,0.5);
    letter-spacing: 2px;
}
QLineEdit {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: white;
    font-size: 14px;
}
QLineEdit:focus { border: 1px solid rgba(212,175,55,0.6); }
QComboBox {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: white;
    font-size: 14px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #1a1a2e;
    color: white;
    selection-background-color: rgba(212,175,55,0.3);
}
QDateEdit {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 10px;
    padding: 8px 12px;
    color: white;
    font-size: 13px;
}
QDateEdit::drop-down { border: none; }
QPushButton#primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #d4af37,stop:1 #b8962e);
    border: none;
    border-radius: 12px;
    padding: 14px;
    color: #0a0a0f;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 2px;
}
QPushButton#primary:hover   { background: #e8c84a; }
QPushButton#primary:disabled { background: #333; color: #666; }
QPushButton#secondary {
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 12px;
    padding: 14px;
    color: #d4af37;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 2px;
}
QPushButton#secondary:hover   { background: rgba(212,175,55,0.15); }
QPushButton#secondary:disabled { background: #111; color: #444; border-color: #333; }
QPushButton#danger {
    background: rgba(255,60,60,0.08);
    border: 1px solid rgba(255,60,60,0.3);
    border-radius: 12px;
    padding: 12px;
    color: #ff3c3c;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 2px;
}
QPushButton#danger:hover { background: rgba(255,60,60,0.18); }
QPushButton#success {
    background: rgba(0,200,100,0.08);
    border: 1px solid rgba(0,200,100,0.3);
    border-radius: 12px;
    padding: 12px;
    color: #00c864;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 2px;
}
QPushButton#success:hover { background: rgba(0,200,100,0.18); }
QPushButton#back {
    background: transparent;
    border: none;
    color: rgba(212,175,55,0.6);
    font-size: 13px;
    letter-spacing: 2px;
    text-align: left;
}
QPushButton#back:hover { color: #d4af37; }
QPushButton#toggle_off {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px 16px;
    color: rgba(255,255,255,0.4);
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}
QPushButton#toggle_on {
    background: rgba(0,200,100,0.12);
    border: 1px solid rgba(0,200,100,0.4);
    border-radius: 10px;
    padding: 10px 16px;
    color: #00c864;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}
QFrame#card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(212,175,55,0.12);
    border-radius: 20px;
}
QFrame#stat_card {
    background: rgba(212,175,55,0.05);
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: 16px;
}
QFrame#id_card {
    background: rgba(212,175,55,0.06);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 14px;
}
QFrame#absent_card {
    background: rgba(255,60,60,0.04);
    border: 1px solid rgba(255,60,60,0.15);
    border-radius: 14px;
}
QLabel#status_green {
    background: rgba(0,200,100,0.1);
    border: 1px solid rgba(0,200,100,0.3);
    color: #00c864;
    border-radius: 15px;
    padding: 6px 16px;
    font-size: 11px;
    letter-spacing: 2px;
}
QLabel#status_gold {
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    color: #d4af37;
    border-radius: 15px;
    padding: 6px 16px;
    font-size: 11px;
    letter-spacing: 2px;
}
QLabel#status_red {
    background: rgba(255,60,60,0.1);
    border: 1px solid rgba(255,60,60,0.3);
    color: #ff3c3c;
    border-radius: 15px;
    padding: 6px 16px;
    font-size: 11px;
    letter-spacing: 2px;
}
QLabel#status_blue {
    background: rgba(0,150,255,0.1);
    border: 1px solid rgba(0,150,255,0.3);
    color: #0096ff;
    border-radius: 15px;
    padding: 6px 16px;
    font-size: 11px;
    letter-spacing: 2px;
}
QScrollArea { border: none; background: transparent; }
QProgressBar {
    background: rgba(255,255,255,0.06);
    border: none;
    border-radius: 6px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #d4af37,stop:1 #b8962e);
    border-radius: 6px;
}
QTableWidget {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: 10px;
    gridline-color: rgba(255,255,255,0.04);
    color: white;
    font-size: 13px;
}
QTableWidget::item { padding: 8px 12px; }
QTableWidget::item:selected {
    background: rgba(212,175,55,0.15);
    color: #d4af37;
}
QHeaderView::section {
    background: rgba(212,175,55,0.08);
    color: rgba(212,175,55,0.8);
    padding: 8px 12px;
    border: none;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: bold;
}
"""

# ═════════════════════════════════════════════════════════════════════════════
# Data Helpers
# ═════════════════════════════════════════════════════════════════════════════
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def load_users():
    if not os.path.exists(USERS_PATH):
        return {"users": {}}
    with open(USERS_PATH) as f:
        return json.load(f)

def save_users(data):
    with open(USERS_PATH, "w") as f:
        json.dump(data, f, indent=4)

def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return {"attendance": {}}
    with open(ATTENDANCE_PATH) as f:
        data = json.load(f)
    if "attendance" not in data:
        return {"attendance": data}
    return data

def save_attendance(data):
    with _ATTENDANCE_LOCK:
        with open(ATTENDANCE_PATH, "w") as f:
            json.dump(data, f, indent=4)

def get_camera_index():
    try:
        return load_config().get("camera_index", 0)
    except Exception:
        return 0

def get_all_attendance_flat():
    data = load_attendance()
    rows = []
    for uid, records in data["attendance"].items():
        if isinstance(records, dict):
            records = [records]
        for r in records:
            rows.append({
                "id":       uid,
                "name":     r.get("name", uid),
                "dept":     r.get("dept", "N/A"),
                "datetime": r.get("date_time", ""),
            })
    rows.sort(key=lambda x: x["datetime"], reverse=True)
    return rows

def get_today_str():
    return datetime.now().strftime("%Y-%m-%d")

def get_dashboard_stats():
    users = load_users()
    att   = load_attendance()
    today = get_today_str()
    total = len(users["users"])

    present_today = set()
    for uid, records in att["attendance"].items():
        if isinstance(records, dict):
            records = [records]
        if records and records[-1]["date_time"].startswith(today):
            present_today.add(uid)

    absent_today = [
        {"id": uid, "name": info.get("name", uid), "dept": info.get("dept", "N/A")}
        for uid, info in users["users"].items()
        if uid not in present_today
    ]

    total_records = sum(
        len(r) if isinstance(r, list) else 1
        for r in att["attendance"].values()
    )

    return {
        "total":         total,
        "present_today": len(present_today),
        "absent_today":  len(absent_today),
        "absent_list":   absent_today,
        "total_records": total_records,
    }

def export_attendance_csv(filepath):
    rows = get_all_attendance_flat()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Student ID", "Name", "Department", "Date", "Time"])
        for r in rows:
            parts = r["datetime"].split(" ")
            writer.writerow([r["id"], r["name"], r["dept"],
                             parts[0] if parts else "",
                             parts[1] if len(parts) > 1 else ""])
    return len(rows)

def backup_data(filepath):
    conf = load_config()
    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [USERS_PATH, ATTENDANCE_PATH]:
            if os.path.exists(p):
                zf.write(p)
        for p in [conf.get("encodings_path", ""), conf.get("recognizer_path", ""), conf.get("le_path", "")]:
            if p and os.path.exists(p):
                zf.write(p)

def retrain_svm(conf, all_embeddings, all_names):
    """Retrain SVM in place. Returns True if trained, False if not enough data."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(conf["encodings_path"], "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "names": all_names}, f)

    if len(set(all_names)) >= 2:
        le     = LabelEncoder()
        labels = le.fit_transform(all_names)
        rec    = SVC(C=1.0, kernel="linear", probability=True)
        rec.fit(np.array(all_embeddings), labels)
        with open(conf["recognizer_path"], "wb") as f:
            pickle.dump(rec, f)
        with open(conf["le_path"], "wb") as f:
            pickle.dump(le, f)
        return True
    else:
        print("[INFO] Need 2+ people for SVM training. Existing model kept.")
        return False

def encode_and_train_incremental(user_id, conf):
    fa = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    encodings_path = conf["encodings_path"]

    if os.path.exists(encodings_path):
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        all_embeddings = list(data["embeddings"])
        all_names      = list(data["names"])
    else:
        all_embeddings, all_names = [], []

    filtered = [(e, n) for e, n in zip(all_embeddings, all_names) if n != user_id]
    if filtered:
        all_embeddings, all_names = zip(*filtered)
        all_embeddings, all_names = list(all_embeddings), list(all_names)
    else:
        all_embeddings, all_names = [], []

    user_dir  = os.path.join(DATASET_PATH, user_id)
    new_count = 0
    for fname in sorted(os.listdir(user_dir)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue
        img = cv2.imread(os.path.join(user_dir, fname))
        if img is None:
            continue
        faces = fa.get(img)
        if not faces:
            continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        all_embeddings.append(face.embedding)
        all_names.append(user_id)
        new_count += 1

    print(f"[INFO] Encoded {new_count} images for {user_id}")
    retrain_svm(conf, all_embeddings, all_names)

# ═════════════════════════════════════════════════════════════════════════════
# Signals
# ═════════════════════════════════════════════════════════════════════════════
class Signals(QObject):
    frame_ready    = pyqtSignal(np.ndarray)
    progress       = pyqtSignal(int, int)
    enroll_done    = pyqtSignal(bool, str)
    verify_result  = pyqtSignal(dict)
    status_update  = pyqtSignal(str)   # for background ops

# ═════════════════════════════════════════════════════════════════════════════
# Shared Camera Widget helpers
# ═════════════════════════════════════════════════════════════════════════════
def make_cam_label():
    lbl = QLabel()
    lbl.setFixedSize(380, 280)
    lbl.setStyleSheet("border:1px solid rgba(212,175,55,0.2);border-radius:12px;background:#000;")
    lbl.setAlignment(Qt.AlignCenter)
    return lbl

def frame_to_pixmap(frame, w=380, h=280):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw, c = rgb.shape
    img = QImage(rgb.data, fw, fh, fw * c, QImage.Format_RGB888)
    return QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def back_btn(callback):
    b = QPushButton("← BACK")
    b.setObjectName("back")
    b.setFixedWidth(100)
    b.clicked.connect(callback)
    return b

def page_header(title_txt, sub_txt):
    vbox = QVBoxLayout()
    t = QLabel(title_txt)
    t.setObjectName("page_title")
    t.setAlignment(Qt.AlignCenter)
    s = QLabel(sub_txt)
    s.setObjectName("page_sub")
    s.setAlignment(Qt.AlignCenter)
    vbox.addWidget(t)
    vbox.addWidget(s)
    return vbox

# ═════════════════════════════════════════════════════════════════════════════
# Home / Dashboard
# ═════════════════════════════════════════════════════════════════════════════
class HomeWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(18)

        # ── Title ──
        title = QLabel("Smart Face\nAttendance")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 30, QFont.Bold))
        title.setStyleSheet("color:#d4af37;font-size:30px;font-weight:bold;")
        root.addWidget(title)

        sub = QLabel("BIOMETRIC RECOGNITION PLATFORM")
        sub.setObjectName("page_sub")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        # ── Stat Cards ──
        self.stat_row = QHBoxLayout()
        self.stat_row.setSpacing(14)
        self._stat_total   = self._stat_card("—", "TOTAL STUDENTS")
        self._stat_present = self._stat_card("—", "PRESENT TODAY",  "#00c864")
        self._stat_absent  = self._stat_card("—", "ABSENT TODAY",   "#ff3c3c")
        self._stat_records = self._stat_card("—", "TOTAL RECORDS",  "#0096ff")
        root.addLayout(self.stat_row)

        # ── Absent today panel ──
        self.absent_frame = QFrame()
        self.absent_frame.setObjectName("absent_card")
        af_layout = QVBoxLayout(self.absent_frame)
        af_layout.setContentsMargins(16, 12, 16, 12)
        absent_title = QLabel("ABSENT TODAY")
        absent_title.setStyleSheet("color:rgba(255,60,60,0.7);font-size:11px;letter-spacing:2px;font-weight:bold;")
        af_layout.addWidget(absent_title)
        self.absent_scroll = QScrollArea()
        self.absent_scroll.setWidgetResizable(True)
        self.absent_scroll.setMaximumHeight(80)
        self.absent_inner = QWidget()
        self.absent_inner_layout = QHBoxLayout(self.absent_inner)
        self.absent_inner_layout.setAlignment(Qt.AlignLeft)
        self.absent_inner_layout.setSpacing(8)
        self.absent_scroll.setWidget(self.absent_inner)
        af_layout.addWidget(self.absent_scroll)
        root.addWidget(self.absent_frame)
        self.absent_frame.hide()

        # ── Nav Buttons ──
        row1 = QHBoxLayout(); row1.setSpacing(16)
        row2 = QHBoxLayout(); row2.setSpacing(16)

        def nav(label, page, style="secondary", w=210, h=60):
            b = QPushButton(label)
            b.setObjectName(style)
            b.setFixedSize(w, h)
            b.clicked.connect(lambda: self.parent.show_page(page))
            return b

        row1.addWidget(nav("🪪  ENROLL",       "enroll",     "primary", h=65))
        row1.addWidget(nav("✅  ATTENDANCE",    "attendance", "secondary", h=65))
        row2.addWidget(nav("📋  VIEW HISTORY",  "history"))
        row2.addWidget(nav("👥  STUDENTS",      "students"))
        row2.addWidget(nav("💾  BACKUP",        "backup"))

        root.addLayout(row1)
        root.addLayout(row2)

        footer = QLabel(f"© {datetime.now().year} Smart Face Attendance System")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:rgba(255,255,255,0.08);font-size:11px;")
        root.addWidget(footer)

    def _stat_card(self, value, label, color="#d4af37"):
        card = QFrame()
        card.setObjectName("stat_card")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(16, 14, 16, 14)
        vbox.setSpacing(4)
        val_lbl = QLabel(value)
        val_lbl.setObjectName("stat_value")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet(f"font-size:28px;font-weight:bold;color:{color};")
        lbl = QLabel(label)
        lbl.setObjectName("stat_label")
        lbl.setAlignment(Qt.AlignCenter)
        vbox.addWidget(val_lbl)
        vbox.addWidget(lbl)
        self.stat_row.addWidget(card)

        # store reference
        card._val_lbl = val_lbl
        return card

    def showEvent(self, event):
        self._refresh_stats()

    def _refresh_stats(self):
        stats = get_dashboard_stats()
        self._stat_total._val_lbl.setText(str(stats["total"]))
        self._stat_present._val_lbl.setText(str(stats["present_today"]))
        self._stat_absent._val_lbl.setText(str(stats["absent_today"]))
        self._stat_records._val_lbl.setText(str(stats["total_records"]))

        # Absent list
        absent = stats["absent_list"]
        for i in reversed(range(self.absent_inner_layout.count())):
            w = self.absent_inner_layout.itemAt(i).widget()
            if w: w.deleteLater()

        if absent:
            self.absent_frame.show()
            for a in absent[:20]:
                chip = QLabel(f"  {a['name']} ({a['id']})  ")
                chip.setStyleSheet(
                    "background:rgba(255,60,60,0.1);border:1px solid rgba(255,60,60,0.25);"
                    "color:#ff6060;border-radius:12px;padding:4px 8px;font-size:11px;"
                )
                self.absent_inner_layout.addWidget(chip)
            if len(absent) > 20:
                more = QLabel(f"  +{len(absent)-20} more  ")
                more.setStyleSheet("color:rgba(255,255,255,0.3);font-size:11px;")
                self.absent_inner_layout.addWidget(more)
        else:
            self.absent_frame.hide()

# ═════════════════════════════════════════════════════════════════════════════
# Enroll Window
# ═════════════════════════════════════════════════════════════════════════════
class EnrollWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent       = parent
        self.signals      = Signals()
        self.vs           = None
        self.timer        = QTimer()
        self.enrolling    = False
        self._stop_event  = threading.Event()
        self.timer.timeout.connect(self._read_frame)
        self._build()
        self._connect()

    def _build(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(30, 20, 30, 20)
        main.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(back_btn(self._go_back))
        top.addStretch()
        main.addLayout(top)

        for w in page_header("Face Enrollment", "REGISTER NEW IDENTITY").children():
            if isinstance(w, QLabel): main.addWidget(w)

        hdr = page_header("Face Enrollment", "REGISTER NEW IDENTITY")
        main.addLayout(hdr)

        card = QFrame(); card.setObjectName("card")
        cl   = QHBoxLayout(card)
        cl.setContentsMargins(20, 20, 20, 20)
        cl.setSpacing(20)

        # ── Camera side ──
        cam = QVBoxLayout()
        self.cam_label = make_cam_label()
        cam.addWidget(self.cam_label)

        self.status_badge = QLabel("CAMERA READY")
        self.status_badge.setObjectName("status_gold")
        self.status_badge.setAlignment(Qt.AlignCenter)
        cam.addWidget(self.status_badge)
        cl.addLayout(cam)

        # ── Form side ──
        form = QVBoxLayout(); form.setSpacing(10)

        def field(lbl_txt, widget):
            l = QLabel(lbl_txt); l.setObjectName("field_label")
            form.addWidget(l); form.addWidget(widget)

        self.name_input = QLineEdit(); self.name_input.setPlaceholderText("e.g. MD Naimur Rashid")
        self.id_input   = QLineEdit(); self.id_input.setPlaceholderText("e.g. IRE2101042")
        self.dept_input = QComboBox()
        self.dept_input.addItem("Select Department")
        for d in DEPTS: self.dept_input.addItem(d)

        field("FULL NAME",           self.name_input)
        field("STUDENT / PERSON ID", self.id_input)
        field("DEPARTMENT",          self.dept_input)

        # Progress
        self.progress_bar   = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("0 / ? captured")
        self.progress_label.setStyleSheet("color:rgba(212,175,55,0.6);font-size:12px;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        form.addWidget(self.progress_bar)
        form.addWidget(self.progress_label)

        self.enroll_btn = QPushButton("📸  PROCEED ENROLL")
        self.enroll_btn.setObjectName("primary")
        self.enroll_btn.clicked.connect(self._start_enroll)
        form.addWidget(self.enroll_btn)

        self.msg_label = QLabel("")
        self.msg_label.setAlignment(Qt.AlignCenter)
        self.msg_label.setWordWrap(True)
        form.addWidget(self.msg_label)

        form.addStretch()
        cl.addLayout(form)
        main.addWidget(card)

    def _connect(self):
        self.signals.frame_ready.connect(lambda f: self.cam_label.setPixmap(frame_to_pixmap(f)))
        self.signals.progress.connect(self._update_progress)
        self.signals.enroll_done.connect(self._enroll_finished)

    def showEvent(self, event):
        self._refresh_face_count_label()
        self._start_camera()

    def hideEvent(self, event):
        self._stop_camera()

    def _refresh_face_count_label(self):
        try:
            fc = load_config().get("face_count", 30)
        except Exception:
            fc = 30
        self.progress_label.setText(f"0 / {fc} captured")
        self.progress_bar.setMaximum(fc)
        self.progress_bar.setValue(0)

    def _start_camera(self):
        idx = get_camera_index()
        self.vs = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.timer.start(30)

    def _stop_camera(self):
        self.timer.stop()
        if self.vs:
            self.vs.release()
            self.vs = None

    def _read_frame(self):
        if self.vs and not self.enrolling:
            ret, frame = self.vs.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.rectangle(frame,
                    (cx - int(w*.22), cy - int(h*.35)),
                    (cx + int(w*.22), cy + int(h*.35)),
                    (212, 175, 55), 1)
                self.signals.frame_ready.emit(frame)

    def _start_enroll(self):
        name = self.name_input.text().strip()
        uid  = self.id_input.text().strip()
        dept = self.dept_input.currentText()

        if not name: return self._msg("Please enter Full Name.", "red")
        if not uid:  return self._msg("Please enter Person ID.", "red")
        if dept == "Select Department": return self._msg("Please select Department.", "red")

        users = load_users()
        if uid in users["users"]:
            reply = QMessageBox.question(
                self, "Already Enrolled",
                f"ID '{uid}' is already enrolled as '{users['users'][uid].get('name', uid)}'.\n\nRe-enroll (overwrite)?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            old = os.path.join(DATASET_PATH, uid)
            if os.path.exists(old):
                shutil.rmtree(old)

        self._stop_camera()
        self._stop_event.clear()
        self.enrolling = True
        self.enroll_btn.setEnabled(False)
        self._badge("ENROLLING...", "blue")
        self._msg("Please look at the camera steadily...", "gray")
        threading.Thread(target=self._enroll_thread, args=(uid, name, dept), daemon=True).start()

    def _enroll_thread(self, uid, name, dept):
        try:
            conf       = load_config()
            face_count = conf.get("face_count", 30)
            save_dir   = os.path.join(DATASET_PATH, uid)
            os.makedirs(save_dir, exist_ok=True)

            fa = get_insight_app()
            vs = cv2.VideoCapture(get_camera_index(), cv2.CAP_DSHOW)

            saved, attempts, max_att = 0, 0, face_count * 25

            while saved < face_count and attempts < max_att:
                if self._stop_event.is_set():
                    vs.release()
                    self.enrolling = False
                    return
                attempts += 1
                ret, frame = vs.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                self.signals.frame_ready.emit(frame.copy())

                faces = fa.get(frame)
                if not faces: continue

                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                pad = 30
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                crop = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(save_dir, f"{str(saved).zfill(5)}.png"), crop)
                saved += 1
                self.signals.progress.emit(saved, face_count)

            vs.release()
            self.enrolling = False

            if saved == 0:
                self.signals.enroll_done.emit(False, "No face detected. Check lighting & camera.")
                return

            users = load_users()
            users["users"][uid] = {
                "name":        name,
                "dept":        dept,
                "status":      "enrolled",
                "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_users(users)
            encode_and_train_incremental(uid, conf)
            self.signals.enroll_done.emit(True, f"{name} enrolled successfully! ({saved} images)")

        except Exception as e:
            logging.exception("Enroll thread error")
            self.enrolling = False
            self.signals.enroll_done.emit(False, str(e))

    def _update_progress(self, saved, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(saved)
        self.progress_label.setText(f"{saved} / {total} captured")

    def _enroll_finished(self, success, msg):
        self.enroll_btn.setEnabled(True)
        self._start_camera()
        if success:
            self._badge("ENROLLED ✓", "green")
            self._msg(f"✅ {msg}", "green")
            self.name_input.clear()
            self.id_input.clear()
            self.dept_input.setCurrentIndex(0)
            self._refresh_face_count_label()
        else:
            self._badge("FAILED", "red")
            self._msg(f"❌ {msg}", "red")

    def _badge(self, text, color):
        styles = {
            "gold":  "background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.3);color:#d4af37;",
            "blue":  "background:rgba(0,150,255,0.1);border:1px solid rgba(0,150,255,0.3);color:#0096ff;",
            "green": "background:rgba(0,200,100,0.1);border:1px solid rgba(0,200,100,0.3);color:#00c864;",
            "red":   "background:rgba(255,60,60,0.1);border:1px solid rgba(255,60,60,0.3);color:#ff3c3c;",
        }
        base = "border-radius:15px;padding:6px 16px;font-size:11px;letter-spacing:2px;"
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(styles.get(color, "") + base)

    def _msg(self, text, color="gray"):
        c = {"green": "#00c864", "red": "#ff3c3c", "gray": "rgba(192,192,192,0.5)"}
        self.msg_label.setText(text)
        self.msg_label.setStyleSheet(f"color:{c.get(color,'white')};font-size:13px;")

    def _go_back(self):
        self._stop_event.set()
        self._stop_camera()
        self.enrolling = False
        self.parent.show_page("home")

# ═════════════════════════════════════════════════════════════════════════════
# Attendance Window
# ═════════════════════════════════════════════════════════════════════════════
class AttendanceWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent     = parent
        self.signals    = Signals()
        self.vs         = None
        self.timer      = QTimer()
        self.verifying  = False
        self.auto_mode  = False
        self.auto_cooldown = 0        # frames to wait after a verify
        self.timer.timeout.connect(self._read_frame)
        self._build()
        self._connect()

    def _build(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(30, 20, 30, 20)
        main.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(back_btn(self._go_back))
        top.addStretch()
        main.addLayout(top)

        main.addLayout(page_header("Attendance Verification", "BIOMETRIC IDENTITY CHECK"))

        card = QFrame(); card.setObjectName("card")
        cl   = QHBoxLayout(card)
        cl.setContentsMargins(20, 20, 20, 20)
        cl.setSpacing(20)

        # ── Camera side ──
        cam = QVBoxLayout()
        self.cam_label = make_cam_label()
        cam.addWidget(self.cam_label)

        self.status_label = QLabel("Position your face and click verify.")
        self.status_label.setStyleSheet("color:rgba(192,192,192,0.5);font-size:12px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        cam.addWidget(self.status_label)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.verify_btn = QPushButton("📸  GIVE ATTENDANCE")
        self.verify_btn.setObjectName("primary")
        self.verify_btn.clicked.connect(self._manual_verify)
        btn_row.addWidget(self.verify_btn)

        self.auto_btn = QPushButton("🔄  AUTO OFF")
        self.auto_btn.setObjectName("toggle_off")
        self.auto_btn.clicked.connect(self._toggle_auto)
        self.auto_btn.setFixedWidth(130)
        btn_row.addWidget(self.auto_btn)
        cam.addLayout(btn_row)
        cl.addLayout(cam)

        # ── Result side ──
        result = QVBoxLayout(); result.setSpacing(12)

        self.id_card = QFrame(); self.id_card.setObjectName("id_card"); self.id_card.hide()
        idl = QVBoxLayout(self.id_card); idl.setSpacing(6)

        self.r_name = QLabel("—")
        self.r_name.setStyleSheet("color:#d4af37;font-size:20px;font-weight:bold;")
        idl.addWidget(self.r_name)

        br = QHBoxLayout()
        self.r_id     = QLabel("ID: —");   self.r_id.setObjectName("status_gold")
        self.r_dept   = QLabel("Dept: —"); self.r_dept.setObjectName("status_gold")
        self.r_status = QLabel("—");       self.r_status.setObjectName("status_green")
        br.addWidget(self.r_id); br.addWidget(self.r_dept); br.addWidget(self.r_status); br.addStretch()
        idl.addLayout(br)

        tl = QLabel("RECORDED AT")
        tl.setStyleSheet("color:rgba(212,175,55,0.5);font-size:10px;letter-spacing:2px;")
        idl.addWidget(tl)
        self.r_time = QLabel("—")
        self.r_time.setStyleSheet("color:white;font-size:14px;font-weight:bold;")
        idl.addWidget(self.r_time)
        result.addWidget(self.id_card)

        self.history_title = QLabel("ATTENDANCE HISTORY")
        self.history_title.setStyleSheet("color:rgba(212,175,55,0.6);font-size:11px;letter-spacing:2px;")
        self.history_title.hide()
        result.addWidget(self.history_title)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMaximumHeight(190)
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget); self.history_layout.setSpacing(4)
        scroll.setWidget(self.history_widget)
        self.history_scroll = scroll; self.history_scroll.hide()
        result.addWidget(self.history_scroll)

        self.no_result = QLabel("Verify a face to see results.")
        self.no_result.setStyleSheet("color:rgba(192,192,192,0.3);font-size:13px;")
        self.no_result.setAlignment(Qt.AlignCenter)
        result.addWidget(self.no_result)

        result.addStretch()
        cl.addLayout(result)
        main.addWidget(card)

    def _connect(self):
        self.signals.frame_ready.connect(lambda f: self.cam_label.setPixmap(frame_to_pixmap(f)))
        self.signals.verify_result.connect(self._show_result)

    def showEvent(self, event):
        self._start_camera()

    def hideEvent(self, event):
        self.auto_mode = False
        self._stop_camera()

    def _start_camera(self):
        self.vs = cv2.VideoCapture(get_camera_index(), cv2.CAP_DSHOW)
        self.timer.start(30)

    def _stop_camera(self):
        self.timer.stop()
        if self.vs:
            self.vs.release()
            self.vs = None

    def _read_frame(self):
        if not self.vs:
            return
        ret, frame = self.vs.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        # draw guide ellipse
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 200, 100) if self.auto_mode else (212, 175, 55)
        cv2.ellipse(frame, (cx, cy), (int(w*.22), int(h*.38)), 0, 0, 360, color, 1)

        self.signals.frame_ready.emit(frame)

        # Auto mode logic
        if self.auto_mode and not self.verifying:
            if self.auto_cooldown > 0:
                self.auto_cooldown -= 1
            else:
                self._trigger_verify(frame.copy())

    def _toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_btn.setText("🔄  AUTO ON")
            self.auto_btn.setObjectName("toggle_on")
            self.auto_cooldown = 0
            self.status_label.setText("Auto mode ON — look at camera")
        else:
            self.auto_btn.setText("🔄  AUTO OFF")
            self.auto_btn.setObjectName("toggle_off")
            self.status_label.setText("Position your face and click verify.")
        # force style refresh
        self.auto_btn.style().unpolish(self.auto_btn)
        self.auto_btn.style().polish(self.auto_btn)

    def _manual_verify(self):
        if self.verifying or not self.vs:
            return
        ret, frame = self.vs.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self._trigger_verify(frame.copy())

    def _trigger_verify(self, frame):
        if self.verifying:
            return
        self.verifying = True
        self.verify_btn.setEnabled(False)
        self.status_label.setText("Verifying...")
        threading.Thread(target=self._verify_thread, args=(frame,), daemon=True).start()

    def _verify_thread(self, frame):
        try:
            conf      = load_config()
            threshold = conf.get("confidence_threshold", 0.6)

            if not os.path.exists(conf.get("recognizer_path", "")):
                self.signals.verify_result.emit({
                    "recognized": False,
                    "message": "Model not trained yet. Enroll 2+ people first."
                })
                return

            fa    = get_insight_app()
            faces = fa.get(frame)

            if not faces:
                self.signals.verify_result.emit({"recognized": False, "message": "No face detected."})
                return

            with open(conf["recognizer_path"], "rb") as f:
                recognizer = pickle.load(f)
            with open(conf["le_path"], "rb") as f:
                le = pickle.load(f)

            users = load_users()
            face  = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb   = face.embedding.reshape(1, -1)
            proba = recognizer.predict_proba(emb)[0]
            max_p = float(np.max(proba))
            pred  = int(np.argmax(proba))

            if max_p < threshold:
                self.signals.verify_result.emit({
                    "recognized": False,
                    "message": f"Not recognized ({max_p:.0%} confidence)"
                })
                return

            user_id   = le.classes_[pred]
            user_info = users["users"].get(user_id, {})
            name      = user_info.get("name", user_id)
            dept      = user_info.get("dept", "N/A")

            att_data     = load_attendance()
            today        = get_today_str()
            already_today = False

            if user_id in att_data["attendance"]:
                last = att_data["attendance"][user_id]
                if isinstance(last, list): last = last[-1]
                already_today = last["date_time"].startswith(today)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not already_today:
                record = {"name": name, "dept": dept, "date_time": now_str}
                if user_id not in att_data["attendance"]:
                    att_data["attendance"][user_id] = []
                if isinstance(att_data["attendance"][user_id], dict):
                    att_data["attendance"][user_id] = [att_data["attendance"][user_id]]
                att_data["attendance"][user_id].append(record)
                save_attendance(att_data)

            history = att_data["attendance"].get(user_id, [])
            if isinstance(history, dict): history = [history]

            self.signals.verify_result.emit({
                "recognized":    True,
                "user_id":       user_id,
                "name":          name,
                "dept":          dept,
                "time":          now_str,
                "already_today": already_today,
                "history":       history,
                "confidence":    f"{max_p:.0%}",
                "total_days":    len(history),
            })

        except Exception as e:
            logging.exception("Verify thread error")
            self.signals.verify_result.emit({"recognized": False, "message": str(e)})

    def _show_result(self, data):
        self.verifying = False
        self.verify_btn.setEnabled(True)

        # auto mode: set cooldown (~5 seconds = ~165 frames at 30fps)
        if self.auto_mode:
            self.auto_cooldown = 165

        if not data.get("recognized"):
            self.status_label.setText(f"❌ {data.get('message', 'Unknown error')}")
            self.id_card.hide(); self.history_title.hide()
            self.history_scroll.hide(); self.no_result.show()
            return

        self.no_result.hide(); self.id_card.show()
        self.r_name.setText(data["name"])
        self.r_id.setText(f"ID: {data['user_id']}")
        self.r_dept.setText(f"Dept: {data['dept']}")
        self.r_time.setText(f"{data['time']}  •  {data.get('confidence','')} confidence")

        if data.get("already_today"):
            self.r_status.setText("Already Recorded")
            self.r_status.setStyleSheet("background:rgba(255,165,0,0.1);border:1px solid rgba(255,165,0,0.3);color:orange;border-radius:15px;padding:6px 16px;font-size:11px;letter-spacing:2px;")
            self.status_label.setText(f"Welcome back, {data['name']}! 👋")
        else:
            self.r_status.setText("✓ Recorded")
            self.r_status.setStyleSheet("background:rgba(0,200,100,0.1);border:1px solid rgba(0,200,100,0.3);color:#00c864;border-radius:15px;padding:6px 16px;font-size:11px;letter-spacing:2px;")
            self.status_label.setText(f"✅ Attendance recorded for {data['name']}!")

        history = data.get("history", [])
        total   = data.get("total_days", len(history))
        if history:
            self.history_title.setText(f"ATTENDANCE HISTORY  ({total} day(s) total)")
            self.history_title.show(); self.history_scroll.show()

            for i in reversed(range(self.history_layout.count())):
                w = self.history_layout.itemAt(i).widget()
                if w: w.deleteLater()

            for record in reversed(history):
                row = QFrame()
                row.setStyleSheet("background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;")
                rl  = QHBoxLayout(row); rl.setContentsMargins(12, 6, 12, 6)
                parts = record["date_time"].split(" ")
                dl = QLabel(parts[0]); dl.setStyleSheet("color:rgba(192,192,192,0.5);font-size:12px;")
                tl = QLabel(parts[1] if len(parts)>1 else ""); tl.setStyleSheet("color:#d4af37;font-size:12px;font-weight:bold;")
                rl.addWidget(dl); rl.addStretch(); rl.addWidget(tl)
                self.history_layout.addWidget(row)

    def _go_back(self):
        self.auto_mode = False
        self._stop_camera()
        self.parent.show_page("home")

# ═════════════════════════════════════════════════════════════════════════════
# History Window
# ═════════════════════════════════════════════════════════════════════════════
class HistoryWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent   = parent
        self._all_rows = []
        self._build()

    def _build(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(30, 20, 30, 20)
        main.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(back_btn(lambda: self.parent.show_page("home")))
        top.addStretch()
        exp_btn = QPushButton("⬇  EXPORT CSV")
        exp_btn.setObjectName("secondary")
        exp_btn.setFixedSize(150, 38)
        exp_btn.clicked.connect(self._export)
        top.addWidget(exp_btn)
        main.addLayout(top)

        main.addLayout(page_header("Attendance History", "ALL RECORDS — NEWEST FIRST"))

        # ── Filters ──
        fil = QHBoxLayout(); fil.setSpacing(10)

        self.search = QLineEdit(); self.search.setPlaceholderText("🔍  Search name or ID...")
        self.search.textChanged.connect(self._filter)
        fil.addWidget(self.search)

        self.dept_filter = QComboBox()
        self.dept_filter.addItem("All Departments")
        for d in DEPTS: self.dept_filter.addItem(d)
        self.dept_filter.currentTextChanged.connect(self._filter)
        self.dept_filter.setFixedWidth(180)
        fil.addWidget(self.dept_filter)

        date_lbl = QLabel("From:")
        date_lbl.setStyleSheet("color:rgba(212,175,55,0.6);font-size:12px;")
        fil.addWidget(date_lbl)

        self.date_from = QDateEdit(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        self.date_from.setFixedWidth(130)
        self.date_from.dateChanged.connect(self._filter)
        fil.addWidget(self.date_from)

        date_lbl2 = QLabel("To:")
        date_lbl2.setStyleSheet("color:rgba(212,175,55,0.6);font-size:12px;")
        fil.addWidget(date_lbl2)

        self.date_to = QDateEdit(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setFixedWidth(130)
        self.date_to.dateChanged.connect(self._filter)
        fil.addWidget(self.date_to)

        reset_btn = QPushButton("✕ Reset")
        reset_btn.setObjectName("back")
        reset_btn.clicked.connect(self._reset_filters)
        fil.addWidget(reset_btn)

        main.addLayout(fil)

        # ── Table ──
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["STUDENT ID", "NAME", "DEPARTMENT", "DATE", "TIME"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        main.addWidget(self.table)

        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet("color:rgba(192,192,192,0.4);font-size:11px;")
        self.count_lbl.setAlignment(Qt.AlignRight)
        main.addWidget(self.count_lbl)

    def showEvent(self, event):
        self._all_rows = get_all_attendance_flat()
        self._filter()

    def _reset_filters(self):
        self.search.clear()
        self.dept_filter.setCurrentIndex(0)
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_to.setDate(QDate.currentDate())

    def _filter(self):
        query  = self.search.text().strip().lower()
        dept   = self.dept_filter.currentText()
        d_from = self.date_from.date().toString("yyyy-MM-dd")
        d_to   = self.date_to.date().toString("yyyy-MM-dd")

        result = []
        for r in self._all_rows:
            date_str = r["datetime"][:10]
            if dept != "All Departments" and r["dept"] != dept: continue
            if query and query not in r["name"].lower() and query not in r["id"].lower(): continue
            if date_str < d_from or date_str > d_to: continue
            result.append(r)

        self.table.setRowCount(len(result))
        for i, r in enumerate(result):
            parts = r["datetime"].split(" ")
            for j, val in enumerate([r["id"], r["name"], r["dept"],
                                      parts[0] if parts else "",
                                      parts[1] if len(parts)>1 else ""]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

        self.count_lbl.setText(f"{len(result)} record(s)")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)")
        if not path: return
        count = export_attendance_csv(path)
        QMessageBox.information(self, "Done", f"✅ Exported {count} records to:\n{path}")

# ═════════════════════════════════════════════════════════════════════════════
# Students Window
# ═════════════════════════════════════════════════════════════════════════════
class StudentsWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent    = parent
        self._all_rows = []
        self._signals  = Signals()
        self._build()
        self._signals.status_update.connect(self._on_status)

    def _build(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(30, 20, 30, 20)
        main.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(back_btn(lambda: self.parent.show_page("home")))
        top.addStretch()
        del_btn = QPushButton("🗑  DELETE SELECTED")
        del_btn.setObjectName("danger")
        del_btn.setFixedSize(190, 38)
        del_btn.clicked.connect(self._delete_selected)
        top.addWidget(del_btn)
        main.addLayout(top)

        main.addLayout(page_header("Enrolled Students", "MANAGE REGISTERED IDENTITIES"))

        self.search = QLineEdit(); self.search.setPlaceholderText("🔍  Search name or ID...")
        self.search.textChanged.connect(self._filter)
        main.addWidget(self.search)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["STUDENT ID", "NAME", "DEPARTMENT", "ENROLLED AT", "TOTAL DAYS", "LAST SEEN"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        main.addWidget(self.table)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:rgba(192,192,192,0.4);font-size:11px;")
        self.status_lbl.setAlignment(Qt.AlignRight)
        main.addWidget(self.status_lbl)

    def showEvent(self, event):
        self._load()

    def _load(self):
        users = load_users()
        att   = load_attendance()
        self._all_rows = []

        for uid, info in users["users"].items():
            records = att["attendance"].get(uid, [])
            if isinstance(records, dict): records = [records]
            total = len(records)
            last  = records[-1]["date_time"][:10] if records else "—"
            self._all_rows.append({
                "id":          uid,
                "name":        info.get("name", uid),
                "dept":        info.get("dept", "N/A"),
                "enrolled_at": info.get("enrolled_at", "—"),
                "total_days":  str(total),
                "last_seen":   last,
            })

        self._filter()

    def _filter(self):
        q = self.search.text().strip().lower()
        rows = [r for r in self._all_rows
                if not q or q in r["name"].lower() or q in r["id"].lower()]
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, val in enumerate([r["id"], r["name"], r["dept"],
                                      r["enrolled_at"], r["total_days"], r["last_seen"]]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 4:  # total days gold
                    item.setForeground(QColor("#d4af37"))
                if j == 5 and val == get_today_str():  # present today green
                    item.setForeground(QColor("#00c864"))
                self.table.setItem(i, j, item)

        self.status_lbl.setText(f"{len(rows)} student(s) enrolled")

    def _delete_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a student row first.")
            return

        uid  = self.table.item(row, 0).text()
        name = self.table.item(row, 1).text()

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}' ({uid})?\n\nThis removes their dataset, embeddings & attendance history.",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return

        self.status_lbl.setText("Deleting & retraining... please wait")
        QApplication.processEvents()

        threading.Thread(
            target=self._delete_thread,
            args=(uid, name),
            daemon=True
        ).start()

    def _delete_thread(self, uid, name):
        try:
            # Dataset
            d = os.path.join(DATASET_PATH, uid)
            if os.path.exists(d): shutil.rmtree(d)

            # users.json
            users = load_users()
            users["users"].pop(uid, None)
            save_users(users)

            # attendance.json
            att = load_attendance()
            att["attendance"].pop(uid, None)
            save_attendance(att)

            # Rebuild embeddings & retrain
            conf = load_config()
            if os.path.exists(conf.get("encodings_path", "")):
                with open(conf["encodings_path"], "rb") as f:
                    data = pickle.load(f)
                embs  = [e for e, n in zip(data["embeddings"], data["names"]) if n != uid]
                names = [n for n in data["names"] if n != uid]
                retrain_svm(conf, embs, names)

            self._signals.status_update.emit(f"✅ '{name}' deleted successfully.")

        except Exception as e:
            logging.exception("Delete thread error")
            self._signals.status_update.emit(f"❌ Error: {e}")

    def _on_status(self, msg):
        self._load()
        self.status_lbl.setText(msg)

# ═════════════════════════════════════════════════════════════════════════════
# Backup Window
# ═════════════════════════════════════════════════════════════════════════════
class BackupWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._build()

    def _build(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(40, 30, 40, 30)
        main.setSpacing(20)
        main.setAlignment(Qt.AlignTop)

        top = QHBoxLayout()
        top.addWidget(back_btn(lambda: self.parent.show_page("home")))
        top.addStretch()
        main.addLayout(top)

        main.addLayout(page_header("Backup & Restore", "DATA MANAGEMENT"))

        card = QFrame(); card.setObjectName("card")
        cl   = QVBoxLayout(card)
        cl.setContentsMargins(30, 25, 30, 25)
        cl.setSpacing(16)

        # Backup section
        bk_title = QLabel("📦  BACKUP")
        bk_title.setStyleSheet("color:#d4af37;font-size:15px;font-weight:bold;")
        cl.addWidget(bk_title)

        bk_desc = QLabel(
            "Creates a .zip archive of users.json, attendance.json, and the trained model files.\n"
            "Use this to transfer data to another machine or keep a safe copy."
        )
        bk_desc.setStyleSheet("color:rgba(255,255,255,0.5);font-size:12px;")
        bk_desc.setWordWrap(True)
        cl.addWidget(bk_desc)

        bk_btn = QPushButton("💾  CREATE BACKUP")
        bk_btn.setObjectName("primary")
        bk_btn.setFixedWidth(220)
        bk_btn.clicked.connect(self._do_backup)
        cl.addWidget(bk_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:rgba(212,175,55,0.1);")
        cl.addWidget(sep)

        # Reset section
        rst_title = QLabel("⚠️  RESET DATA")
        rst_title.setStyleSheet("color:#ff3c3c;font-size:15px;font-weight:bold;")
        cl.addWidget(rst_title)

        rst_desc = QLabel(
            "Clears ALL attendance records. Does not delete enrolled students or the trained model.\n"
            "Useful for starting a new semester."
        )
        rst_desc.setStyleSheet("color:rgba(255,255,255,0.5);font-size:12px;")
        rst_desc.setWordWrap(True)
        cl.addWidget(rst_desc)

        rst_btn = QPushButton("🗑  CLEAR ATTENDANCE RECORDS")
        rst_btn.setObjectName("danger")
        rst_btn.setFixedWidth(260)
        rst_btn.clicked.connect(self._reset_attendance)
        cl.addWidget(rst_btn)

        main.addWidget(card)

        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignCenter)
        self.msg.setWordWrap(True)
        main.addWidget(self.msg)

    def _do_backup(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Backup",
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "ZIP Files (*.zip)")
        if not path: return
        try:
            backup_data(path)
            self.msg.setText(f"✅ Backup saved to:\n{path}")
            self.msg.setStyleSheet("color:#00c864;font-size:13px;")
        except Exception as e:
            self.msg.setText(f"❌ Backup failed: {e}")
            self.msg.setStyleSheet("color:#ff3c3c;font-size:13px;")

    def _reset_attendance(self):
        att   = load_attendance()
        count = sum(
            len(r) if isinstance(r, list) else 1
            for r in att["attendance"].values()
        )
        reply = QMessageBox.question(
            self, "Confirm Reset",
            f"This will permanently delete {count} attendance record(s).\n\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        save_attendance({"attendance": {}})
        self.msg.setText(f"✅ {count} attendance records cleared.")
        self.msg.setStyleSheet("color:#00c864;font-size:13px;")

# ═════════════════════════════════════════════════════════════════════════════
# Main Window
# ═════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Face Attendance System")
        self.setMinimumSize(980, 660)
        self.setStyleSheet(STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.pages = {
            "home":       HomeWindow(self),
            "enroll":     EnrollWindow(self),
            "attendance": AttendanceWindow(self),
            "history":    HistoryWindow(self),
            "students":   StudentsWindow(self),
            "backup":     BackupWindow(self),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self._page_index = {name: i for i, name in enumerate(self.pages)}
        self.show_page("home")

    def show_page(self, name):
        self.stack.setCurrentIndex(self._page_index[name])

    def closeEvent(self, event):
        # Safely release any open cameras
        for name in ["enroll", "attendance"]:
            page = self.pages[name]
            try:
                page._stop_camera()
            except Exception:
                pass
        event.accept()

# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())