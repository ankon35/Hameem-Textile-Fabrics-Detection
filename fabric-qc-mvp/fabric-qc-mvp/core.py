"""Core logic for the FabricEye QC demo.

Detection (YOLO model or a classical computer vision fallback),
four point system grading, demo data generation and cost model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASSES = ["Hole", "Oil", "Crack", "Cutting"]
COLORS = {  # BGR
    "Hole": (60, 60, 230),
    "Oil": (30, 160, 240),
    "Crack": (220, 120, 40),
    "Cutting": (180, 60, 200),
}


@dataclass
class Defect:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length_px(self) -> int:
        return max(self.x2 - self.x1, self.y2 - self.y1)


# Detection

def load_model(path: str | Path):
    """Load a trained YOLO model. Returns None if the file or library is missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(path))
    except Exception:
        return None


def detect_yolo(model, img_bgr: np.ndarray, conf: float = 0.25) -> list[Defect]:
    res = model.predict(img_bgr, conf=conf, verbose=False)[0]
    names = res.names
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append(Defect(str(names[int(box.cls)]).capitalize(), float(box.conf), x1, y1, x2, y2))
    return out


def detect_classical(img_bgr: np.ndarray, sensitivity: float = 0.5) -> list[Defect]:
    """Texture anomaly detector used when no trained model is available.

    Removes the regular weave pattern by smoothing, compares each pixel with the
    local background and keeps regions that differ strongly. Shape and colour
    decide the defect label.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.5)
    background = cv2.GaussianBlur(gray, (0, 0), max(h, w) / 20)
    diff = np.abs(smooth - background)

    # colour deviation catches stains on similar brightness fabric
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.GaussianBlur(lab, (0, 0), 2.5)
    lab_bg = cv2.GaussianBlur(lab, (0, 0), max(h, w) / 20)
    cdiff = np.linalg.norm((lab_s - lab_bg)[:, :, 1:], axis=2)
    score = diff + 1.5 * cdiff

    k = 6.0 - 4.0 * float(np.clip(sensitivity, 0, 1))  # sensitivity 1 means threshold mean+2std
    thr = score.mean() + k * score.std()
    mask = (score > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.0004
    defects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        region = mask[y:y + bh, x:x + bw] > 0
        bg_gray = float(background[y:y + bh, x:x + bw].mean())
        dark_ratio = float(np.percentile(gray[y:y + bh, x:x + bw][region], 10)) / max(1.0, bg_gray)

        if aspect > 4:
            label = "Crack"          # long thin line: broken end, missing pick
        elif dark_ratio < 0.4:       # fabric missing, background shows through
            label = "Cutting" if (aspect > 1.6 or area > (h * w) * 0.012) else "Hole"
        else:
            label = "Oil"            # discoloured area, fabric still present

        strength = float(score[y:y + bh, x:x + bw][region].mean() / (thr + 1e-6))
        conf = float(np.clip(0.45 + 0.25 * (strength - 1), 0.3, 0.97))
        pad = 4
        defects.append(Defect(label, round(conf, 2), max(0, x - pad), max(0, y - pad),
                              min(w, x + bw + pad), min(h, y + bh + pad)))
    return defects


def draw_defects(img_bgr: np.ndarray, defects: list[Defect]) -> np.ndarray:
    out = img_bgr.copy()
    t = max(2, out.shape[1] // 300)
    for d in defects:
        col = COLORS.get(d.label, (0, 255, 0))
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), col, t)
        text = f"{d.label} {d.confidence:.2f}"
        fs = max(0.5, out.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = max(th + 6, d.y1)
        cv2.rectangle(out, (d.x1, y0 - th - 6), (d.x1 + tw + 6, y0), col, -1)
        cv2.putText(out, text, (d.x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# Four point system grading (ASTM D5430 style)

def points_for_defect(label: str, length_in: float) -> int:
    if label in ("Hole", "Cutting"):
        return 4  # any hole or cut gets the maximum
    if length_in <= 3:
        return 1
    if length_in <= 6:
        return 2
    if length_in <= 9:
        return 3
    return 4


def defects_table(defects: list[Defect], px_per_inch: float, frame_no: int = 1,
                  frame_offset_m: float = 0.0, frame_len_m: float = 1.0, img_h: int = 1) -> pd.DataFrame:
    rows = []
    for d in defects:
        length_in = d.length_px / px_per_inch
        cy = (d.y1 + d.y2) / 2
        rows.append({
            **asdict(d),
            "frame": frame_no,
            "length_in": round(length_in, 2),
            "points": points_for_defect(d.label, length_in),
            "roll_position_m": round(frame_offset_m + frame_len_m * cy / max(1, img_h), 2),
            "width_position_in": round(((d.x1 + d.x2) / 2) / px_per_inch, 1),
        })
    cols = ["frame", "label", "confidence", "length_in", "points", "roll_position_m",
            "width_position_in", "x1", "y1", "x2", "y2"]
    df = pd.DataFrame(rows, columns=cols)
    num = {"frame": int, "confidence": float, "length_in": float, "points": int, "roll_position_m": float,
           "width_position_in": float, "x1": int, "y1": int, "x2": int, "y2": int}
    return df.astype(num)


def points_per_100_sq_yd(total_points: float, length_m: float, width_in: float) -> float:
    length_yd = length_m * 1.09361
    if length_yd <= 0 or width_in <= 0:
        return 0.0
    return total_points * 3600 / (length_yd * width_in)


def roll_decision(ppsy: float, limit: float = 40) -> str:
    if ppsy <= limit * 0.5:
        return "Grade A (accept)"
    if ppsy <= limit:
        return "Grade B (accept, notify cutting)"
    return "Reject (hold roll, raise claim)"


# Synthetic fabric samples for offline demo

def synth_fabric(defect: str = "none", seed: int = 0, size: tuple[int, int] = (640, 640),
                 base=(150, 95, 40)) -> np.ndarray:
    """Draw a woven looking fabric (denim tone by default) with an optional defect."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(xx * 0.9) * np.sin(yy * 0.9) + 0.15 * np.sin((xx + yy) * 0.45)
    noise = rng.normal(0, 0.06, (h, w)).astype(np.float32)
    tex = np.clip(weave + noise, 0, 1)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[:, :, c] = base[c] * (0.7 + 0.6 * tex)
    img = np.clip(img, 0, 255).astype(np.uint8)

    cx, cy = int(rng.integers(w * 0.25, w * 0.75)), int(rng.integers(h * 0.25, h * 0.75))
    if defect == "Hole":
        r = int(rng.integers(12, 26))
        cv2.ellipse(img, (cx, cy), (r, int(r * 0.8)), float(rng.integers(0, 180)), 0, 360, (18, 14, 12), -1)
        cv2.ellipse(img, (cx, cy), (r + 3, int(r * 0.8) + 3), 0, 0, 360, (200, 190, 180), 2)
    elif defect == "Oil":
        layer = np.zeros((h, w), np.float32)
        for _ in range(4):
            cv2.circle(layer, (cx + int(rng.integers(-25, 25)), cy + int(rng.integers(-25, 25))),
                       int(rng.integers(18, 38)), 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 6)[:, :, None]
        stain = np.array([40, 70, 85], np.float32)
        img = (img * (1 - 0.7 * layer) + stain * 0.7 * layer).astype(np.uint8)
    elif defect == "Crack":
        if rng.random() < 0.5:
            y = cy
            cv2.line(img, (int(w * 0.1), y), (int(w * 0.9), y + int(rng.integers(-6, 6))), (25, 20, 15), 3)
        else:
            x = cx
            cv2.line(img, (x, int(h * 0.1)), (x + int(rng.integers(-6, 6)), int(h * 0.9)), (230, 220, 210), 3)
    elif defect == "Cutting":
        pts = np.array([[cx - 60, cy - 10], [cx + 70, cy - 25], [cx + 60, cy + 15], [cx - 55, cy + 25]], np.int32)
        cv2.fillPoly(img, [pts], (12, 10, 10))
    return img


# Demo factory history (simulated)

MACHINES = [f"Loom {i:02d}" for i in range(1, 13)]
SUPPLIERS = ["Yarn Supplier A", "Yarn Supplier B", "Yarn Supplier C"]
FABRICS = ["Denim 3/1 Twill", "Stretch Denim", "Cotton Twill", "Single Jersey"]


def simulate_history(days: int = 60, seed: int = 7) -> pd.DataFrame:
    """Simulated inspection log. Loom 07 on night shift and Supplier B are made worse on purpose
    so the dashboard has a clear root cause story to show."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-09-01")
    for d in range(days):
        date = start + pd.Timedelta(days=d)
        for _ in range(int(rng.integers(25, 40))):  # rolls per day
            machine = rng.choice(MACHINES)
            shift = rng.choice(["Day", "Night"])
            supplier = rng.choice(SUPPLIERS, p=[0.45, 0.30, 0.25])
            fabric = rng.choice(FABRICS, p=[0.4, 0.3, 0.2, 0.1])
            length_m = float(rng.integers(80, 120))
            lam = 14.0
            if machine == "Loom 07":
                lam *= 2.2 if shift == "Night" else 1.4
            if supplier == "Yarn Supplier B":
                lam *= 1.5
            n_def = int(rng.poisson(lam))
            type_p = [0.18, 0.32, 0.38, 0.12]
            if supplier == "Yarn Supplier B":
                type_p = [0.12, 0.25, 0.53, 0.10]  # more yarn related cracks
            types = rng.choice(CLASSES, size=n_def, p=type_p)
            pts = sum(points_for_defect(t, float(rng.gamma(2.0, 2.0))) for t in types)
            ppsy = points_per_100_sq_yd(pts, length_m, 58)
            rec = {"date": date, "machine": machine, "shift": shift, "supplier": supplier,
                   "fabric": fabric, "length_m": length_m, "defects": n_def, "points": pts,
                   "points_per_100_sq_yd": round(ppsy, 1), "rejected": ppsy > 40}
            for c in CLASSES:
                rec[c] = int((types == c).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


# Cost model

def cost_impact(meters_per_month: float, defects_per_1000m: float, manual_catch: float, ai_catch: float,
                cost_fabric_stage: float, mult_cut: float, mult_garment: float,
                share_escape_to_garment: float, inspectors_saved: int, inspector_salary: float,
                fabric_saved_pct: float, fabric_price_per_m: float) -> dict:
    """All money in BDT per month. Every input is an assumption the user can change."""
    defects = meters_per_month / 1000 * defects_per_1000m
    extra_caught = defects * max(0.0, ai_catch - manual_catch)
    # a defect that escapes inspection is found either at cutting or at garment/buyer stage
    avg_escape_cost = cost_fabric_stage * ((1 - share_escape_to_garment) * mult_cut
                                           + share_escape_to_garment * mult_garment)
    saving_escape = extra_caught * (avg_escape_cost - cost_fabric_stage)
    saving_labour = inspectors_saved * inspector_salary
    saving_fabric = meters_per_month * fabric_saved_pct / 100 * fabric_price_per_m
    total = saving_escape + saving_labour + saving_fabric
    return {
        "defects_per_month": defects,
        "extra_defects_caught": extra_caught,
        "saving_late_detection": saving_escape,
        "saving_labour": saving_labour,
        "saving_fabric_cutting": saving_fabric,
        "total_monthly": total,
    }


def bdt(x: float) -> str:
    """Format money the Bangladeshi way: 12,34,567."""
    neg = x < 0
    s = f"{abs(int(round(x)))}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "৳ " + s
