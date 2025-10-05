import os
import cv2
import time
import argparse
import threading
import math
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template_string, send_file, redirect, url_for
from ultralytics import YOLO
import easyocr

# ---------------- CONFIG ----------------
# Model to use for general object detection (ultralytics will download if missing)
YOLO_MODEL = "yolov8n.pt"  # nano model for speed

# Detection classes to treat as vehicles
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}  # COCO ids

# Output folders
EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)
LOG_CSV = Path("violations_log.csv")
OUTPUT_VIDEO = Path("output.mp4")

# Line coordinates for violation detection (x1,y1,x2,y2) -- default is center horizontal line
STOP_LINE = None  # example: (100, 400, 1180, 400)

# Minimum area to consider (filter tiny detections)
MIN_VEHICLE_AREA = 500

# IOU or distance threshold for simple tracking
MAX_TRACK_DISTANCE = 60

# OCR reader (EasyOCR)
OCR_LANGS = ['en']
OCR_GPU = False  # set True if you have GPU support
# ----------------------------------------

# Simple centroid tracker
class CentroidTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE):
        self.next_id = 1
        self.objects = {}  # id -> centroid
        self.last_seen = {}  # id -> last_time
        self.tracks = {}  # id -> list of centroids
        self.max_distance = max_distance

    def update(self, detections):
        """detections: list of centroids [(x,y), ...]
        returns dict of id -> centroid
        """
        assigned = {}
        now = time.time()
        if not self.objects:
            for c in detections:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = now
                self.tracks[oid] = [c]
                assigned[oid] = c
            return assigned

        # For each detection, find nearest existing object
        used_ids = set()
        for c in detections:
            best_id = None
            best_dist = None
            for oid, cent in self.objects.items():
                if oid in used_ids:
                    continue
                dist = math.hypot(c[0] - cent[0], c[1] - cent[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = oid
            if best_dist is not None and best_dist <= self.max_distance:
                # assign
                self.objects[best_id] = c
                self.last_seen[best_id] = now
                self.tracks[best_id].append(c)
                assigned[best_id] = c
                used_ids.add(best_id)
            else:
                # new object
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = now
                self.tracks[oid] = [c]
                assigned[oid] = c
                used_ids.add(oid)

        # remove stale objects (not seen for a while)
        stale = []
        for oid, t in self.last_seen.items():
            if now - t > 5.0:
                stale.append(oid)
        for oid in stale:
            self.objects.pop(oid, None)
            self.last_seen.pop(oid, None)
            self.tracks.pop(oid, None)

        return assigned


# Utility functions

def draw_line(img, line, color=(0, 0, 255), thickness=2):
    (x1, y1, x2, y2) = map(int, line)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def point_side_of_line(pt, line):
    (x1, y1, x2, y2) = line
    x, y = pt
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)


def save_evidence(img, prefix="violation"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}_{ts}.jpg"
    path = EVIDENCE_DIR / fname
    cv2.imwrite(str(path), img)
    return str(path)


def append_log(row: dict):
    df = pd.DataFrame([row])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, mode='w', header=True, index=False)


# Main Detector + server
class TrafficDetector:
    def __init__(self, src=0, stop_line=None, save_output=True, output_path=OUTPUT_VIDEO):
        self.src = 0 if str(src).isdigit() and int(src) == 0 else src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source {src}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS) or 20))
        self.stop_line = stop_line or (0, int(self.height*0.6), self.width, int(self.height*0.6))
        self.model = YOLO(YOLO_MODEL)
        self.ocr = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)
        self.tracker = CentroidTracker()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.violated_ids = set()
        self.save_output = save_output
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(script_dir, "output.avi")
        # setup video writer if required
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (self.width, self.height))
        else:
            self.writer = None

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass
        if self.writer is not None:
            self.writer.release()

    def process_loop(self):
        print("Starting detection loop...")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.width, self.height))
            if self.save_output and self.writer is not None:
                self.writer.write(frame)
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            img = frame.copy()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model.predict(rgb, imgsz=640, conf=0.35, verbose=False)
            r = results[0]
            boxes = []
            class_ids = []
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls in VEHICLE_CLASSES:
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_VEHICLE_AREA:
                        continue
                    boxes.append((x1, y1, x2, y2))
                    class_ids.append(cls)

            centroids = [centroid_from_bbox(b) for b in boxes]
            assigned = self.tracker.update(centroids)

            cent_to_info = {centroid_from_bbox(b): (b, cls) for b, cls in zip(boxes, class_ids)}

            for oid, cent in assigned.items():
                info = cent_to_info.get(cent)
                if info is None:
                    min_dist = None
                    found = None
                    for b, cls in zip(boxes, class_ids):
                        c = centroid_from_bbox(b)
                        d = math.hypot(c[0] - cent[0], c[1] - cent[1])
                        if min_dist is None or d < min_dist:
                            min_dist = d
                            found = (b, cls)
                    if found:
                        bbox, cls = found
                    else:
                        continue
                else:
                    bbox, cls = info

                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{VEHICLE_CLASSES.get(cls)} ID:{oid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                track = self.tracker.tracks.get(oid, [])
                if len(track) >= 2:
                    prev = track[-2]
                    curr = track[-1]
                    side_prev = point_side_of_line(prev, self.stop_line)
                    side_curr = point_side_of_line(curr, self.stop_line)
                    if side_prev != 0 and side_curr != 0 and (side_prev * side_curr) < 0:
                        if oid not in self.violated_ids:
                            pad = 20
                            xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad); xx2 = min(self.width-1, x2 + pad); yy2 = min(self.height-1, y2 + pad)
                            crop = frame[yy1:yy2, xx1:xx2]
                            plate_text = None
                            try:
                                ocr_res = self.ocr.readtext(crop[int(crop.shape[0]*0.5):, :])
                                texts = [t[1] for t in ocr_res if len(t[1]) >= 4]
                                if texts:
                                    plate_text = sorted(texts, key=lambda s: len(s), reverse=True)[0]
                            except Exception:
                                plate_text = None

                            img_path = save_evidence(crop, prefix=f"cross_{VEHICLE_CLASSES.get(cls)}")
                            logrow = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "object_id": oid,
                                "class": VEHICLE_CLASSES.get(cls),
                                "plate": plate_text or "",
                                "image": img_path,
                                "reason": "line_crossing"
                            }
                            append_log(logrow)
                            print("Logged violation:", logrow)
                            self.violated_ids.add(oid)

            draw_line(img, self.stop_line, color=(0,0,255), thickness=2)

            # write to output video if enabled
            if self.writer is not None:
                self.writer.write(img)

            with self.lock:
                self.frame = img.copy()

        print("Detector stopped.")

    def get_frame_bytes(self):
        with self.lock:
            if self.frame is None:
                blank = np.zeros((480,640,3), dtype=np.uint8)
                ret, jpg = cv2.imencode('.jpg', blank)
                return jpg.tobytes()
            ret, jpg = cv2.imencode('.jpg', self.frame)
            return jpg.tobytes()


# Flask app
app = Flask(__name__)
_detector = None

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Violation - Demo</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            padding: 12px;
        }
        h3 {
            margin-bottom: 6px;
        }
        img {
            width: 300px;
            height: auto;
            border-radius: 5px;
        }
        hr {
            margin: 12px 0;
        }
    </style>
</head>
<body>
    <h2>Traffic Violation Detection - Demo</h2>
    <div>
        <h3>Live Stream</h3>
        <img src="{{ url_for('video_feed') }}" alt="Live Stream">
    </div>
    <hr>
    <h3>Recent Violations</h3>
    {% if violationsdf %}
        {% for row in violationsdf %}
            {% if row.plate != 'NA' %}
                <h3>{{ row.timestamp }} - {{ row.class }} - {{ row.plate }}</h3>
            {% else %}
                <h3>{{ row.timestamp }} - {{ row.class }} - N/A</h3>
            {% endif %}
            <img src="{{ row.image }}" alt="Evidence">
            <hr>
        {% endfor %}
    {% else %}
        <p>No violations yet.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
@app.route('/')
def index():
    violationsdf = []
    if os.path.exists(LOG_CSV):
        try:
            df = pd.read_csv(LOG_CSV)
            df = df.sort_values('timestamp', ascending=False)
            for _, row in df.iterrows():
                violationsdf.append({
                    'timestamp': row.get('timestamp', ''),
                    'class': row.get('class', ''),
                    'plate': row.get('plate', ''),
                    'image': row.get('image', '')
                })
        except Exception as e:
            print("Error reading CSV:", e)
    return render_template_string(TEMPLATE, violationsdf=violationsdf)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            b = _detector.get_frame_bytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n')
            time.sleep(0.05)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/evidence/<fname>')
def serve_evidence(fname):
    path = EVIDENCE_DIR / fname
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype='image/jpeg')

@app.route('/clear')
def clear():
    if LOG_CSV.exists():
        LOG_CSV.unlink()
    for f in EVIDENCE_DIR.iterdir():
        try:
            f.unlink()
        except:
            pass
    # remove output video if exists
    if OUTPUT_VIDEO.exists():
        try:
            OUTPUT_VIDEO.unlink()
        except:
            pass
    return redirect(url_for('index'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default="C:/Users/user/OneDrive/Desktop/hackthon/traffic.mp4", help='video source: 0 for webcam or path to file')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--set-line', action='store_true', help='Interactively set stop line by clicking two points on a frame')
    parser.add_argument('--no-output', action='store_true', help='Disable saving processed output video')
    args = parser.parse_args()

    src = args.source
    if args.set_line:
        cap = cv2.VideoCapture(0 if str(src).isdigit() and int(src) == 0 else src)
        ret, frame = cap.read()
        if not ret:
            print('Failed to read from source for line setting')
        else:
            pts = []
            clone = frame.copy()
            def click(event,x,y,flags,param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    pts.append((x,y))
                    cv2.circle(clone, (x,y), 5, (0,0,255), -1)
                    cv2.imshow('Set STOP line - click two points', clone)
            cv2.imshow('Set STOP line - click two points', clone)
            cv2.setMouseCallback('Set STOP line - click two points', click)
            print('Click two points on the window to set the stop line. Press q when done.')
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or len(pts) >= 2:
                    break
            cv2.destroyAllWindows()
            if len(pts) >= 2:
                STOP_LINE = (pts[0][0], pts[0][1], pts[1][0], pts[1][1])
                print('Set STOP_LINE to', STOP_LINE)
        cap.release()

    detector = TrafficDetector(src=src, stop_line=STOP_LINE, save_output=not args.no_output)
    _detector = detector
    t = threading.Thread(target=detector.process_loop, daemon=True)
    t.start()

    # start flask
    app.run(host=args.host, port=args.port, threaded=True)
