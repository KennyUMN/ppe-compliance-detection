"""
Real-Time PPE Compliance Detection & Video Recording
Powered by YOLOv9-c and Intersection over PPE Area (IoPPE) State Machine.
Includes Anatomical Zoning, Exclusive Bipartite Matching, and Executive HUD.
"""

import argparse
import cv2
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
from ultralytics import YOLO

# Anatomical vertical bounds relative to worker height [y_min_ratio, y_max_ratio]
PPE_ANATOMICAL_ZONES = {
    "helmet":          (0.00, 0.40),  # Upper 40% of worker box
    "face-mask":       (0.05, 0.35),
    "face-protection": (0.05, 0.35),
    "glasses":         (0.05, 0.35),
    "safety-vest":     (0.12, 0.88),  # Torso region
    "gloves":          (0.25, 0.95),  # Arms/hands
    "shoes":           (0.55, 1.05),  # Lower 45%
    "foot":            (0.55, 1.05),
}

def compute_ioppe(box_person, box_ppe):
    """
    Computes Intersection over PPE Area (IoPPE).
    IoPPE = Area(Intersection) / Area(PPE).
    """
    ix1 = max(box_person[0], box_ppe[0])
    iy1 = max(box_person[1], box_ppe[1])
    ix2 = min(box_person[2], box_ppe[2])
    iy2 = min(box_person[3], box_ppe[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_ppe = (box_ppe[2] - box_ppe[0]) * (box_ppe[3] - box_ppe[1])

    return inter / area_ppe if area_ppe > 0 else 0.0

def satisfies_anatomical_zone(box_person, box_ppe, class_name):
    """Verifies that the detected PPE is located within the proper human body section."""
    cname = class_name.lower()
    zone = None
    for k, v in PPE_ANATOMICAL_ZONES.items():
        if k in cname:
            zone = v
            break
    if zone is None:
        return True

    py1, py2 = box_person[1], box_person[3]
    ph = max(1.0, py2 - py1)

    ppe_cy = (box_ppe[1] + box_ppe[3]) / 2.0
    rel_y = (ppe_cy - py1) / ph

    return zone[0] <= rel_y <= zone[1]

def evaluate_compliance(detections, model_names, ioppe_threshold=0.30):
    """
    Maps detected PPE items to individual workers using IoPPE,
    anatomical height validation, and greedy exclusive matching.
    """
    persons = []
    ppe_items = []

    for d in detections:
        cls_name = model_names.get(d["cls_id"], f"cls_{d['cls_id']}").lower()
        if cls_name == "person":
            persons.append(d)
        else:
            ppe_items.append({**d, "name": cls_name})

    # Precompute candidate matches
    candidates = []
    for pi, p in enumerate(persons):
        pbox = p["box"]
        for ppei, ppe in enumerate(ppe_items):
            cname = ppe["name"]
            ppe_box = ppe["box"]
            score = compute_ioppe(pbox, ppe_box)
            if score >= ioppe_threshold and satisfies_anatomical_zone(pbox, ppe_box, cname):
                candidates.append((score, pi, ppei, cname))

    # Exclusive matching: each PPE box is assigned to at most ONE worker
    candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_ppe = set()
    worker_ppe_map = {pi: set() for pi in range(len(persons))}

    for score, pi, ppei, cname in candidates:
        if ppei not in assigned_ppe:
            assigned_ppe.add(ppei)
            worker_ppe_map[pi].add(cname)

    workers = []
    for pi, p in enumerate(persons):
        pbox = p["box"]
        worn_classes = worker_ppe_map[pi]

        has_helmet = any("helmet" in c for c in worn_classes)
        has_vest   = any("vest" in c for c in worn_classes)

        missing = []
        if not has_helmet: missing.append("Helmet")
        if not has_vest:   missing.append("Safety-Vest")

        if has_helmet and has_vest:
            state = "Fully Compliant"
            color = (46, 204, 113)     # Emerald Green (BGR: 113, 204, 46)
        elif has_helmet or has_vest:
            state = "Partially Compliant"
            color = (0, 165, 255)       # Amber Orange
        else:
            state = "Non-Compliant"
            color = (52, 73, 235)       # Strong Red

        workers.append({
            "box": pbox,
            "confidence": p["conf"],
            "state": state,
            "color": color,
            "missing": missing,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "worn": list(worn_classes)
        })

    return workers, ppe_items

def draw_hud(frame, workers, fps=0.0):
    """Renders a sleek industrial safety dashboard HUD on the top of the frame."""
    H, W = frame.shape[:2]
    hud_h = 70
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, hud_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, hud_h), (W, hud_h), (60, 70, 85), 2)

    total = len(workers)
    fully = sum(1 for w in workers if w["state"] == "Fully Compliant")
    part  = sum(1 for w in workers if w["state"] == "Partially Compliant")
    non_c = sum(1 for w in workers if w["state"] == "Non-Compliant")
    rate  = (fully / total * 100.0) if total > 0 else 100.0

    # Title & Stats
    cv2.putText(frame, "PPE COMPLIANCE MONITOR (YOLOv9-c SSL)", (18, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    
    stats_str = f"WORKERS: {total}  |  RATE: {rate:.1f}%"
    cv2.putText(frame, stats_str, (18, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1, cv2.LINE_AA)

    # Badges
    badge_x = max(W - 420, 300)
    # Green Fully
    cv2.rectangle(frame, (badge_x, 15), (badge_x + 95, 55), (35, 160, 80), -1)
    cv2.putText(frame, f"FULL: {fully}", (badge_x + 8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Orange Partial
    cv2.rectangle(frame, (badge_x + 105, 15), (badge_x + 205, 55), (0, 140, 220), -1)
    cv2.putText(frame, f"PART: {part}", (badge_x + 113, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Red Non-Compliant
    cv2.rectangle(frame, (badge_x + 215, 15), (badge_x + 315, 55), (40, 50, 200), -1)
    cv2.putText(frame, f"FAIL: {non_c}", (badge_x + 225, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def process_frame(frame, model, conf=0.25, iou=0.45, ioppe_threshold=0.30):
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "cls_id": cls_id,
                "box": [x1, y1, x2, y2],
                "conf": conf_val
            })

    workers, ppe_items = evaluate_compliance(detections, model.names, ioppe_threshold)

    # Draw PPE boxes with thin outline
    for ppe in ppe_items:
        px1, py1, px2, py2 = [int(v) for v in ppe["box"]]
        cname = ppe["name"]
        color = (180, 180, 180)
        if "helmet" in cname: color = (255, 230, 0)      # Cyan / yellow for helmet
        elif "vest" in cname: color = (20, 230, 255)     # Bright yellow/orange
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

    # Render worker compliance boxes
    for wi, w in enumerate(workers):
        x1, y1, x2, y2 = [int(v) for v in w["box"]]
        color = w["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        state_prefix = "[OK]" if w["state"] == "Fully Compliant" else ("[WARN]" if w["state"] == "Partially Compliant" else "[FAIL]")
        label = f"{state_prefix} W{wi+1}"
        if w["missing"]:
            label += f" | No {','.join(w['missing'])}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    draw_hud(frame, workers)
    return frame, workers

def main():
    parser = argparse.ArgumentParser(description="Real-Time PPE Compliance Detection & Video Recording")
    parser.add_argument("--model", type=str, default="best_phase2.pt", help="Path to trained .pt model weights")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or video/image path)")
    parser.add_argument("--output", type=str, default="output_compliance.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 for full video)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--ioppe", type=float, default=0.30, help="IoPPE threshold")
    args = parser.parse_args()

    # Model resolution: CLI arg -> local directory -> Downloads
    model_path = Path(args.model)
    if not model_path.exists():
        fallback_candidates = [
            Path("best_phase1.pt"),
            Path("best_phase2.pt"),
            Path("/Users/kennyvws/Downloads/Code/best pt phase1a.pt")
        ]
        found = False
        for c in fallback_candidates:
            if c.exists():
                model_path = c
                found = True
                break
        if not found:
            print(f"Error: Model file '{args.model}' not found. Please provide path to trained weights.")
            return

    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(str(model_path))

    is_webcam = args.source.isdigit()
    src = int(args.source) if is_webcam else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error opening source: {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_avi:
        temp_raw_avi = tmp_avi.name

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_raw_avi, fourcc, fps, (width, height))

    limit = args.max_frames if args.max_frames > 0 else (total_frames if total_frames > 0 else 999999)
    print(f"Processing source '{args.source}' ({width}x{height} @ {fps:.1f} fps)... (Press 'q' in preview to stop)")

    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < limit:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, workers = process_frame(frame, model, conf=args.conf, ioppe_threshold=args.ioppe)
            out.write(annotated_frame)
            frame_idx += 1

            if is_webcam:
                cv2.imshow("PPE Compliance Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_idx % 30 == 0:
                pct = f"{100*frame_idx/limit:.1f}%" if limit < 999999 else f"{frame_idx} frames"
                print(f"  Processed {frame_idx} frames ({pct})...")
    finally:
        cap.release()
        out.release()
        if is_webcam:
            cv2.destroyAllWindows()

    print("Inference complete. Re-encoding to H.264 MP4...")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", temp_raw_avi,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", args.output
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Success! Output video saved to: {args.output}")
    except Exception as e:
        print(f"ffmpeg conversion note ({e}); moving raw output to {args.output}")
        os.replace(temp_raw_avi, args.output)
    finally:
        if os.path.exists(temp_raw_avi):
            try:
                os.remove(temp_raw_avi)
            except OSError:
                pass

if __name__ == "__main__":
    main()
