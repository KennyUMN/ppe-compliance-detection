"""
Standalone Real-Time PPE Compliance Detection Script
Using YOLOv9-c and Enhanced IoPPE (Intersection over PPE Area) State Machine.
Includes Anatomical Zoning and Exclusive Bipartite Worker Assignment.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 12 Classes from Merged PPE Benchmark
CLASS_NAMES = {
    0:  "person",
    1:  "face",
    2:  "face-mask",
    3:  "foot",
    4:  "glasses",
    5:  "gloves",
    6:  "helmet",
    7:  "hands",
    8:  "head",
    9:  "body-protection",
    10: "shoes",
    11: "safety-vest"
}

# Mandatory items required for "Fully Compliant" state
REQUIRED_PPE = {
    6:  "Helmet",
    11: "Safety Vest"
}

# Optional PPE tracked by system
OPTIONAL_PPE = {
    2:  "Face Mask",
    4:  "Glasses",
    5:  "Gloves",
    10: "Shoes"
}

# Vertical anatomical bounds relative to worker height [y_min_ratio, y_max_ratio]
PPE_ANATOMICAL_ZONES = {
    6:  (0.00, 0.35),  # Helmet: upper 35% of worker bounding box
    2:  (0.05, 0.35),  # Face Mask: head region
    4:  (0.05, 0.35),  # Glasses: eye region
    11: (0.15, 0.85),  # Safety Vest: torso region
    5:  (0.30, 0.90),  # Gloves: arm/hand region
    10: (0.60, 1.05),  # Shoes: lower 40% of worker box
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

def satisfies_anatomical_zone(box_person, box_ppe, cls_id):
    """Verifies that the detected PPE is located within the proper human body section."""
    if cls_id not in PPE_ANATOMICAL_ZONES:
        return True
    py1, py2 = box_person[1], box_person[3]
    ph = max(1.0, py2 - py1)

    ppe_cy = (box_ppe[1] + box_ppe[3]) / 2.0
    rel_y = (ppe_cy - py1) / ph

    zone_min, zone_max = PPE_ANATOMICAL_ZONES[cls_id]
    return zone_min <= rel_y <= zone_max

def evaluate_compliance(detections, ioppe_threshold=0.30):
    """
    Maps detected PPE items to individual workers using IoPPE,
    anatomical height validation, and greedy exclusive matching.
    """
    persons = [d for d in detections if d["cls_id"] == 0]
    ppe_items = [d for d in detections if d["cls_id"] != 0]

    # Precompute candidate matches
    candidates = []
    for pi, p in enumerate(persons):
        pbox = p["box"]
        for ppei, ppe in enumerate(ppe_items):
            cls_id = ppe["cls_id"]
            ppe_box = ppe["box"]
            score = compute_ioppe(pbox, ppe_box)
            if score >= ioppe_threshold and satisfies_anatomical_zone(pbox, ppe_box, cls_id):
                candidates.append((score, pi, ppei, cls_id))

    # Exclusive matching: each PPE box is assigned to at most ONE worker
    candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_ppe = set()
    worker_ppe_map = {pi: set() for pi in range(len(persons))}

    for score, pi, ppei, cls_id in candidates:
        if ppei not in assigned_ppe:
            assigned_ppe.add(ppei)
            worker_ppe_map[pi].add(cls_id)

    workers = []
    for pi, p in enumerate(persons):
        pbox = p["box"]
        worn_classes = worker_ppe_map[pi]

        worn_req = {name: (cls_id in worn_classes) for cls_id, name in REQUIRED_PPE.items()}
        worn_opt = {name: (cls_id in worn_classes) for cls_id, name in OPTIONAL_PPE.items()}

        n_worn = sum(worn_req.values())
        if n_worn == len(REQUIRED_PPE):
            state = "Fully Compliant"
            color = (0, 200, 0)      # Green (BGR)
        elif n_worn > 0:
            state = "Partially Compliant"
            color = (0, 165, 255)    # Orange (BGR)
        else:
            state = "Non-Compliant"
            color = (50, 50, 220)    # Red (BGR)

        workers.append({
            "box": pbox,
            "confidence": p["conf"],
            "state": state,
            "color": color,
            "missing": [k for k, v in worn_req.items() if not v],
            "optional": [k for k, v in worn_opt.items() if v]
        })

    return workers

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

    workers = evaluate_compliance(detections, ioppe_threshold)

    # Render worker compliance boxes
    for wi, w in enumerate(workers):
        x1, y1, x2, y2 = [int(v) for v in w["box"]]
        color = w["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{w['state']} (Worker {wi+1})"
        if w["missing"]:
            label += f" | Missing: {', '.join(w['missing'])}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return frame, workers

def main():
    parser = argparse.ArgumentParser(description="Real-Time PPE Compliance Detection")
    parser.add_argument("--model", type=str, default="best_phase2.pt", help="Path to best_phase2.pt")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or video/image path)")
    parser.add_argument("--output", type=str, default="output_compliance.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--ioppe", type=float, default=0.30, help="IoPPE threshold for person-PPE assignment")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found. Please provide path to trained weights.")
        return

    print(f"Loading YOLOv9c model from {args.model}...")
    model = YOLO(args.model)

    is_webcam = args.source.isdigit()
    src = int(args.source) if is_webcam else args.source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Running compliance tracking on: {args.source} (Press 'q' to stop)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, workers = process_frame(frame, model, conf=args.conf, ioppe_threshold=args.ioppe)
        out.write(annotated_frame)

        if is_webcam:
            cv2.imshow("PPE Compliance Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Completed! Output written to: {args.output}")

if __name__ == "__main__":
    main()
