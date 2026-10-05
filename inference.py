"""
inference_ppe_compliance_v2.py
==============================
Production-grade PPE Compliance Detection & Temporal Smoothing Pipeline.
Features:
- Two-stage detection: robust person localization + 12-class specialized PPE inference.
- Multi-worker tracking with box smoothing (EMA) and relaxed centroid/IoU matching.
- Temporal hysteresis / debounce window to eliminate state flickering.
- Modern high-definition HUD & badge overlay.
- Output encoded for broad web and mobile compatibility (libx264, yuv420p).
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

# Correct class mapping from best pt phase1a.pt:
# {0: 'person', 1: 'face', 2: 'foot', 3: 'glasses', 4: 'gloves',
#  5: 'helmet', 6: 'hands', 7: 'head', 8: 'shoes', 9: 'safety-vest',
#  10: 'face-protection', 11: 'body-protection'}
CLASS_HELMET = 5
CLASS_VEST = 9
CLASS_BODY_PROT = 11
CLASS_GLASSES = 3
CLASS_GLOVES = 4
CLASS_SHOES = 8

COLOR_COMPLIANT = (34, 197, 94)      # Emerald Green (RGB: 94, 197, 34 -> BGR: 34, 197, 94)
COLOR_PARTIAL = (22, 115, 249)        # Vibrant Orange (BGR: 22, 115, 249)
COLOR_VIOLATION = (68, 68, 239)      # Bright Red (BGR: 68, 68, 239)
COLOR_HUD_BG = (20, 24, 33)          # Dark Slate Navy
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 185, 195)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def compute_center_dist(box1, box2, norm_w, norm_h):
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    dx = (c1x - c2x) / norm_w
    dy = (c1y - c2y) / norm_h
    return np.sqrt(dx * dx + dy * dy)

def compute_ioppe(box_person, box_ppe):
    """Intersection over PPE Area."""
    x1 = max(box_person[0], box_ppe[0])
    y1 = max(box_person[1], box_ppe[1])
    x2 = min(box_person[2], box_ppe[2])
    y2 = min(box_person[3], box_ppe[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a_ppe = (box_ppe[2] - box_ppe[0]) * (box_ppe[3] - box_ppe[1])
    return inter / a_ppe if a_ppe > 0 else 0.0

class WorkerTrack:
    def __init__(self, track_id, box, hold_frames=25):
        self.track_id = track_id
        self.box = np.array(box, dtype=np.float32)
        self.hold_frames = hold_frames
        self.missing_frames = 0
        self.total_frames = 1
        
        # Debounce counters for PPE
        self.last_helmet_conf = 0.0
        self.last_vest_conf = 0.0
        self.helmet_frames_ago = 999
        self.vest_frames_ago = 999
        
    def update(self, new_box, has_helmet, helmet_conf, has_vest, vest_conf, alpha=0.75):
        # Smooth bbox coordinates
        self.box = alpha * np.array(new_box, dtype=np.float32) + (1.0 - alpha) * self.box
        self.missing_frames = 0
        self.total_frames += 1
        
        if has_helmet:
            self.helmet_frames_ago = 0
            self.last_helmet_conf = helmet_conf
        else:
            self.helmet_frames_ago += 1
            
        if has_vest:
            self.vest_frames_ago = 0
            self.last_vest_conf = vest_conf
        else:
            self.vest_frames_ago += 1

    def mark_missed(self):
        self.missing_frames += 1
        self.helmet_frames_ago += 1
        self.vest_frames_ago += 1

    @property
    def is_helmet_active(self):
        return self.helmet_frames_ago <= self.hold_frames

    @property
    def is_vest_active(self):
        return self.vest_frames_ago <= self.hold_frames

    @property
    def status(self):
        if self.is_helmet_active and self.is_vest_active:
            return "Fully Compliant", COLOR_COMPLIANT
        elif self.is_helmet_active or self.is_vest_active:
            return "Partially Compliant", COLOR_PARTIAL
        else:
            return "Non-Compliant", COLOR_VIOLATION


class PPESystem:
    def __init__(self, person_model_path, ppe_model_path, hold_frames=25):
        print(f"Loading Person Detector: {person_model_path}")
        self.person_model = YOLO(person_model_path)
        print(f"Loading Specialized PPE Model: {ppe_model_path}")
        self.ppe_model = YOLO(ppe_model_path)
        
        self.tracks = {}
        self.next_track_id = 1
        self.hold_frames = hold_frames
        
    def process_frame(self, frame, p_conf=0.30, ppe_conf=0.18):
        h, w = frame.shape[:2]
        
        # 1. Run Person Detection
        p_res = self.person_model.predict(frame, classes=[0], conf=p_conf, verbose=False)[0]
        detected_persons = [b.xyxy[0].cpu().numpy().tolist() for b in p_res.boxes]
        
        # 2. Run PPE Detection
        ppe_res = self.ppe_model.predict(frame, conf=ppe_conf, verbose=False)[0]
        ppe_items = []
        for b in ppe_res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            box = b.xyxy[0].cpu().numpy().tolist()
            ppe_items.append({'cls': cls_id, 'conf': conf, 'box': box})
            
        # If no persons detected by person_model, fallback to body-protection or helmet regions
        if len(detected_persons) == 0:
            for item in ppe_items:
                if item['cls'] in [CLASS_BODY_PROT, 0] and item['conf'] >= 0.40:
                    detected_persons.append(item['box'])
                    
        # 3. Match detected persons with existing tracks (Combined IoU + Distance Matching)
        matched_tracks = set()
        unmatched_dets = []
        
        for pbox in detected_persons:
            best_score = -1.0
            best_tid = None
            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = compute_iou(pbox, track.box)
                cdist = compute_center_dist(pbox, track.box, w, h)
                
                # Match if IoU >= 0.18 or center distance <= 0.12 (worker bent down/moved slightly)
                if iou >= 0.18 or cdist <= 0.12:
                    score = iou + (1.0 - min(1.0, cdist))
                    if score > best_score:
                        best_score = score
                        best_tid = tid
                    
            if best_tid is not None:
                matched_tracks.add(best_tid)
                # Check PPE items inside this person
                has_h, h_conf = False, 0.0
                has_v, v_conf = False, 0.0
                for item in ppe_items:
                    if compute_ioppe(pbox, item['box']) >= 0.15:
                        if item['cls'] == CLASS_HELMET:
                            has_h = True
                            h_conf = max(h_conf, item['conf'])
                        elif item['cls'] in [CLASS_VEST, CLASS_BODY_PROT]:
                            has_v = True
                            v_conf = max(v_conf, item['conf'])
                self.tracks[best_tid].update(pbox, has_h, h_conf, has_v, v_conf)
            else:
                unmatched_dets.append(pbox)
                
        # Handle unmatched detections (new tracks)
        for pbox in unmatched_dets:
            has_h, h_conf = False, 0.0
            has_v, v_conf = False, 0.0
            for item in ppe_items:
                if compute_ioppe(pbox, item['box']) >= 0.15:
                    if item['cls'] == CLASS_HELMET:
                        has_h = True
                        h_conf = max(h_conf, item['conf'])
                    elif item['cls'] in [CLASS_VEST, CLASS_BODY_PROT]:
                        has_v = True
                        v_conf = max(v_conf, item['conf'])
            new_track = WorkerTrack(self.next_track_id, pbox, self.hold_frames)
            new_track.update(pbox, has_h, h_conf, has_v, v_conf)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
            
        # Update tracks that weren't matched
        active_tids = list(self.tracks.keys())
        for tid in active_tids:
            if tid not in matched_tracks and tid not in [t.track_id for t in self.tracks.values() if t.total_frames == 1]:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].missing_frames > 20:
                    del self.tracks[tid]
                    
        return self._render_frame(frame, ppe_items)
        
    def _render_frame(self, frame, ppe_items):
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # 1. Draw top HUD Banner
        hud_h = 70
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)
        
        # Draw HUD bottom border line
        cv2.line(vis, (0, hud_h), (w, hud_h), (60, 70, 90), 2)
        
        # HUD Text
        cv2.putText(vis, "PPE COMPLIANCE SYSTEM", (24, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_WHITE, 2)
        cv2.putText(vis, "Edge AI Safety Monitor | Semi-Supervised YOLOv9-c", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GRAY, 1)
        
        # Filter valid tracks
        valid_tracks = {tid: t for tid, t in self.tracks.items() if t.missing_frames <= 5}
        total_workers = len(valid_tracks)
        n_compliant = sum(1 for t in valid_tracks.values() if t.status[0] == "Fully Compliant")
        rate = int((n_compliant / total_workers) * 100) if total_workers > 0 else 100
        
        status_text = f"WORKERS: {total_workers}  |  COMPLIANCE: {rate}%  |  STATUS: {'SAFE' if rate == 100 else 'ALERT'}"
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
        
        # Status Pill Badge
        badge_color = COLOR_COMPLIANT if rate == 100 else (COLOR_PARTIAL if rate >= 50 else COLOR_VIOLATION)
        bx1 = w - tw - 45
        cv2.rectangle(vis, (bx1 - 10, 18), (w - 20, 52), badge_color, -1)
        cv2.putText(vis, status_text, (bx1, 42), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 2)
        
        # 2. Draw PPE Item Highlights (Subtle)
        for item in ppe_items:
            cls_id = item['cls']
            if cls_id in [CLASS_HELMET, CLASS_VEST]:
                ix1, iy1, ix2, iy2 = [int(v) for v in item['box']]
                cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 240, 255), 1)
                
        # 3. Draw Worker Tracks
        for tid, track in sorted(valid_tracks.items()):
            x1, y1, x2, y2 = [int(v) for v in track.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            state_str, color = track.status
            
            # Draw Worker Box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            
            # Corner accents
            c_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(vis, (x1, y1), (x1 + c_len, y1), COLOR_WHITE, 4)
            cv2.line(vis, (x1, y1), (x1, y1 + c_len), COLOR_WHITE, 4)
            cv2.line(vis, (x2, y1), (x2 - c_len, y1), COLOR_WHITE, 4)
            cv2.line(vis, (x2, y1), (x2, y1 + c_len), COLOR_WHITE, 4)
            cv2.line(vis, (x1, y2), (x1 + c_len, y2), COLOR_WHITE, 4)
            cv2.line(vis, (x1, y2), (x1, y2 - c_len), COLOR_WHITE, 4)
            cv2.line(vis, (x2, y2), (x2 - c_len, y2), COLOR_WHITE, 4)
            cv2.line(vis, (x2, y2), (x2, y2 - c_len), COLOR_WHITE, 4)
            
            # Worker Header Badge
            h_status = "OK" if track.is_helmet_active else "MISSING"
            v_status = "OK" if track.is_vest_active else "MISSING"
            label = f"Worker {tid}: {state_str.upper()}"
            sublabel = f"Helmet: {h_status} | Vest: {v_status}"
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
            (sw, sh), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bw = max(lw, sw) + 20
            bh = lh + sh + 16
            
            by1 = max(hud_h + 5, y1 - bh - 6)
            cv2.rectangle(vis, (x1, by1), (x1 + bw, by1 + bh), color, -1)
            cv2.putText(vis, label, (x1 + 10, by1 + lh + 4), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 2)
            cv2.putText(vis, sublabel, (x1 + 10, by1 + lh + sh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            
        return vis


def process_video(src_path, dst_path, person_model, ppe_model, start_sec=0.0, duration=None, target_fps=24):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"Error opening {src_path}")
        return False
        
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * src_fps)
    max_frames = int(duration * src_fps) if duration else (total_frames - start_frame)
    end_frame = min(total_frames, start_frame + max_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    temp_avi = dst_path.replace(".mp4", "_temp.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_avi, fourcc, src_fps, (w, h))
    
    system = PPESystem(person_model, ppe_model, hold_frames=25)
    
    print(f"Processing '{src_path}' ({w}x{h}, {start_sec}s to {end_frame/src_fps:.1f}s)...")
    curr_frame = start_frame
    while cap.isOpened() and curr_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = system.process_frame(frame)
        out.write(annotated)
        curr_frame += 1
        
    cap.release()
    out.release()
    
    # Remux with ffmpeg for pristine web compatibility (H.264 + YUV420p)
    print("Remuxing with ffmpeg (H.264, yuv420p)...")
    cmd = [
        "ffmpeg", "-y", "-i", temp_avi,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "19",
        dst_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if os.path.exists(temp_avi):
        os.remove(temp_avi)
    print(f"✅ Rendered successfully: {dst_path}")
    
    # Generate high quality GIF if requested
    gif_path = dst_path.replace(".mp4", ".gif")
    print(f"Generating optimized GIF: {gif_path}...")
    gif_cmd = [
        "ffmpeg", "-y", "-i", dst_path,
        "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        gif_path
    ]
    subprocess.run(gif_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(f"✅ GIF generated: {gif_path} ({os.path.getsize(gif_path)/(1024*1024):.2f} MB)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--person-model", default="/Users/kennyvws/ppe-compliance-detection/yolov8n.pt")
    parser.add_argument("--ppe-model", default="/Users/kennyvws/Downloads/Code/best pt phase1a.pt")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    
    process_video(args.source, args.output, args.person_model, args.ppe_model, args.start, args.duration)
