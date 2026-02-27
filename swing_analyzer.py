import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_foundation_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    fs, thick = h / 1000, int(2 * (h / 1000))
    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(raw_tfile.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    addr_head_y, addr_hip_x = None, None
    is_downswing, max_w_y = False, 1.0
    seq_status, max_stretch = "ANALYZING...", 0

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # Lock Anchors at Address
                if addr_head_y is None:
                    addr_head_y, addr_hip_x = lm[0].y, (lm[23].x + lm[24].x) / 2

                if not is_downswing:
                    if lm[16].y < max_w_y: max_w_y = lm[16].y
                    elif lm[16].y > (max_h_trig := max_w_y + 0.05): is_downswing = True

                # --- STABILITY CALCULATIONS ---
                head_drift = abs(lm[0].y - addr_head_y)
                hip_drift = abs(((lm[23].x + lm[24].x) / 2) - addr_hip_x)
                
                head_stable = head_drift < 0.04 # Threshold for "Fat" shots
                hip_stable = hip_drift < 0.05   # Threshold for "Sway"
                
                # --- STRETCH & SEQUENCE ---
                curr_s = int(abs(abs(lm[11].x - lm[12].x) - abs(lm[23].x - lm[24].x)) * 100)
                if curr_s > max_stretch: max_stretch = curr_s

                if is_downswing:
                    h_w, s_w = abs(lm[23].x - lm[24].x), abs(lm[11].x - lm[12].x)
                    seq_status = "SHOULDER SPIN" if s_w < (h_w * 0.85) else "PRO HIP LEAD"

                # --- DRAWING STABILITY BOXES ---
                # Head Box (Green if stable, Red if dipping/lifting)
                h_box_col = (0, 255, 0) if head_stable else (0, 0, 255)
                h_x, h_y = int(lm[0].x * w), int(lm[0].y * h)
                cv2.rectangle(frame, (h_x-30, int(addr_head_y*h)-30), (h_x+30, int(addr_head_y*h)+30), h_col := h_box_col, 2)
                
                # Hip Box (Green if rotating, Red if sliding)
                hip_col = (0, 255, 0) if hip_stable else (0, 0, 255)
                cv_hip_x = int(((lm[23].x + lm[24].x) / 2) * w)
                cv_hip_y = int(((lm[23].y + lm[24].y) / 2) * h)
                cv2.rectangle(frame, (cv_hip_x-40, cv_hip_y-40), (cv_hip_x+40, cv_hip_y+40), hip_col, 2)
                cv2.line(frame, (int(addr_hip_x * w), 0), (int(addr_hip_x * w), h), (255, 255, 255), 1)

                # --- STABILITY SCOREBOARD (Top Right) ---
                cv2.putText(frame, "HEAD", (w-250, 60), 2, fs, h_box_col, thick)
                cv2.putText(frame, "HIPS", (w-250, 120), 2, fs, hip_col, thick)

                # --- SEQUENCE SCOREBOARD (Top Left) ---
                cv2.putText(frame, f"STRETCH: {max_stretch}", (50, 60), 2, fs, (255, 0, 255), thick)
                cv2.putText(frame, f"SEQ: {seq_status}", (50, 120), 2, fs, (255, 255, 0), thick)

            out.write(frame)
    cap.release(); out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_path := raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### ⚖️ Stability Report\nHead Stability: {'PASS' if head_stable else 'FAIL'} | Hip Stability: {'PASS' if hip_stable else 'FAIL'}", web_tfile.name

# (Keep analyze_swing_plane the same as the previous version)
