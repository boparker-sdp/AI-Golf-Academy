import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_foundation_sequence(video_path):
    """FACE-ON VIEW: Focuses on Body Stability, Hip-Shoulder Stretch, and Timing."""
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
                if addr_head_y is None:
                    addr_head_y, addr_hip_x = lm[0].y, (lm[23].x + lm[24].x) / 2

                # Detect Downswing Trigger
                if not is_downswing:
                    if lm[16].y < max_w_y: max_w_y = lm[16].y
                    elif lm[16].y > (max_w_y + 0.05): is_downswing = True

                # Sequence Stretch (X-Factor) - Higher is better coil
                stretch = int(abs(abs(lm[11].x - lm[12].x) - abs(lm[23].x - lm[24].x)) * 100)
                if stretch > max_stretch: max_stretch = stretch

                if is_downswing:
                    h_w, s_w = abs(lm[23].x - lm[24].x), abs(lm[11].x - lm[12].x)
                    seq_status = "SHOULDER SPIN" if s_w < (h_w * 0.85) else "PRO HIP LEAD"

                # OVERLAYS - Sequence Stretch (X-Factor)
                label_s = f"STRETCH: {stretch}"
                cv2.putText(frame, label_s, (50, int(h*0.1)), 2, fs*1.2, (0,0,0), thick+2)
                cv2.putText(frame, label_s, (50, int(h*0.1)), 2, fs*1.2, (255, 0, 255), thick)
                
                # Sequence Label
                label_q = f"SEQ: {seq_status}"
                cv2.putText(frame, label_q, (50, int(h*0.18)), 2, fs, (0,0,0), thick+2)
                cv2.putText(frame, label_q, (50, int(h*0.18)), 2, fs, (255, 255, 0), thick)
                
                # Stability Check
                head_err = abs(lm[0].y - addr_head_y)
                h_col = (0, 255, 0) if head_err < 0.04 else (0, 0, 255)
                cv2.putText(frame, "STABILITY", (50, int(h*0.25
