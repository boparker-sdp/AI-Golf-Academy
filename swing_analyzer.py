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
                    elif lm[16].y > (max_w_y + 0.05): is_downswing = True

                # Stretch Logic
                curr_s = int(abs(abs(lm[11].x - lm[12].x) - abs(lm[23].x - lm[24].x)) * 100)
                if curr_s > max_stretch: max_stretch = curr_s

                if is_downswing:
                    h_w, s_w = abs(lm[23].x - lm[24].x), abs(lm[11].x - lm[12].x)
                    seq_status = "SHOULDER SPIN" if s_w < (h_w * 0.85) else "PRO HIP LEAD"

                # 1. DRAW STABILITY (Head Box & Hip Line)
                head_x = int(lm[0].x * w)
                cv2.rectangle(frame, (head_x - 30, int(addr_head_y * h) - 30), (head_x + 30, int(addr_head_y * h) + 30), (0, 255, 0), 2)
                cv2.line(frame, (int(addr_hip_x * w), 0), (int(addr_hip_x * w), h), (255, 255, 0), 1)

                # 2. PERSISTENT TEXT
                cv2.putText(frame, f"MAX STRETCH: {max_stretch}", (50, int(h*0.1)), 2, fs, (0,0,0), thick+2)
                cv2.putText(frame, f"MAX STRETCH: {max_stretch}", (50, int(h*0.1)), 2, fs, (255, 0, 255), thick)
                cv2.putText(frame, f"SEQ: {seq_status}", (50, int(h*0.18)), 2, fs, (0,0,0), thick+2)
                cv2.putText(frame, f"SEQ: {seq_status}", (50, int(h*0.18)), 2, fs, (255, 255, 0), thick)

            out.write(frame)
    cap.release(); out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### ⚖️ Foundation & Sequence\nMax Stretch: {max_stretch} | Sequence: {seq_status}", web_tfile.name

def analyze_swing_plane(video_path):
    cap = cv2.VideoCapture(video_path)
    h_pix, w_pix = int(cap.get(4)), int(cap.get(3))
    fps, fs, thick = int(cap.get(5)), h_pix/1000, int(2*(h_pix/1000))
    s_y_lock, h_y_lock, frozen_apex = None, None, None
    wrist_trail, is_downswing, max_h = [], False, 1.0
    lag_top, lag_impact = None, None

    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_pix, h_pix))

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                curr_rw = (int(lm[16].x*w_pix), int(lm[16].y*h_pix))
                
                # Setup Plane Cone
                if s_y_lock is None:
                    s_y_lock, h_y_lock = int(lm[12].y*h_pix)-150, int(lm[24].y*h_pix)-50
                    dx, dy = curr_rw[0]-int(lm[12].x*w_pix), curr_rw[1]-int(lm[12].y*h_pix)
                    frozen_apex = (int(lm[12].x*w_pix + (abs(lm[12].x-lm[11].x)*w_pix*3.5)), int(s_y_lock+(dy/dx if dx!=0 else 1)*3.5))

                # Hinge Math
                ba = np.array([lm[12].x-lm[14].x, lm[12].y-lm[14].y])
                bc = np.array([lm[16].x-lm[14].x, lm[16].y-lm[14].y])
                ang = int(np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1))))

                # Draw Cone (40% Opacity)
                if frozen_apex:
                    overlay = frame.copy()
                    pts = np.array([[frozen_apex[0], frozen_apex[1]], [0, s_y_lock], [0, h_y_lock+100]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.line(frame, frozen_apex, (0, s_y_lock), (0,0,0), 2)
                    cv2.line(frame, frozen_apex, (0, h_y_lock+100), (0,0,0), 2)

                # Snapshots
                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    wrist_trail.append(curr_rw)
                
                if is_downswing and lag_top is None: lag_top = ang
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = ang

                # Draw Trail
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 2)

                # PERSISTENT LABELS
                if lag_top:
                    cv2.putText(frame, f"TOP: {lag_top}deg", (50, int(h_pix*0.1)), 2, fs, (0,0,0), thick+2)
                    cv2.putText(frame, f"TOP: {lag_top}deg", (50, int(h_pix*0.1)), 2, fs, (0,255,255), thick)
                if lag_impact:
                    cv2.putText(frame, f"IMPACT: {lag_impact}deg", (50, int(h_pix*0.18)), 2, fs, (0,0,0), thick+2)
                    cv2.putText(frame, f"IMPACT: {lag_impact}deg", (50, int(h_pix*0.18)), 2, fs, (0,255,255), thick)

            out.write(frame)
    cap.release(); out.release()
    web_p = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_p.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### 🚀 Swing Plane Lab\nTop Hinge: {lag_top}° | Impact Hinge: {lag_impact}°", web_p.name
