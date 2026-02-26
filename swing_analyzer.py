import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_diagnostic_swing(video_path, club_type=None):
    """General stability scan for Head and Hips."""
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    
    # Dynamic Scale for Big Text
    fs = h / 1000  
    thick = int(2 * fs)

    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(raw_tfile.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    addr_head_y, addr_hip_x = None, None
    is_downswing = False
    max_w_y = 1.0
    seq_status = "CALIBRATING..."

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                if addr_head_y is None:
                    addr_head_y = lm[0].y
                    addr_hip_x = (lm[23].x + lm[24].x) / 2

                if not is_downswing:
                    if lm[16].y < max_w_y: max_w_y = lm[16].y
                    elif lm[16].y > (max_w_y + 0.05): is_downswing = True

                # Sequence Math: Shoulder width vs Hip width
                h_w = abs(lm[23].x - lm[24].x)
                s_w = abs(lm[11].x - lm[12].x)
                if is_downswing and seq_status == "CALIBRATING...":
                    seq_status = "SHOULDER LEAD (PULL)" if s_w < h_w else "HIP LEAD (PRO)"

                # BIG TEXT OVERLAYS with Shadow for Readability
                label = f"SEQ: {seq_status}"
                cv2.putText(frame, label, (50, int(h*0.1)), 2, fs*1.2, (0,0,0), thick+2)
                cv2.putText(frame, label, (50, int(h*0.1)), 2, fs*1.2, (255, 255, 0), thick)

                head_err = abs(lm[0].y - addr_head_y)
                h_col = (0, 255, 0) if head_err < 0.04 else (0, 0, 255)
                cv2.putText(frame, "HEAD STABLE", (50, int(h*0.18)), 2, fs, (0,0,0), thick+2)
                cv2.putText(frame, "HEAD STABLE", (50, int(h*0.18)), 2, fs, h_col, thick)

            out.write(frame)

    cap.release(); out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### ⚖️ Stability Report\n**Sequence:** {seq_status}", web_tfile.name

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    """Anatomy Plane & X-Factor Lab with Scale-Corrected Text."""
    cap = cv2.VideoCapture(video_path)
    h_pix, w_pix = int(cap.get(4)), int(cap.get(3))
    fps = int(cap.get(5))
    fs = h_pix / 1000  
    thick = int(2 * fs)
    
    frozen_apex, s_y_lock, h_y_lock = None, None, None
    conf = 0
    wrist_trail = []
    is_downswing = False
    max_h = 1.0
    lag_top, lag_impact, max_xf = None, None, 0

    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_pix, h_pix))

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                curr_rs = (int(lm[12].x * w_pix), int(lm[12].y * h_pix))
                curr_re = (int(lm[14].x * w_pix), int(lm[14].y * h_pix))
                curr_rw = (int(lm[16].x * w_pix), int(lm[16].y * h_pix))

                if s_y_lock is None and lm[12].visibility > 0.9:
                    conf += 1
                    if conf > 10:
                        s_y_lock, h_y_lock = curr_rs[1] - 180, int(lm[24].y * h_pix) - 80
                        dx, dy = curr_rw[0] - curr_rs[0], curr_rw[1] - curr_rs[1]
                        if dx != 0:
                            body_w = abs(lm[12].x - lm[11].x) * w_pix
                            frozen_apex = (int(curr_rs[0] + (body_w * 3.5)), int(s_y_lock + (dy/dx * (body_w * 3.5))))

                # X-Factor (Shoulder width vs Hip width stretch)
                xf = int(abs(abs(lm[11].x - lm[12].x) - abs(lm[23].x - lm[24].x)) * 100)
                if xf > max_xf: max_xf = xf
                
                # Lag Math
                ba = np.array([lm[12].x - lm[14].x, lm[12].y - lm[14].y])
                bc = np.array([lm[16].x - lm[14].x, lm[16].y - lm[14].y])
                ang = int(np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1))))

                if frozen_apex:
                    overlay = frame.copy()
                    pts = np.array([[frozen_apex[0], frozen_apex[1]], [0, s_y_lock], [0, h_y_lock + 120]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, frozen_apex, (0, s_y_lock), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(frame, frozen_apex, (0, h_y_lock + 120), (0, 0, 0), 2, cv2.LINE_AA)

                # BIG READABLE OVERLAYS
                cv2.putText(frame, f"X-FACTOR: {xf}", (int(w_pix*0.05), int(h_pix*0.1)), 2, fs*1.2, (0,0,0), thick+2)
                cv2.putText(frame, f"X-FACTOR: {xf}", (int(w_pix*0.05), int(h_pix*0.1)), 2, fs*1.2, (255, 0, 255), thick)
                cv2.putText(frame, f"{ang}deg", (curr_re[0]+30, curr_re[1]), 2, fs, (0,0,0), thick+2)
                cv2.putText(frame, f"{ang}deg", (curr_re[0]+30, curr_re[1]), 2, fs, (0,255,255), thick)

                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    wrist_trail.append(curr_rw)
                for i in range(1, len(wrist_trail)): 
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3, cv2.LINE_AA)

                if is_downswing and lag_top is None: lag_top = ang
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = ang

            out.write(frame)

    cap.release(); out.release()
    web_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### 🏌️ Plane & X-Factor Lab\n**Max X-Factor:** {max_xf} | **Impact Lag:** {lag_impact}°", web_path
