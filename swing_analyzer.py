import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_diagnostic_swing(video_path, club_type=None):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(raw_tfile.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    # State for Sequence Analysis
    addr_head_y, addr_hip_x = None, None
    max_hip_turn_frame = 0
    max_shoulder_turn_frame = 0
    is_downswing = False
    max_wrist_height = 1.0
    sequence_status = "PENDING"

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 1. SETUP ANCHORS
                if addr_head_y is None:
                    addr_head_y = lm[0].y
                    addr_hip_x = (lm[23].x + lm[24].x) / 2

                # 2. TRACK DOWNSWING TRIGGER
                if not is_downswing:
                    if lm[16].y < max_wrist_height: max_wrist_height = lm[16].y
                    elif lm[16].y > (max_wrist_height + 0.05): is_downswing = True

                # 3. SEQUENCE LOGIC (Width Analysis)
                # The 'narrower' the width, the deeper the turn
                hip_width = abs(lm[23].x - lm[24].x)
                shoulder_width = abs(lm[11].x - lm[12].x)

                if is_downswing:
                    # Capture the frame where rotation starts back toward the target
                    if sequence_status == "PENDING":
                        # If shoulders are narrower (more turned) than hips at trigger
                        if shoulder_width < hip_width:
                            sequence_status = "SHOULDER LEAD (PULL RISK)"
                        else:
                            sequence_status = "HIP LEAD (GOOD)"

                # 4. OVERLAYS
                cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
                cv2.putText(frame, f"SEQUENCE: {sequence_status}", (20, 40), 2, 0.6, (255, 255, 0), 2)
                
                # Keep original stability markers
                head_color = (0, 255, 0) if abs(lm[0].y - addr_head_y) < 0.04 else (0, 0, 255)
                cv2.putText(frame, "HEAD STABILITY", (20, 75), 2, 0.6, head_color, 2)
                
                hip_sway = abs(((lm[23].x + lm[24].x)/2) - addr_hip_x)
                sway_color = (0, 255, 0) if hip_sway < 0.05 else (0, 0, 255)
                cv2.putText(frame, "HIP SWAY", (20, 105), 2, 0.6, sway_color, 2)

            out.write(frame)

    cap.release(); out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"### ⚖️ Stability & Sequence Report\n**Sequence Result:** {sequence_status}\n\n*Note: A 'Shoulder Lead' usually results in a pull-left as the upper body outpaces the lower body clearance.*", web_tfile.name

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    h_pix, w_pix = int(cap.get(4)), int(cap.get(3))
    fps = int(cap.get(5))
    
    frozen_apex = None
    shoulder_y_lock, hip_y_lock = None, None
    confidence_frames = 0
    wrist_trail = []
    is_downswing = False
    max_h = 1.0
    lag_top, lag_impact = None, None
    max_xfactor = 0

    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_pix, h_pix))

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Joint Mapping
                curr_rs = (int(lm[12].x * w_pix), int(lm[12].y * h_pix))
                curr_re = (int(lm[14].x * w_pix), int(lm[14].y * h_pix))
                curr_rw = (int(lm[16].x * w_pix), int(lm[16].y * h_pix))

                # --- 1. PLANE LOCK ---
                if shoulder_y_lock is None and lm[12].visibility > 0.9:
                    confidence_frames += 1
                    if confidence_frames > 10:
                        shoulder_y_lock, hip_y_lock = curr_rs[1] - 180, int(lm[24].y * h_pix) - 80
                        dx, dy = curr_rw[0] - curr_rs[0], curr_rw[1] - curr_rs[1]
                        if dx != 0:
                            body_w = abs(lm[12].x - lm[11].x) * w_pix
                            frozen_apex = (int(curr_rs[0] + (body_w * 3.5)), int(shoulder_y_lock + (dy/dx * (body_w * 3.5))))

                # --- 2. X-FACTOR MATH ---
                # Compare shoulder width to hip width to estimate rotation differential
                s_width = abs(lm[11].x - lm[12].x)
                h_width = abs(lm[23].x - lm[24].x)
                # This ratio approximates the "twist" between the two planes
                current_xfactor = int(abs(s_width - h_width) * 100) 
                if current_xfactor > max_xfactor: max_xfactor = current_xfactor
                
                # Draw X-Factor near the spine
                spine_x = int((lm[12].x + lm[24].x) / 2 * w_pix)
                spine_y = int((lm[12].y + lm[24].y) / 2 * h_pix)
                cv2.putText(frame, f"X-FACTOR: {current_xfactor}", (spine_x + 20, spine_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # --- 3. DRAW PLANE & TRAIL ---
                if frozen_apex:
                    overlay = frame.copy()
                    pts = np.array([[frozen_apex[0], frozen_apex[1]], [0, shoulder_y_lock], [0, hip_y_lock + 120]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, frozen_apex, (0, shoulder_y_lock), (0, 0, 0), 2)
                    cv2.line(frame, frozen_apex, (0, hip_y_lock + 120), (0, 0, 0), 2)

                # --- 4. HINGE ANGLES (LAG) ---
                ba = np.array([lm[12].x - lm[14].x, lm[12].y - lm[14].y])
                bc = np.array([lm[16].x - lm[14].x, lm[16].y - lm[14].y])
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                ang = int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
                cv2.putText(frame, f"{ang}deg", (curr_re[0] + 15, curr_re[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    wrist_trail.append(curr_rw)
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3)

                if is_downswing and lag_top is None: lag_top = ang
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = ang

            out.write(frame)

    cap.release(); out.release()
    web_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_path])
    
    summary = f"### 🏌️ Lab Report: X-Factor & Plane\n**Max X-Factor (Stretch):** {max_xfactor}\n**Top Lag:** {lag_top}° | **Impact Lag:** {lag_impact}°\n\n*Coach's Note: A high X-Factor with a 'Shoulder Lead' creates the pull-left. Focus on maintaining that stretch while the hips initiate the downswing.*"
    return summary, web_path

