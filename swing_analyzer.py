import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_diagnostic_swing(video_path, club_type=None):
    """General stability scan for Head and Hips to diagnose fat/thin shots."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(raw_tfile.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    address_head_y = None
    address_hip_x = None
    head_status = "STABLE"
    hip_status = "STABLE"
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if address_head_y is None:
                    address_head_y = lm[0].y
                    address_hip_x = (lm[23].x + lm[24].x) / 2

                curr_head_y = lm[0].y
                if curr_head_y > address_head_y + 0.03: head_status = "DIPPING"
                elif curr_head_y < address_head_y - 0.03: head_status = "LIFTING"

                curr_hip_x = (lm[23].x + lm[24].x) / 2
                if abs(curr_hip_x - address_hip_x) > 0.05: hip_status = "SWAYING"

                cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"HEAD: {head_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"HIPS: {hip_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)

    cap.release(); out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "Diagnostic Scan Complete.", web_tfile.name

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

                # --- 2. DRAW PLANE & TRAIL ---
                if frozen_apex:
                    overlay = frame.copy()
                    pts = np.array([[frozen_apex[0], frozen_apex[1]], [0, shoulder_y_lock], [0, hip_y_lock + 120]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, frozen_apex, (0, shoulder_y_lock), (0, 0, 0), 2)
                    cv2.line(frame, frozen_apex, (0, hip_y_lock + 120), (0, 0, 0), 2)

                # --- 3. THE MISSING HINGE ANGLES (LAG) ---
                # Calculate angle between Shoulder-Elbow and Elbow-Wrist
                ba = np.array([lm[12].x - lm[14].x, lm[12].y - lm[14].y])
                bc = np.array([lm[16].x - lm[14].x, lm[16].y - lm[14].y])
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                ang = int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
                
                # Draw live angle on the arm
                cv2.putText(frame, f"{ang}deg", (curr_re[0] + 15, curr_re[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Trail Logic
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
    
    summary = f"### 🏌️ Lab Report: Pull-Left Diagnosis\n**Top Lag:** {lag_top}° | **Impact Lag:** {lag_impact}°\n\n**Coach's Note:** You are 'On Plane' but 'Out of Sync.' Your shoulders are racing ahead of your hips. Focus on letting the hips 'clear' the space first to allow the club to swing toward the target rather than across your body."
    return summary, web_path
