import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_diagnostic_swing(video_path, club_type=None):
    """Head/Hip stability scan to diagnose fat/thin shots."""
    cap = cv2.VideoCapture(video_path)
    h, w = int(cap.get(4)), int(cap.get(3))
    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), (w, h))

    addr_head_y, addr_hip_x = None, None
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Sync logic for stability
                if addr_head_y is None:
                    addr_head_y = lm[0].y
                    addr_hip_x = (lm[23].x + lm[24].x) / 2
                # (Overlays for Head/Hip stability)
            out.write(frame)
    cap.release(); out.release()
    return "Diagnostic Complete.", raw_path

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    """Anatomy-Synchronized Plane Lab."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    h_pix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_pix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # State Anchors
    anatomy_apex = None
    shoulder_y_lock = None
    hip_y_lock = None
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
                
                # JOINT TRUTH (12 = Right Shoulder, 24 = Right Hip, 16 = Right Wrist)
                # We calculate pixels directly from the frame we are drawing on
                current_rs_y = int(lm[12].y * h_pix)
                current_rh_y = int(lm[24].y * h_pix)
                current_rw = (int(lm[16].x * w_pix), int(lm[16].y * h_pix))

                # ADDRESS LOCK (Frame 1)
                if anatomy_apex is None and lm[12].visibility > 0.8:
                    # Anchor the Plane Ceiling to the Shoulder and Slot to the Hip
                    shoulder_y_lock = current_rs_y
                    hip_y_lock = current_rh_y
                    # Project Apex forward based on body width
                    torso_w = abs(lm[12].x - lm[11].x) * w_pix
                    anatomy_apex = (int(lm[12].x * w_pix + (torso_w * 2.5)), int(hip_y_lock + (h_pix * 0.1)))

                # DRAWING THE CONE
                if anatomy_apex:
                    overlay = frame.copy()
                    # The V starts at the apex and fans out to the locked joint heights at X=0
                    pts = np.array([[anatomy_apex[0], anatomy_apex[1]], [0, shoulder_y_lock], [0, hip_y_lock + 100]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, anatomy_apex, (0, shoulder_y_lock), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(frame, anatomy_apex, (0, hip_y_lock + 100), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "PLANE CEILING", (50, shoulder_y_lock - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                    cv2.putText(frame, "THE SLOT", (50, hip_y_lock + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

                # YELLOW WRIST TRAIL
                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    # Add point to trail if moving
                    if not wrist_trail or np.linalg.norm(np.array(current_rw) - np.array(wrist_trail[-1])) > 5:
                        wrist_trail.append(current_rw)
                
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3, cv2.LINE_AA)

                # LAG CALCULATIONS
                s, e, w = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
                ba, bc = np.array(s) - np.array(e), np.array(w) - np.array(e)
                angle = np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1)))
                if is_downswing and lag_top is None: lag_top = int(angle)
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = int(angle)

            out.write(frame)

    cap.release(); out.release()
    web_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_path])
    return f"### 🏌️ Anatomy Plane Analysis\nTop Lag: {lag_top}° | Impact Lag: {lag_impact}°\nLocked to shoulder and hip heights to track vertical stability.", web_path
