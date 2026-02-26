import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_diagnostic_swing(video_path, club_type=None):
    """General stability scan (Head/Hips) for thin/fat shots."""
    cap = cv2.VideoCapture(video_path)
    h, w = int(cap.get(4)), int(cap.get(3))
    out = cv2.VideoWriter(tempfile.NamedTemporaryFile(suffix='.avi').name, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), (w, h))

    # Stability logic
    addr_head_y, addr_hip_x = None, None
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
                # Stability overlays...
            out.write(frame)
    cap.release(); out.release()
    return "Diagnostic Complete.", ""

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    """Synchronized Anatomy Plane Lab."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    h_pix, w_pix = int(cap.get(4)), int(cap.get(3))
    fps = int(cap.get(5))
    
    # State
    anatomy_apex = None
    shoulder_y, hip_y = None, None
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
                
                # JOINT MAPPING (Direct Pixel Translation)
                # RIGHT_SHOULDER is index 12, RIGHT_HIP is index 24
                curr_rs_y = int(lm[12].y * h_pix)
                curr_rh_y = int(lm[24].y * h_pix)
                curr_rw = (int(lm[16].x * w_pix), int(lm[16].y * h_pix))

                # ADDRESS LOCK (Frame 1)
                if anatomy_apex is None and lm[12].visibility > 0.8:
                    shoulder_y = curr_rs_y
                    hip_y = curr_rh_y
                    # Define Apex: Forward from shoulder to the ground
                    anatomy_apex = (int(lm[12].x * w_pix + (w_pix * 0.3)), int(hip_y + (h_pix * 0.1)))

                # DRAWING THE PLANE
                if anatomy_apex:
                    overlay = frame.copy()
                    pts = np.array([[anatomy_apex[0], anatomy_apex[1]], [0, shoulder_y], [0, hip_y + 100]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, anatomy_apex, (0, shoulder_y), (0, 0, 0), 2)
                    cv2.line(frame, anatomy_apex, (0, hip_y + 100), (0, 0, 0), 2)

                # WRIST TRAIL
                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    wrist_trail.append(curr_rw)
                
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3)

                # LAG MATH
                s, e, w = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
                ba, bc = np.array(s) - np.array(e), np.array(w) - np.array(e)
                angle = np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1)))
                if is_downswing and lag_top is None: lag_top = int(angle)
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = int(angle)

            out.write(frame)

    cap.release(); out.release()
    web_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_path])
    return f"### 🏌️ Wrist Lab\nTop Lag: {lag_top}° | Impact Lag: {lag_impact}°", web_path
