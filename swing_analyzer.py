import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

mp_pose = mp.solutions.pose

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    h_pix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_pix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frozen_apex = None
    shoulder_y_lock = None
    hip_y_lock = None
    confidence_frames = 0 # Wait for a stable stance
    
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
                
                # Live Data
                cur_rs_x, cur_rs_y = int(lm[12].x * w_pix), int(lm[12].y * h_pix)
                cur_rh_y = int(lm[24].y * h_pix)
                cur_rw_x, cur_rw_y = int(lm[16].x * w_pix), int(lm[16].y * h_pix)

                # --- THE INTELLIGENT LOCK ---
                if shoulder_y_lock is None and lm[12].visibility > 0.9:
                    confidence_frames += 1
                    if confidence_frames > 10: # Wait for ~1/3 of a second of stability
                        # 1. Higher Anchors: Moving to the actual shoulder top and high hip
                        shoulder_y_lock = cur_rs_y - 180 
                        hip_y_lock = cur_rh_y - 80 # Lift hip anchor significantly
                        
                        # 2. Decoupled Slopes
                        dx = cur_rw_x - cur_rs_x
                        dy = cur_rw_y - cur_rs_y
                        
                        if dx != 0:
                            arm_slope = dy / dx
                            body_w = abs(lm[12].x - lm[11].x) * w_pix
                            
                            # APEX: Project the intersection further out (3.5 body widths)
                            # so the lines don't cross too close to the body
                            apex_x = int(cur_rs_x + (body_w * 3.5))
                            apex_y = int(shoulder_y_lock + (arm_slope * (body_w * 3.5)))
                            frozen_apex = (apex_x, apex_y)

                # --- DRAWING THE PLANE ---
                if frozen_apex is not None:
                    overlay = frame.copy()
                    # Use the 'frozen_apex' for the top line, 
                    # but a flatter projection for the bottom line
                    bottom_line_exit_y = hip_y_lock + 100 
                    
                    pts = np.array([[frozen_apex[0], frozen_apex[1]], [0, shoulder_y_lock], [0, bottom_line_exit_y]], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    
                    # Top line follows arm angle to the Apex
                    cv2.line(frame, frozen_apex, (0, shoulder_y_lock), (0, 0, 0), 2, cv2.LINE_AA)
                    # Bottom line creates the 'Slot' (Projected flatter)
                    cv2.line(frame, frozen_apex, (0, bottom_line_exit_y), (0, 0, 0), 2, cv2.LINE_AA)

                # Trail & Lag Math (Keep existing working logic)
                if not is_downswing:
                    if lm[16].y < max_h: max_h = lm[16].y
                    elif lm[16].y > (max_h + 0.05): is_downswing = True
                    wrist_trail.append((cur_rw_x, cur_rw_y))
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3, cv2.LINE_AA)

                s, e, w = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
                ba, bc = np.array(s) - np.array(e), np.array(w) - np.array(e)
                ang = np.degrees(np.arccos(np.clip(np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1, 1)))
                if is_downswing and lag_top is None: lag_top = int(ang)
                if is_downswing and lm[16].y > 0.5 and lag_impact is None: lag_impact = int(ang)

            out.write(frame)

    cap.release(); out.release()
    web_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_path])
    return f"### 🏌️ Anatomy Plane Analysis\nTop Lag: {lag_top}° | Impact Lag: {lag_impact}°", web_path
