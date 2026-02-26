import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

# Initialize MediaPipe Pose
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
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Syncing logic identical to X-ray for diagnostic consistency
                if address_head_y is None:
                    address_head_y = lm[mp_pose.PoseLandmark.NOSE].y
                    address_hip_x = (lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2

                curr_head_y = lm[mp_pose.PoseLandmark.NOSE].y
                if curr_head_y > address_head_y + 0.03: head_status = "DIPPING"
                elif curr_head_y < address_head_y - 0.03: head_status = "LIFTING"

                curr_hip_x = (lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                if abs(curr_hip_x - address_hip_x) > 0.05: hip_status = "SWAYING"

                cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"HEAD: {head_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"HIPS: {hip_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(frame)

    cap.release()
    out.release()
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "Diagnostic Scan Complete.", web_tfile.name

def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    """Anatomy-based plane lab with synchronized shoulder anchors and wrist trails."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # State Persistence
    anatomy_apex = None
    shoulder_anchor_y = None
    hip_anchor_y = None
    is_downswing = False
    max_wrist_height = 1.0
    lag_at_top, lag_at_impact = None, None
    impact_locked = False
    wrist_trail = []

    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(raw_tfile.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h_pix, w_pix, _ = frame.shape
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Live Pixel Mapping (X-Ray Sync)
                curr_rs_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w_pix)
                curr_rs_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h_pix)
                curr_rh_y = int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h_pix)
                curr_rw = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w_pix), 
                           int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h_pix))

                # 1. ANCHOR PLANE AT ADDRESS
                if anatomy_apex is None and lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.8:
                    # Subtracting 40 pixels from shoulder joint center to hit the top of the shoulder muscle
                    shoulder_anchor_y = curr_rs_y - 40
                    hip_anchor_y = curr_rh_y
                    
                    # Slope Math (Shoulder to Wrist)
                    dx = curr_rw[0] - curr_rs_x
                    dy = curr_rw[1] - curr_rs_y # Use actual joint center for trajectory math
                    
                    if dx != 0:
                        slope = dy / dx
                        # Calculate body width to project ball position forward
                        body_w = abs(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x) * w_pix
                        apex_x = int(curr_rs_x + (body_w * 2.8))
                        # Project ceiling line from the nudged shoulder height
                        apex_y = int(shoulder_anchor_y + (slope * (body_w * 2.8)))
                        anatomy_apex = (apex_x, apex_y)

                # 2. DRAW ANATOMY CONE
                if anatomy_apex is not None:
                    overlay = frame.copy()
                    pts = np.array([
                        [anatomy_apex[0], anatomy_apex[1]], 
                        [0, shoulder_anchor_y], 
                        [0, hip_anchor_y + 120]
                    ], np.int32)
                    
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, anatomy_apex, (0, shoulder_anchor_y), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(frame, anatomy_apex, (0, hip_anchor_y + 120), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "PLANE CEILING", (50, shoulder_anchor_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, "THE SLOT", (50, hip_anchor_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

                # 3. WRIST TRAIL & TRANSITION
                curr_wrist_y_norm = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
                if not is_downswing:
                    if curr_wrist_y_norm < max_wrist_height:
                        max_wrist_height = curr_wrist_y_norm
                    elif curr_wrist_y_norm > (max_wrist_height + 0.05):
                        is_downswing = True
                    
                    # Backswing Yellow Trace
                    if not wrist_trail or np.linalg.norm(np.array(curr_rw) - np.array(wrist_trail[-1])) > 5:
                        wrist_trail.append(curr_rw)
                
                if len(wrist_trail) > 1:
                    for i in range(1, len(wrist_trail)):
                        cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3, cv2.LINE_AA)

                # 4. LAG CALCULATIONS
                s_coords = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                e_coords = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                w_coords = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                
                ba, bc = np.array(s_coords) - np.array(e_coords), np.array(w_coords) - np.array(e_coords)
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                raw_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                if is_downswing and lag_at_top is None: lag_at_top = int(raw_angle)
                if is_downswing and curr_wrist_y_norm > 0.5 and not impact_locked:
                    if lag_at_impact is None or curr_wrist_y_norm >= max_wrist_height:
                        lag_at_impact = int(raw_angle)
                    if curr_wrist_y_norm < (max_wrist_height - 0.05): impact_locked = True

            out.write(frame)

    cap.release()
    out.release()
    
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)

    sum_text = f"### 🏌️ Anatomy Plane Analysis\n**Top Lag:** {lag_at_top if lag_at_top else 'N/A'}° | **Impact Lag:** {lag_at_impact if lag_at_impact else 'N/A'}°"
    return sum_text, web_tfile.name
