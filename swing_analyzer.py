import cv2
import mediapipe as mp
import numpy as np
import tempfile
import subprocess
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_swing_plane_position(wrist, shoulder, ball):
    """Simple math to see if wrist is 'above' or 'below' the line between shoulder and ball."""
    # Vector from ball to shoulder
    v_line = np.array([shoulder[0] - ball[0], shoulder[1] - ball[1]])
    # Vector from ball to wrist
    v_wrist = np.array([wrist[0] - ball[0], wrist[1] - ball[1]])
    
    # Cross product gives the signed distance from the line
    return np.cross(v_line, v_wrist)

def analyze_diagnostic_swing(video_path, club_type=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 1. SETUP RAW RECORDER (Linux friendly)
    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(raw_tfile.name, fourcc, fps, (width, height))

    address_plane_line = None
    max_wrist_height = 1.0 
    is_downswing = False
    ott_detected = False
    address_head_y = None
    address_hip_x = None
    head_status = "STABLE"
    hip_status = "STABLE"
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Setup Plane Line at Address
                if address_plane_line is None:
                    r_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width), 
                                  int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
                    r_foot = (int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * width), 
                                  int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * height))
                    address_plane_line = (r_shoulder, r_foot)

                # DRAW: The Red Plane Line
                cv2.line(frame, address_plane_line[0], address_plane_line[1], (0, 0, 255), 3)

                # --- NEW DRAWING CODE STARTS HERE ---
                # 1. Head/Chin Box (Tracking vertical movement)
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                head_center = (int(nose.x * width), int(nose.y * height))
                
                cv2.circle(frame, head_center, 25, (0, 255, 0), 2) # Green circle for head

                # 2. Hip/Core Box (Tracking lateral sway)
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_center_x = int((l_hip.x + r_hip.x) / 2 * width)
                hip_center_y = int((l_hip.y + r_hip.y) / 2 * height)
                
                # Green box for hip stability
                cv2.rectangle(frame, 
                             (hip_center_x - 40, hip_center_y - 40), 
                             (hip_center_x + 40, hip_center_y + 40), 
                             (0, 255, 0), 2) 
                # --- NEW DRAWING CODE ENDS HERE ---

                # --- HEAD & HIP STABILITY LOGIC ---
                if address_head_y is None:
                    address_head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
                    address_hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2

                # Check Vertical Head Movement (Dipping or Lifting)
                curr_head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
                if curr_head_y > address_head_y + 0.03: # Threshold
                    head_status = "DIPPING"
                elif curr_head_y < address_head_y - 0.03:
                    head_status = "LIFTING"

                # Check Lateral Hip Movement (Swaying)
                curr_hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                if abs(curr_hip_x - address_hip_x) > 0.05:
                    hip_status = "SWAYING"

                # --- DRAW THE "POST-IT" STATUS BOARD ---
                # Draw a dark semi-transparent overlay for readability
                cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
                
                # Write the text
                cv2.putText(frame, f"HEAD: {head_status}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"HIPS: {hip_status}", (20, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Tracking Logic
                curr_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
                if curr_wrist_y < max_wrist_height:
                    max_wrist_height = curr_wrist_y
                elif not is_downswing and curr_wrist_y > (max_wrist_height + 0.05):
                    is_downswing = True

                if is_downswing:
                    wrist_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                    plane_score = calculate_swing_plane_position(wrist_pos, 
                        [address_plane_line[0][0]/width, address_plane_line[0][1]/height],
                        [address_plane_line[1][0]/width, address_plane_line[1][1]/height])
                    
                    if plane_score < -0.02: # Threshold for Over-the-Top
                        ott_detected = True
                        wrist_pixels = (int(wrist_pos[0]*width), int(wrist_pos[1]*height))
                        cv2.circle(frame, wrist_pixels, 10, (0, 165, 255), -1) # Orange Alert

            out.write(frame)

        cap.release()
        out.release()

    # 2. CONVERSION STEP: AVI -> Web-Optimized MP4
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    cmd = [
        'ffmpeg', '-y', '-i', raw_tfile.name,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', 'faststart', web_tfile.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup raw file
    if os.path.exists(raw_tfile.name):
        os.remove(raw_tfile.name)

    feedback = "Technical Swing Plane Analysis:"
    if ott_detected:
        feedback += "\n\n⚠️ **OVER-THE-TOP:** Hands moved outside the plane."
    else:
        feedback += "\n\n✅ **ON-PLANE:** Hands stayed inside the plane."

    return feedback, web_tfile.name

# 1. Update the signature to accept the new data from web_coach
def analyze_wrist_action(video_path, ball_coords=None, start_frame=0):
    # 1. INITIALIZATION & VIDEO SETUP
    v_ball_pos = ball_coords 
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 2. PERSISTENT STATE (Defined ONLY ONCE)
    backswing_top_y = None
    forward_bot_y = None
    is_downswing = False
    max_wrist_height = 1.0  # 1.0 is the bottom of the frame
    lag_at_top = None
    lag_at_impact = None
    impact_locked = False
    wrist_trail = []
    
    # 3. SETUP RAW RECORDER
    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(raw_tfile.name, fourcc, fps, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
        
            # --- 2. INSIDE THE LOOP (Process Pose) ---
            # ... (Landmark detection happens here) ...
                    
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
            
                # Lock in the heights ONLY ONCE at the start of the swing
                if backswing_top_y is None:
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                
                    # Convert to pixel heights
                    height, width, _ = frame.shape
                    backswing_top_y = int(elbow[1] * height)
                    forward_bot_y = int(wrist[1] * height)

            # --- 3. DRAW THE CONE ---
            # This now has the heights it needs to stay visible!
            if v_ball_pos is not None and backswing_top_y is not None:
                overlay = frame.copy()
                pts = np.array([
                    [v_ball_pos[0], v_ball_pos[1]], 
                    [0, backswing_top_y - 120],     
                    [0, forward_bot_y + 80]         
                ], np.int32)
            
                cv2.fillPoly(overlay, [pts], (220, 220, 220))
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
                # Draw the Boundary Lines in Black
                cv2.line(frame, (v_ball_pos[0], v_ball_pos[1]), (0, backswing_top_y - 120), (0, 0, 0), 2, cv2.LINE_AA) 
                cv2.line(frame, (v_ball_pos[0], v_ball_pos[1]), (0, forward_bot_y + 80), (0, 0, 0), 2, cv2.LINE_AA)   

                # Labels
                cv2.putText(frame, "PLANE CEILING", (50, backswing_top_y - 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "THE SLOT", (50, forward_bot_y + 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Small white dot at the ball position
                cv2.circle(frame, v_ball_pos, 5, (255, 255, 255), -1)
                
            # --- TRANSITION DETECTION ---
            curr_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            if not is_downswing:
                if curr_wrist_y < max_wrist_height:
                    max_wrist_height = curr_wrist_y 
                elif curr_wrist_y > (max_wrist_height + 0.05):
                    is_downswing = True

                # B. ONLY add to trail during Backswing
                if not is_downswing and wrist_confidence > 0.8:
                    dist = np.linalg.norm(np.array(p3) - np.array(wrist_trail[-1])) if wrist_trail else 0
                    if not wrist_trail or (5 < dist < 50):
                        wrist_trail.append(p3)
                
                # C. Draw the Trail
                if len(wrist_trail) > 1:
                    for i in range(1, len(wrist_trail)):
                        cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 2)

                # D. Status Indicator (New)
                status_text = "BACKSWING TRACKING..." if not is_downswing else "GUIDE ACTIVE"
                status_color = (0, 255, 255) if not is_downswing else (0, 255, 0)
                cv2.putText(frame, status_text, (20, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # --- SKELETON ---
                cv2.line(frame, p1, p2, (255, 255, 255), 2)
                cv2.line(frame, p2, p3, (255, 255, 255), 2)
                cv2.circle(frame, p3, 10, (0, 255, 255), -1)

                # --- LAG LANDMARK CAPTURE ---
                # 1. Calculate the raw hinge angle at the elbow
                ba = np.array(shoulder) - np.array(elbow)
                bc = np.array(wrist) - np.array(elbow)
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                raw_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                # 2. Capture: Lag at the Top (Transition)
                if is_downswing and lag_at_top is None:
                    lag_at_top = int(raw_angle)

                # 3. Capture: Lag at Impact (With Absolute Lockout)
                # If already locked, the code skips this entire block forever
                if is_downswing and curr_wrist_y > 0.5 and not impact_locked:
                    if wrist_confidence > 0.6: # Slightly higher confidence for impact
                        
                        # A. Record the lag as long as hands are moving DOWN
                        if lag_at_impact is None or curr_wrist_y >= max_wrist_height:
                            lag_at_impact = int(raw_angle)
                            max_wrist_height = curr_wrist_y 
                        
                        # B. THE DEADBOLT: Sensitive Trigger
                        # If hands rise even 5% (0.05) from the bottom, lock the door.
                        if curr_wrist_y < (max_wrist_height - 0.05):
                            impact_locked = True
                            
                # --- DISPLAY RECAP LABELS ---
                if lag_at_top is not None:
                    # Top Color: Lower is sharper/better
                    if lag_at_top < 90: top_color = (0, 255, 0)      # Green
                    elif lag_at_top < 110: top_color = (0, 165, 255) # Orange
                    else: top_color = (0, 0, 255)                    # Red
                    
                    cv2.putText(frame, f"TOP LAG: {lag_at_top} DEG", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, top_color, 2)
                
                # --- DISPLAY IMPACT RECAP ---
                if lag_at_impact is not None:
                    # Impact Color Logic: Higher is straighter/better
                    if lag_at_impact > 165: 
                        imp_color = (0, 255, 0)      # Green (Elite Extension)
                    elif lag_at_impact > 145: 
                        imp_color = (0, 165, 255)    # Orange (Solid)
                    else: 
                        imp_color = (0, 0, 255)      # Red (Casting/Scooping)
                    
                    cv2.putText(frame, f"IMPACT LAG: {lag_at_impact} DEG", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, imp_color, 2)

                # --- [A] D. Status Indicator (Moving to the bottom for overlay) ---
                status_text = "BACKSWING TRACKING..." if not is_downswing else "GUIDE ACTIVE"
                status_color = (0, 255, 255) if not is_downswing else (0, 255, 0)
                cv2.putText(frame, status_text, (20, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # --- [B] IDEAL RANGES CHEAT SHEET (Upper Right) ---
                # Background box for readability
                cv2.rectangle(frame, (width-230, 10), (width-10, 85), (50, 50, 50), -1)
                cv2.putText(frame, "IDEAL RANGES:", (width-220, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, "Top Lag: < 90 deg", (width-220, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "Impact: > 165 deg", (width-220, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # --- [C] WRITE FRAME ---
                out.write(frame)

        cap.release()
        out.release()

    # 2. CONVERSION
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(raw_tfile.name):
        os.remove(raw_tfile.name)

    # Create the summary text with a fall-back if stats weren't captured
    top_stat = f"{lag_at_top}°" if lag_at_top else "Not captured"
    impact_stat = f"{lag_at_impact}°" if lag_at_impact else "Not captured"

    summary = (
        f"### 🏌️ Wrist Lab Analysis\n"
        f"**Top of Swing Lag (Hinge):** {top_stat}\n"
        f"**Impact Lag (Release):** {impact_stat}\n\n"
        f"**Coach's Note:** Aim for < 90° at the top and > 165° at impact. "
        f"If your impact lag is low (Red), focus on 'extending' your arms through the ball!"
    )

    return summary, web_tfile.name




























