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
    # --- 1. INITIALIZATION ---
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Anatomy Anchors (Set at Address)
    anatomy_apex = None
    shoulder_anchor = None
    hip_anchor = None
    
    # State Variables
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
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Get exact frame dimensions for this specific frame
                h_pix, w_pix = frame.shape[:2]

                # Convert shoulder and hip to pixel coordinates
                # We use the RIGHT side as the camera is facing your back/side
                rs_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w_pix)
                rs_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h_pix)
                rh_x = int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w_pix)
                rh_y = int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h_pix)

                # --- LOCK ANATOMY AT ADDRESS (First Frame) ---
                if anatomy_apex is None:
                    shoulder_anchor = (rs_x, rs_y)
                    hip_anchor = (rh_x, rh_y)
                    
                    # Project the Apex forward into the 'Ball Zone'
                    # We move 30% of the screen width forward from your shoulder
                    apex_x = int(rs_x + (w_pix * 0.30))
                    # We set the apex height to be slightly below your hip
                    apex_y = int(rh_y + (h_pix * 0.15))
                    anatomy_apex = (apex_x, apex_y)

                # --- DRAW THE ANATOMY CONE ---
                overlay = frame.copy()
                pts = np.array([
                    [anatomy_apex[0], anatomy_apex[1]], # The "Ball"
                    [0, shoulder_anchor[1]],           # Ceiling from Shoulder height
                    [0, hip_anchor[1] + 100]           # Slot from Hip height
                ], np.int32)
                
                cv2.fillPoly(overlay, [pts], (220, 220, 220))
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                
                # Draw sharp lines from Apex back to the left edge
                cv2.line(frame, anatomy_apex, (0, shoulder_anchor[1]), (0, 0, 0), 2, cv2.LINE_AA)
                cv2.line(frame, anatomy_apex, (0, hip_anchor[1] + 100), (0, 0, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame, "PLANE CEILING", (50, shoulder_anchor[1] - 20), 0, 0.6, (0,0,0), 1)
                cv2.putText(frame, "THE SLOT", (50, hip_anchor[1] + 80), 0, 0.6, (0,0,0), 1)

                # (Keep your Wrist Trail and Lag Math logic below this...)
                # ... [Wrist Trail & Lag Code] ...

            out.write(frame)

    cap.release()
    out.release()
    
    # ... [FFMPEG Conversion Code] ...
    # CONVERSION & SUMMARY (Keep your existing code here)
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)
    
    top_stat = f"{lag_at_top}°" if lag_at_top else "Not captured"
    impact_stat = f"{lag_at_impact}°" if lag_at_impact else "Not captured"
    summary = f"### 🏌️ Wrist Lab Analysis\n**Top of Swing Lag:** {top_stat}\n**Impact Lag:** {impact_stat}\n\n**Note:** Aim for < 90° top and > 165° impact."
    
    return summary, web_tfile.name

































