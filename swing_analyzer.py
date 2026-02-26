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
if results.pose_landmarks:
                h_pix, w_pix, _ = frame.shape
                lm = results.pose_landmarks.landmark
                
                # 1. LIVE ANCHORS (Just like your X-ray code)
                # These move with you to prevent the "misalignment"
                curr_rs_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w_pix)
                curr_rs_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h_pix)
                curr_rh_y = int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h_pix)
                curr_rw = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w_pix), 
                           int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h_pix))

                # 2. LOCK THE "IDEAL" PLANE (Only once at address)
                if anatomy_apex is None and lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.7:
                    shoulder_anchor_y = curr_rs_y
                    hip_anchor_y = curr_rh_y
                    # Project the Apex forward from your current shoulder
                    anatomy_apex_x = int(curr_rs_x + (w_pix * 0.35)) 
                    anatomy_apex_y = int(hip_anchor_y + (h_pix * 0.10))
                    anatomy_apex = (anatomy_apex_x, anatomy_apex_y)

                # 3. DRAW THE CONE (Anchored to LIVE Body X, but STATIC Heights)
                if anatomy_apex is not None:
                    overlay = frame.copy()
                    # We use the STATIC apex but anchor the left side to X=0
                    pts = np.array([
                        [anatomy_apex[0], anatomy_apex[1]], 
                        [0, shoulder_anchor_y],           
                        [0, hip_anchor_y + 100]            
                    ], np.int32)
                    
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.line(frame, anatomy_apex, (0, shoulder_anchor_y), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(frame, anatomy_apex, (0, hip_anchor_y + 100), (0, 0, 0), 2, cv2.LINE_AA)

                # 4. RESTORE YELLOW BACKSWING TRAIL
                # Only track during the backswing phase
                if not is_downswing:
                    if not wrist_trail or np.linalg.norm(np.array(curr_rw) - np.array(wrist_trail[-1])) > 5:
                        wrist_trail.append(curr_rw)
                
                if len(wrist_trail) > 1:
                    for i in range(1, len(wrist_trail)):
                        cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 3, cv2.LINE_AA)
                    
                    # Labels using the anchored Y-coordinates
                    cv2.putText(frame, "PLANE CEILING", (50, shoulder_anchor_y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, "THE SLOT", (50, hip_anchor_y + 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

            out.write(frame)

    cap.release()
    out.release()
    
    # --- 4. CONVERSION & FEEDBACK ---
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', 
                    '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)

    summary = "### 🏌️ Anatomy Plane Analysis\nCone anchored to shoulder and hip. Use this to track if you dip (Fat) or lift (Thin) during the strike."
    return summary, web_tfile.name






















