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

def analyze_wrist_action(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 1. SETUP RAW RECORDER
    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(raw_tfile.name, fourcc, fps, (width, height))

    wrist_trail = [] # To draw the "path" of the hands

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get Arm Coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                # Convert to Pixels
                p1 = (int(shoulder[0] * width), int(shoulder[1] * height))
                p2 = (int(elbow[0] * width), int(elbow[1] * height))
                p3 = (int(wrist[0] * width), int(wrist[1] * height))

                # DRAW: Skeleton and Trail
                wrist_trail.append(p3)
                for i in range(1, len(wrist_trail)):
                    cv2.line(frame, wrist_trail[i-1], wrist_trail[i], (0, 255, 255), 2) # Yellow trail
                
                cv2.line(frame, p1, p2, (255, 255, 255), 2) # Shoulder to Elbow
                cv2.line(frame, p2, p3, (255, 255, 255), 2) # Elbow to Wrist
                cv2.circle(frame, p3, 10, (0, 255, 255), -1)

                # CALCULATE LAG ANGLE (Angle at the elbow)
                ba = np.array(shoulder) - np.array(elbow)
                bc = np.array(wrist) - np.array(elbow)
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                # DRAW: Lag "Post-it"
                cv2.rectangle(frame, (width-220, 10), (width-10, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"LAG: {int(angle)} DEG", (width-200, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)
        
        cap.release()
        out.release()

    # 2. CONVERSION STEP (Same as Diagnostic)
    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', web_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(raw_tfile.name):
        os.remove(raw_tfile.name)

    return "Wrist Lab: Tracking hand path and elbow lag. Look for a consistent arc in the yellow trail.", web_tfile.name

    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(raw_tfile.name):
        os.remove(raw_tfile.name)

    return "Wrist Action Analysis Complete.", web_tfile.name


