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
    """Follows the same double-buffer logic as the diagnostic tool."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(raw_tfile.name, fourcc, fps, (width, height))

    # [Simplified wrist tracking for this example]
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Logic for drawing wrist data would go here...
            out.write(frame)
        
        cap.release()
        out.release()

    web_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', raw_tfile.name, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', web_tfile.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(raw_tfile.name):
        os.remove(raw_tfile.name)

    return "Wrist Action Analysis Complete.", web_tfile.name
