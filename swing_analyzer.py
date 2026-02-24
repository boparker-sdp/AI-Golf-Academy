import cv2
import mediapipe as mp
import numpy as np
import tempfile

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_swing_plane_position(wrist_coords, shoulder_coords, ball_coords):
    """
    Calculates if the wrist is above or below the swing plane line.
    (Cross-product math)
    """
    x0, y0 = wrist_coords
    x1, y1 = shoulder_coords
    x2, y2 = ball_coords
    position = (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1)
    return position

def analyze_diagnostic_swing(video_path, club_type=None):
    cap = cv2.VideoCapture(video_path)
    
    # Initialize V2 Plane Tracking Variables
    address_plane_line = None
    max_wrist_height = 1.0 
    is_downswing = False
    ott_detected = False
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 1. SETUP: Capture Shoulder-to-Ball Plane at start
                if address_plane_line is None:
                    # Using right shoulder and right foot index as proxy for ball position
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                    ball_proxy = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
                    address_plane_line = (r_shoulder, ball_proxy)

                # 2. TRIGGER: Detect Transition to Downswing
                curr_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
                if curr_wrist_y < max_wrist_height:
                    max_wrist_height = curr_wrist_y
                elif not is_downswing and curr_wrist_y > (max_wrist_height + 0.05):
                    is_downswing = True

                # 3. ANALYSIS: Check if wrist crosses 'Over the Top'
                if is_downswing:
                    wrist_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                    
                    plane_score = calculate_swing_plane_position(
                        wrist_pos, address_plane_line[0], address_plane_line[1]
                    )
                    
                    # If score goes negative, the hand is 'above' the plane
                    if plane_score < -0.02: 
                        ott_detected = True

        cap.release()

    # Updated Final Feedback Logic in swing_analyzer.py
        feedback = "Technical Swing Plane Analysis:"
    if ott_detected:
        feedback += "\n\n⚠️ **OVER-THE-TOP:** Your hands moved outside the shoulder-to-ball plane during the downswing. This 'Outside-In' path is the primary cause of a slice."
    else:
        feedback += "\n\n✅ **ON-PLANE:** Your hands stayed inside the plane. This allows for an 'Inside-Out' or neutral path."

    return feedback




