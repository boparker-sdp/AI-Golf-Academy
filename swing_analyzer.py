import os
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# STANDARD IMPORTS (The correct way for Streamlit Cloud)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose Engine
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1
)

def analyze_diagnostic_swing(video_path, club_type):
    print("🦴 Booting up the X-Ray Diagnostic Lab...")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temp file for output
    # Use MP4 container with the universal mp4v codec
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tfile.name, fourcc, fps, (width, height))

    address_plane_drawn = False
    plane_line = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- 1. SETUP SHAFT PLANE (ADDRESS) ---
            if not address_plane_drawn:
                # Hip to Hand line
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                plane_line = (
                    (int(hip.x * width), int(hip.y * height)),
                    (int(hand.x * width), int(hand.y * height))
                )
                address_plane_drawn = True

            # --- 2. DRAW STABILITY BOXES ---
            # Hip Box
            rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            cv2.rectangle(frame, (int(rh.x*width)-40, int(rh.y*height)-40), 
                          (int(rh.x*width)+40, int(rh.y*height)+40), (255, 255, 0), 2)
            
            # Head Box
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            cv2.rectangle(frame, (int(nose.x*width)-30, int(nose.y*height)-30), 
                          (int(nose.x*width)+30, int(nose.y*height)+30), (0, 255, 255), 2)

            # --- 3. DRAW PLANE LINE ---
            if plane_line:
                cv2.line(frame, plane_line[0], plane_line[1], (0, 165, 255), 3)

            # --- 4. DRAW SKELETON ---
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()

    # The Universal Translator: Convert OpenCV's mp4v into a browser-friendly H.264 MP4
    final_video_path = tfile.name.replace('.mp4', '_h264.mp4')
    os.system(f"ffmpeg -y -i {tfile.name} -vcodec libx264 {final_video_path}")

    return final_video_path








