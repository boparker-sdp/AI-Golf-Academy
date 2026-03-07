import os
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# CLEAN CLOUD IMPORTS
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    model_complexity=1
)

def calculate_angle(a, b, c):
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def drill_coach(video_path, club_type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Track swing phase and captured hinge values
    is_downswing = False
    top_wrist_y = 1.0          # min wrist height before transition into downswing
    down_max_wrist_y = 0.0     # deepest point of the arc during downswing
    impact_locked = False      # once True, stop updating impact hinge
    lag_top = None             # hinge at top of backswing
    lag_impact = None          # hinge at impact (bottom of arc)

    # Use MP4 container with the universal mp4v codec
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tfile.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Target Lead Arm (Assuming Right-Handed Golfer)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Detect top of backswing and bottom-of-arc (impact proxy) using wrist height
            wrist_y = wrist[1]  # normalized y (0=top, 1=bottom)
            if not is_downswing:
                # Track the highest point (smallest y) of the wrist
                if wrist_y < top_wrist_y:
                    top_wrist_y = wrist_y
                # When the wrist starts coming down past a small buffer, we are in the downswing
                elif wrist_y > (top_wrist_y + 0.05):
                    is_downswing = True

            if is_downswing and lag_top is None:
                lag_top = int(angle)

            # During downswing, capture the hinge at the deepest point (bottom of arc)
            if is_downswing and not impact_locked:
                if wrist_y > down_max_wrist_y:
                    # Wrist is still moving down; update bottom-of-arc and impact hinge
                    down_max_wrist_y = wrist_y
                    lag_impact = int(angle)
                elif wrist_y < (down_max_wrist_y - 0.02):
                    # Wrist has started moving back up after the deepest point -> lock impact
                    impact_locked = True

            # Draw Angle on Screen
            cv2.putText(frame, f"Hinge: {int(angle)}deg", 
                        (int(elbow[0]*width), int(elbow[1]*height)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Alert if wrist is "Bowing" (Closing the face)
            if angle > 170:
                cv2.putText(frame, "SHUT FACE ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Show captured hinge values in the upper-right corner
            if lag_top is not None:
                cv2.putText(
                    frame,
                    f"HINGE AT TOP: {int(lag_top)}deg",
                    (width - 450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # If we successfully locked an impact value, show it.
            # Otherwise, once we are in the downswing, give camera-angle guidance instead of a bogus number.
            if lag_impact is not None:
                cv2.putText(
                    frame,
                    f"HINGE AT IMPCT: {int(lag_impact)}deg",
                    (width - 450, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif is_downswing:
                cv2.putText(
                    frame,
                    "TAKE VIDEO FACING GOLFER",
                    (width - 650, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()

    # The Universal Translator: Convert OpenCV's mp4v into a browser-friendly H.264 MP4
    final_video_path = tfile.name.replace('.mp4', '_h264.mp4')
    os.system(f"ffmpeg -y -i {tfile.name} -vcodec libx264 {final_video_path}")

    return final_video_path





