import cv2
import mediapipe as mp
import numpy as np
import tempfile

# STABLE IMPORTS: Reaching directly into the python solutions
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# INITIALIZE POSE: Use the mp_pose we just imported above
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1  # Optimized for cloud speed
)

# No need for the "mp.solutions" lines anymore!

def analyze_diagnostic_swing(video_path):
    print("🦴 Booting up the X-Ray Diagnostic Lab engine...")
    cap = cv2.VideoCapture(video_path)
    
    # 1. Safely read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read video file.")
        return None
        
    # 2. Shrink the video for App Speed
    original_height, original_width = first_frame.shape[:2]
    scale = 640.0 / float(original_height) 
    new_width = int(original_width * scale)
    new_height = 640
    
    # 3. Reset the video back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Setup the VideoWriter for the Cloud (Tempfile + Apple MP4 Codec)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0 or fps > 60: 
        fps = 30 
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = temp_file.name
    temp_file.close() 
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_width, new_height))
    
    # Terminal Trackers
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎯 Video loaded! Total frames to process: {total_frames}")
    
    initial_nose = None
    initial_hip = None
    initial_wrist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 15 == 0:
            print(f"⏳ Processing frame {frame_count} of {total_frames}...")

        # Resize the frame immediately
        image = cv2.resize(frame, (new_width, new_height))
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            h, w, c = image.shape
            
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_px = (int(nose.x * w), int(nose.y * h))
            
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            hip_px = (int(((l_hip.x + r_hip.x) / 2) * w), int(((l_hip.y + r_hip.y) / 2) * h))
            
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            wrist_px = (int(l_wrist.x * w), int(l_wrist.y * h))

            # Lock in Address Position
            if initial_nose is None:
                initial_nose = nose_px
                initial_hip = hip_px
                initial_wrist = wrist_px
                
            # THE SWING PLANE LINE 
            dx = initial_hip[0] - initial_wrist[0]
            dy = initial_hip[1] - initial_wrist[1]
            p1 = (initial_wrist[0] - dx, initial_wrist[1] - dy) 
            p2 = (initial_hip[0] + dx * 2, initial_hip[1] + dy * 2) 
            cv2.line(image, p1, p2, (0, 165, 255), 3) 

            # STATIC Stability Boxes
            cv2.rectangle(image, (initial_nose[0] - 30, initial_nose[1] - 30), 
                                 (initial_nose[0] + 30, initial_nose[1] + 30), (0, 255, 255), 2)
            cv2.rectangle(image, (initial_hip[0] - 50, initial_hip[1] - 40), 
                                 (initial_hip[0] + 50, initial_hip[1] + 40), (255, 0, 255), 2)

            # MOVING Body Trackers
            cv2.circle(image, nose_px, 8, (0, 255, 255), -1) 
            cv2.circle(image, hip_px, 10, (255, 0, 255), -1) 

            # ERROR-PROOF SWAY ALERT 
            sway_x = nose_px[0] < initial_nose[0] - 30 or nose_px[0] > initial_nose[0] + 30
            sway_y = nose_px[1] < initial_nose[1] - 30 or nose_px[1] > initial_nose[1] + 30
            
            if sway_x or sway_y:
                cv2.putText(image, "SWAY ALERT!", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        # WRITE TO FILE 
        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete! Sending X-Ray video to your iPhone.")
    
    return out_path

# --- RUN IT SECTION ---
if __name__ == "__main__":
    print("\n" + "="*50)
    video_file = input("🎬 Drag and drop your video file here: ")
    video_file = video_file.strip('"').strip("'").strip()

    analyze_diagnostic_swing(video_file)
