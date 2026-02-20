import cv2
import mediapipe as mp
import numpy as np
import tempfile

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def drill_coach(video_path):
    print("⌚ Booting up the Wrist Lab engine...")
    cap = cv2.VideoCapture(video_path)
    
    # 1. Safely read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read video file.")
        return None
        
    # 2. Resize for Speed
    original_height, original_width = first_frame.shape[:2]
    scale = 640.0 / float(original_height) 
    new_width = int(original_width * scale)
    new_height = 640
    
    # 3. Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 4. Setup Cloud-Safe MP4 VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0 or fps > 60: 
        fps = 30
        
    # Generate a unique temp file name with .mp4 suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = temp_file.name
    temp_file.close() 
    
    # Use the Apple-friendly 'avc1' codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎯 Video loaded! Total frames to process: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 15 == 0:
            print(f"⏳ Processing frame {frame_count} of {total_frames}...")

        image = cv2.resize(frame, (new_width, new_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Tracking LEFT side (adjust to RIGHT if you are a lefty)
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                
                angle = calculate_angle(elbow, wrist, index)
                
                h, w, c = image.shape
                wrist_px = tuple(np.multiply(wrist, [w, h]).astype(int))
                
                cv2.putText(image, str(int(angle)), wrist_px, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
                if angle <= 140:
                    cv2.putText(image, 'GREAT HINGE!', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                elif angle > 160:
                    cv2.putText(image, 'TOO FLAT', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        except:
            pass 
            
        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete! Sending Wrist video to your iPhone.")
    
    return out_path

if __name__ == "__main__":
    print("\n" + "="*50)
    video_file = input("🎬 Drag and drop your video file here: ")
    video_file = video_file.strip('"').strip("'").strip()
    drill_coach(video_file)