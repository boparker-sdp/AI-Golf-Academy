import cv2
import mediapipe as mp
import numpy as np
import tempfile
import shutil
import subprocess
import math

mp_pose = mp.solutions.pose

def analyze_foundation_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    fs, thick = h / 1000, int(2 * (h / 1000))

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))

    # --- STATE VARIABLES ---
    addr_head_y, addr_hip_x = None, None
    head_stable, hip_stable = True, True
    is_downswing, max_w_y = False, 1.0
    lag_top, lag_impact = None, None
    impact_locked = False
    
    # NEW: Cone Locking Logic
    locked_apex = None
    locked_top_end = None
    locked_bottom_end = None

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # Setup Address Baselines
                if addr_head_y is None:
                    addr_head_y, addr_hip_x = lm[0].y, (lm[23].x + lm[24].x) / 2

                # Detect Takeaway to lock the cone
                wrist_y = lm[16].y
                if not is_downswing:
                    if wrist_y < max_w_y: max_w_y = wrist_y
                    elif wrist_y > (max_w_y + 0.05): is_downswing = True

                # --- DYNAMIC VS LOCKED CONE ---
                # Only update the cone position BEFORE the swing starts (Address)
                if not is_downswing and locked_apex is None:
                    shldr_x, shldr_y = lm[12].x * w, lm[12].y * h
                    wrist_x, wrist_y = lm[16].x * w, lm[16].y * h
                    hip_x, hip_y = lm[24].x * w, lm[24].y * h

                    arm_dist = math.sqrt((shldr_x - wrist_x)**2 + (shldr_y - wrist_y)**2)
                    apex_x = int(wrist_x + (arm_dist * 0.33))
                    apex_y = int(wrist_y + (0.03 * h))
                    
                    locked_apex = (apex_x, apex_y)

                    def get_end(p1, p2):
                        v = np.array([p2[0]-p1[0], p2[1]-p1[1]])
                        v = v / np.linalg.norm(v)
                        return tuple((np.array(p1) + v * 2000).astype(int))

                    locked_top_end = get_end(locked_apex, (shldr_x, shldr_y))
                    locked_bottom_end = get_end(locked_apex, (hip_x, hip_y))

                # Draw the static cone (frozen at address position)
                if locked_apex:
                    overlay = frame.copy()
                    pts = np.array([locked_apex, locked_top_end, locked_bottom_end], np.int32)
                    cv2.fillPoly(overlay, [pts], (220, 220, 220))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.line(frame, locked_apex, locked_top_end, (0, 0, 0), 2)
                    cv2.line(frame, locked_apex, locked_bottom_end, (0, 0, 0), 2)

                # Stability Check (Head/Hips)
                head_stable = abs(lm[0].y - addr_head_y) < 0.04
                hip_stable = abs(((lm[23].x + lm[24].x)/2) - addr_hip_x) < 0.05
                
                # Draw Stability Boxes
                h_col = (0, 255, 0) if head_stable else (0, 0, 255)
                cv2.rectangle(frame, (int(lm[0].x*w)-30, int(addr_head_y*h)-30), (int(lm[0].x*w)+30, int(addr_head_y*h)+30), h_col, 2)

            out.write(frame)

    cap.release()
    out.release()

    # Graceful FFmpeg Fallback
    final_video_path = tfile.name  # Changed from final_path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264 = tfile.name.replace(".mp4", "_h264.mp4")
        try:
            subprocess.run([ffmpeg, "-y", "-i", tfile.name, "-c:v", "libx264", "-pix_fmt", "yuv420p", h264], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_video_path = h264  # Changed from final_path
        except: 
            pass

    # --- THESE MUST HAVE ZERO INDENTATION (ALL THE WAY LEFT) ---
    impact_val = f"{lag_impact}°" if lag_impact is not None else "N/A (Try Slo-Mo for better detection)"

    report = (
        "### 🦴 X-Ray Diagnostic\n"
        f"Head: {'PASS' if head_stable else 'FAIL'} | Hip: {'PASS' if hip_stable else 'FAIL'}\n"
        f"Top Hinge: {lag_top or 'N/A'}° | Impact Hinge: {impact_val}"
    )

    return report, final_video_path, lag_top, lag_impact