import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Import your custom modules
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action
from ai_coach import vibe_coach, coach_chat
from wrist_tracker import drill_coach

st.set_page_config(page_title="AI Golf Academy", layout="centered")

# --- INITIALIZE APP MEMORY ---
if "coach_report" not in st.session_state:
    st.session_state.coach_report = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- SIDEBAR: Coach Settings ---
st.sidebar.title("⚙️ Coach Settings")

MODELS = {
    "⚡ Gemini 3 Flash (Fastest)": "gemini-3-flash-preview",
    "🧠 Gemini 3.1 Pro (Elite)": "gemini-3.1-pro-preview",
    "🐎 Gemini 2.5 Flash (Workhorse)": "gemini-2.5-flash",
    "💎 Gemini 2.5 Pro (Technical)": "gemini-2.5-pro"
}

selected_model_display = st.sidebar.selectbox(
    "AI Brain Version", 
    options=list(MODELS.keys()),
    index=0
)
selected_model_id = MODELS[selected_model_display]

st.sidebar.divider()
st.sidebar.info(f"Active Model: {selected_model_display}")

# --- MAIN UI ---
st.title("🏌️‍♂️ AI Golf Academy")
st.warning("📱 **iPhone Users:** Run this app directly in Safari or Chrome, rather than saving it to your Home Screen.")

uploaded_file = st.file_uploader("Upload your swing...", type=["mp4", "mov", "avi", "m4v", "webm"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    # --- CALIBRATION UI ---
    st.divider()
    st.subheader("🎯 Step 1: Set Your Swing Plane")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info("Scrub to the address position (club behind ball), then click the ball.")
    frame_idx = st.slider("Find Address Frame:", 0, total_frames - 1, value=0)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        max_ui_height = 600
        scale_ratio = max_ui_height / img.height
        new_width = int(img.width * scale_ratio)
        img_resized = img.resize((new_width, max_ui_height))
        
        st.write("Now, **click directly on the golf ball**:")
        
        coords = streamlit_image_coordinates(
            img_resized,
            key="ball_picker"
        )
        
        if coords:
            # --- HIGH-RES SCALING MATH ---
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_w = min(raw_w, raw_h)
            video_h = max(raw_w, raw_h)
            
            display_h = 600 
            display_w = new_width 
            
            ratio_h = video_h / display_h
            ratio_w = video_w / display_w
            
            real_x = int(round(coords['x'] * ratio_w))
            real_y = int(round(coords['y'] * ratio_h))
            ball_pos = (real_x, real_y)
            
            st.success(f"🎯 Ball calibrated to high-res: {ball_pos}")

            # --- THE LAUNCH BUTTON ---
            if st.button("🚀 Run Wrist Lab Analysis", use_container_width=True, key="calibrated_run"):
                with st.spinner("Analyzing your path and lag..."):
                    summary, video_out = analyze_wrist_action(
                        video_path, 
                        ball_coords=ball_pos, 
                        start_frame=frame_idx
                    )
                    st.divider()
                    st.header("📊 Your Wrist Lab Report")
                    st.markdown(summary)
                    st.video(video_out)
       
                    with open(video_out, "rb") as v_file:
                        st.download_button(
                            "💾 Save Wrist Lab Video",
                            data=v_file.read(),
                            file_name="Wrist_Lab_Analysis.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

    cap.release()

    # --- OTHER AI COACH FEATURES ---
    st.divider()
    st.markdown("### Tell the Coach About the Shot")
    
    club_type = st.radio("Club Used:", ["Iron / Wedge", "Wood / Driver"], horizontal=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        shape = st.selectbox("Shape", ["Straight", "Draw", "Fade", "Pull", "Push", "Hook", "Slice", "Unknown"])
    with col2:
        contact = st.selectbox("Contact", ["Flush", "Thin", "Fat", "Toe", "Heel", "Topped", "Unknown"])
    with col3:
        direction = st.selectbox("Direction", ["On Target", "Left", "Right", "Short", "Long", "Unknown"])

    if st.button("💬 Ask AI Vibe Coach", use_container_width=True):
        with st.spinner(f"Consulting {selected_model_display}..."):
            try:
                math_feedback, _ = analyze_diagnostic_swing(video_path)
                result_context = (
                    f"Club: {club_type}, Shape: {shape}, Contact: {contact}, "
                    f"Direction: {direction}. OpenCV Analysis: {math_feedback}"
                )
                coach_report = vibe_coach(video_path, result_context, selected_model_id)
                st.session_state.coach_report = coach_report
                st.session_state.chat_messages = []
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.coach_report:
        st.success("Analysis Complete!")
        st.markdown(st.session_state.coach_report)
        
        st.download_button(
            label="📄 Save Coach Report",
            data=st.session_state.coach_report,
            file_name="Coach_Report.txt",
            mime="application/octet-stream",
            use_container_width=True
        )
        
        st.divider()
        st.markdown("### 🗣️ Chat with your Coach")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_q := st.chat_input("Ask about your swing:"):
            st.session_state.chat_messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                answer = coach_chat(user_q, st.session_state.coach_report, selected_model_id)
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if st.button("🔄 Clear Screen for Next Swing", type="primary", use_container_width=True):
        st.session_state.coach_report = None
        st.session_state.chat_messages = []
        st.rerun()
