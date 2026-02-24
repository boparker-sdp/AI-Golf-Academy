import streamlit as st
import os

from ai_coach import vibe_coach, coach_chat
# Import BOTH functions from swing_analyzer
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action
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
st.warning("📱 **iPhone Users:** To ensure video uploads and report downloads work perfectly, please run this app directly in your Safari or Chrome browser, rather than saving it to your Home Screen.")

uploaded_file = st.file_uploader("Upload your swing...", type=["mp4", "mov", "avi", "m4v", "webm"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(video_path, format="video/mp4")    
    
    # --- BALL STRIKING CONTEXT (V2) ---
    st.markdown("### Tell the Coach About the Shot")
    
    club_type = st.radio("Club Used:", ["Iron / Wedge", "Wood / Driver"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        shape = st.selectbox("Shape", ["Straight", "Draw", "Fade", "Pull", "Push", "Hook", "Slice", "Unknown"])
    with col2:
        contact = st.selectbox("Contact", ["Flush", "Thin", "Fat", "Toe", "Heel", "Topped", "Unknown"])
    with col3:
        direction = st.selectbox("Direction", ["On Target", "Left", "Right", "Short", "Long", "Unknown"])

    st.divider()

    # --- 1. AI VIBE COACH ---
    if st.button("💬 Ask AI Vibe Coach", use_container_width=True):
        with st.spinner(f"Consulting {selected_model_display}..."):
            try:
                # A. Run the technical OpenCV Math first
                # (video_path is the variable where your video is saved)
                # NEW CODE
                # The "_" is a placeholder that ignores the video path because the AI only needs the text
                math_feedback, _ = analyze_diagnostic_swing(video_path)
                
                # B. Combine your inputs with the math result
                result_context = (
                    f"Club: {club_type}, Shape: {shape}, Contact: {contact}, "
                    f"Direction: {direction}. OpenCV Analysis: {math_feedback}"
                )
                
                # C. Send everything to the AI Vibe Coach
                coach_report = vibe_coach(video_path, result_context, selected_model_id)
                
                # Save to memory to trigger the chat box!
                st.session_state.coach_report = coach_report
                st.session_state.chat_messages = []
                
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
    
    # --- CHAT UI (Only shows if a report exists) ---
    if st.session_state.coach_report:
        st.success("Analysis Complete!")
        st.markdown(st.session_state.coach_report)
        
        # Download Button for the Text Report
        st.download_button(
            label="📄 Save Coach Report",
            data=st.session_state.coach_report,  # <-- Just add st.session_state. here!
            file_name="AI_Golf_Coach_Report.txt",
            mime="application/octet-stream",  # <-- This is the magic line that fixes the freeze
            key="save_report_btn",
            use_container_width=True
        )
        
        st.divider()
        st.markdown("### 🗣️ Chat with your Coach")
        
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_q := st.chat_input("Ask about your swing (e.g., 'What does laid off mean?'):"):
            st.session_state.chat_messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
                
            with st.chat_message("assistant"):
                with st.spinner("Coach is thinking..."):
                    answer = coach_chat(user_q, st.session_state.coach_report, selected_model_id)
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    st.divider()

# --- 2. X-RAY DIAGNOSTIC ---
    if st.button("🦴 Run X-Ray Diagnostic", use_container_width=True):
        with st.spinner("Processing X-Ray Vision..."):
            try:
                # We expect a tuple back: (report_text, file_path)
                report, xray_video_path = analyze_diagnostic_swing(video_path, club_type)
                
                # Read the newly converted MP4 into memory
                with open(xray_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # Display with the explicit MP4 format for Safari/Firefox
                st.video(video_bytes, format="video/mp4")
                
                st.info(report)
                
                # The download button now points to the same MP4
                st.download_button(
                    label="💾 Save X-Ray Video", 
                    data=video_bytes, 
                    file_name="XRay_Swing_Analysis.mp4", 
                    mime="video/mp4", 
                    key="save_xray", 
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing X-Ray: {e}")

    # --- 3. WRIST LAB ---
    if st.button("⌚ Run Wrist Lab", use_container_width=True):
        with st.spinner("Analyzing Wrist Release..."):
            try:
                # Same tuple pattern for the wrist lab
                wrist_report, wrist_video_path = analyze_wrist_action(video_path)
                
                with open(wrist_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.video(video_bytes, format="video/mp4")
                
                st.info(wrist_report)
                
                st.download_button(
                    label="💾 Save Wrist Video", 
                    data=video_bytes, 
                    file_name="Wrist_Lab_Analysis.mp4", 
                    mime="video/mp4", 
                    key="save_wrist",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error in Wrist Lab: {e}")

    # --- CLEAR SCREEN ---
    st.divider()
    if st.button("🔄 Clear Screen for Next Swing", type="primary", use_container_width=True):
        # Wipes the memory so you can start fresh!
        st.session_state.coach_report = None
        st.session_state.chat_messages = []
        st.rerun()











