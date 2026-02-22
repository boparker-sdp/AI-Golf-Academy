import streamlit as st
import os

# Your custom modules
from ai_coach import vibe_coach, coach_chat
from swing_analyzer import analyze_diagnostic_swing
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

uploaded_file = st.file_uploader("Upload your swing...", type=["mp4", "mov", "avi", "m4v", "webm"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(video_path)
    
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
                result_context = f"Club: {club_type}, Shape: {shape}, Contact: {contact}, Direction: {direction}"
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
        
        st.download_button(
            label="📄 Save Coach Report",
            data=st.session_state.coach_report,
            file_name="AI_Golf_Coach_Report.txt",
            mime="text/plain",
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
                xray_video_path = analyze_diagnostic_swing(video_path, club_type)
                with open(xray_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, format="video/mp4")
                st.download_button("💾 Save X-Ray Video", data=video_bytes, file_name="XRay_Swing.mp4", mime="video/mp4", key="save_xray", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing X-Ray: {e}")

    # --- 3. WRIST LAB ---
    if st.button("⌚ Run Wrist Lab", use_container_width=True):
        with st.spinner("Analyzing Wrist Hinge..."):
            try:
                wrist_video_path = drill_coach(video_path, club_type)
                with open(wrist_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, format="video/mp4")
                st.download_button("💾 Save Wrist Lab", data=video_bytes, file_name="Wrist_Lab.mp4", mime="video/mp4", key="save_wrist", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing Wrist Lab: {e}")

    # --- CLEAR SCREEN ---
    st.divider()
    if st.button("🔄 Clear Screen for Next Swing", type="primary", use_container_width=True):
        # Wipes the memory so you can start fresh!
        st.session_state.coach_report = None
        st.session_state.chat_messages = []
        st.rerun()
