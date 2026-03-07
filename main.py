import streamlit as st
import os

from ai_coach import vibe_coach, coach_chat
from legacy.swing_analyzer_dev import analyze_foundation_sequence
from wrist_tracker import drill_coach

st.set_page_config(page_title="AI Golf Academy", layout="centered")

# --- INITIALIZE APP MEMORY ---
if "coach_report" not in st.session_state:
    st.session_state.coach_report = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "analysis_video" not in st.session_state:
    st.session_state.analysis_video = None  # Unified video storage

# --- SIDEBAR: Coach Settings ---
st.sidebar.title("⚙️ Coach Settings")

MODELS = {
    "⚡ Gemini 3 Flash (Fastest)": "gemini-3-flash-preview",
    "🧠 Gemini 3.1 Pro (Elite)": "gemini-3.1-pro-preview",
    "🐎 Gemini 2.5 Flash (Workhorse)": "gemini-2.5-flash",
    "💎 Gemini 2.5 Pro (Technical)": "gemini-2.5-pro",
}

selected_model_display = st.sidebar.selectbox(
    "AI Brain Version",
    options=list(MODELS.keys()),
    index=0,
)
selected_model_id = MODELS[selected_model_display]

st.sidebar.divider()
st.sidebar.info(f"Active Model: {selected_model_display}")

# --- MAIN UI ---
st.title("🏌️‍♂️ AI Golf Academy")
st.warning(
    "📱 **iPhone Users:** To ensure video uploads and report downloads work perfectly, "
    "please run this app directly in your Safari or Chrome browser, rather than saving it "
    "to your Home Screen."
)

uploaded_file = st.file_uploader(
    "Upload your swing...", type=["mp4", "mov", "avi", "m4v", "webm"]
)

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

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
                st.session_state.coach_report = coach_report
                st.session_state.analysis_started = True
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")

    # --- 2. X-RAY DIAGNOSTIC ---
    if st.button("🦴 Run X-Ray Diagnostic", use_container_width=True):
        with st.spinner("Processing X-Ray Vision..."):
            try:
                report, v_path, top_h, imp_h = analyze_foundation_sequence(video_path)
                st.session_state.analysis_video = v_path
                st.session_state.coach_report = report
                st.session_state.analysis_started = True
            except Exception as e:
                st.error(f"Error processing X-Ray: {e}")

    # --- 3. WRIST LAB ---
    if st.button("⌚ Run Wrist Lab", use_container_width=True):
        with st.spinner("Analyzing Wrist Hinge..."):
            try:
                # Reusing the dev analyzer which has the best hinge/cone logic
                report, v_path, top_h, imp_h = analyze_foundation_sequence(video_path)
                st.session_state.analysis_video = v_path
                st.session_state.coach_report = report.replace("X-Ray Diagnostic", "Wrist Lab Analysis")
                st.session_state.analysis_started = True
            except Exception as e:
                st.error(f"Error processing Wrist Lab: {e}")

# --- 4. UNIVERSAL DISPLAY & SAVE ---
if st.session_state.analysis_video:
    st.divider()
    st.video(st.session_state.analysis_video)
    
    with open(st.session_state.analysis_video, "rb") as f:
        st.download_button(
            label="💾 Save Analysis Video",
            data=f,
            file_name="Golf_Academy_Analysis.mp4",
            mime="video/mp4",
            use_container_width=True
        )

# --- 5. CHAT UI ---
if st.session_state.analysis_started:
    st.divider()
    st.success("Analysis Complete!")

    if st.session_state.coach_report:
        st.markdown(st.session_state.coach_report)
        st.download_button(
            label="📄 Save Coach Report",
            data=st.session_state.coach_report,
            file_name="AI_Golf_Coach_Report.txt",
            mime="application/octet-stream",
            key="save_report_btn",
            use_container_width=True,
        )

    st.markdown("### 🗣️ Chat with your Coach")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("Ask about your swing:"):
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        
        with st.chat_message("assistant"):
            with st.spinner("Coach is thinking..."):
                answer = coach_chat(user_q, st.session_state.coach_report, selected_model_id)
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# --- 6. CLEAR SCREEN ---
st.divider()
if st.button("🔄 Clear Screen for Next Swing", type="primary", use_container_width=True):
    st.session_state.coach_report = None
    st.session_state.chat_messages = []
    st.session_state.analysis_video = None
    st.session_state.analysis_started = False
    st.rerun()