import streamlit as st
import os
from ai_coach import vibe_coach
from swing_analyzer import analyze_diagnostic_swing
from wrist_tracker import drill_coach

st.set_page_config(page_title="AI Golf Academy", layout="centered")

# --- SIDEBAR: Coach Settings ---
st.sidebar.title("⚙️ Coach Settings")

# Model Roster for 2026
MODELS = {
    "⚡ Gemini 3 Flash (Fastest)": "gemini-3-flash-preview",
    "🧠 Gemini 3.1 Pro (Elite)": "gemini-3.1-pro-preview",
    "🐎 Gemini 2.5 Flash (Workhorse)": "gemini-2.5-flash",
    "💎 Gemini 2.5 Pro (Technical)": "gemini-2.5-pro"
}

selected_model_display = st.sidebar.selectbox(
    "AI Brain Version", 
    options=list(MODELS.keys()),
    index=0  # Defaults to Gemini 3 Flash
)
selected_model_id = MODELS[selected_model_display]

st.sidebar.divider()
st.sidebar.info(f"Active Model: {selected_model_display}")

# --- MAIN UI ---
st.title("🏌️‍♂️ AI Golf Academy")

uploaded_file = st.file_uploader("Upload your swing...", type=["mp4", "mov", "avi", "m4v", "webm"])

if uploaded_file is not None:
    # Save video to a temporary local file
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    # Ball Result Context
    st.divider()
    st.subheader("📊 What was the result?")
    col1, col2, col3 = st.columns(3)
    with col1:
        curve = st.selectbox("Shape", ["Unknown", "Straight", "Draw/Hook", "Fade/Slice", "Push", "Pull"])
    with col2:
        contact = st.selectbox("Contact", ["Unknown", "Good", "Fat", "Thin/Topped", "Shank"])
    with col3:
        direction = st.selectbox("Direction", ["Unknown", "Center", "Left", "Right"])

    # Clean the context for the AI
    if curve == "Unknown" and contact == "Unknown" and direction == "Unknown":
        result_context = "Ball flight outcome is unknown."
    else:
        result_context = f"Shape: {curve}, Contact: {contact}, Direction: {direction}"

    st.divider()

    # Analysis Buttons
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("🧠 AI Vibe Coach"):
            with st.spinner(f"Coaching with {selected_model_display}..."):
                # SYNCED: Passes model_id from the sidebar
                report = vibe_coach(video_path, result_context, selected_model_id)
                st.markdown("### 📝 AI Coaching Report")
                st.write(report)

    with col_b:
        if st.button("🦴 X-Ray Diagnostic"):
            with st.spinner("Analyzing Stability..."):
                out = analyze_diagnostic_swing(video_path)
                if out: st.video(out)

    with col_c:
        if st.button("⌚ Wrist Lab"):
            with st.spinner("Analyzing Hinge..."):
                out = drill_coach(video_path)
                if out: st.video(out)