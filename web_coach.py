import streamlit as st
import os

# Your custom modules
from ai_coach import vibe_coach
from swing_analyzer import analyze_diagnostic_swing
from wrist_tracker import drill_coach

st.set_page_config(page_title="AI Golf Academy", layout="centered")

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
        f.write(uploaded_file.read())
        
    st.video(video_path)
    
    # --- BALL STRIKING CONTEXT ---
    st.markdown("### Tell the Coach About the Shot")
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
                # BUNDLE THE CONTEXT: Combine the 3 dropdowns into one string
                result_context = f"Shape: {shape}, Contact: {contact}, Direction: {direction}"
                
                # Pass exactly 3 arguments: the video, the bundled context, and the model
                coach_report = vibe_coach(video_path, result_context, selected_model_id)
                
                st.success("Analysis Complete!")
                st.markdown(coach_report)
                
                # Download Button for the Text Report
                st.download_button(
                    label="📄 Save Coach Report",
                    data=coach_report,
                    file_name="AI_Golf_Coach_Report.txt",
                    mime="text/plain",
                    key="save_report_btn",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")

   # --- 2. X-RAY DIAGNOSTIC ---
    if st.button("🦴 Run X-Ray Diagnostic", use_container_width=True):
        with st.spinner("Processing X-Ray Vision..."):
            try:
                xray_video_path = analyze_diagnostic_swing(video_path)
                
                # Read the file directly into memory
                with open(xray_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # Force the browser to play it natively
                st.video(video_bytes, format="video/webm")
                
                st.download_button(
                    label="💾 Save X-Ray Video",
                    data=video_bytes,
                    file_name="XRay_Swing.webm",
                    mime="video/webm",
                    key="save_xray_btn",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing X-Ray: {e}")

    # --- 3. WRIST LAB ---
    if st.button("⌚ Run Wrist Lab", use_container_width=True):
        with st.spinner("Analyzing Wrist Hinge..."):
            try:
                wrist_video_path = drill_coach(video_path)
                
                # Read the file directly into memory
                with open(wrist_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # Force the browser to play it natively
                st.video(video_bytes, format="video/webm")
                
                st.download_button(
                    label="💾 Save Wrist Lab Video",
                    data=video_bytes,
                    file_name="Wrist_Lab.webm",
                    mime="video/webm",
                    key="save_wrist_btn",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing Wrist Lab: {e}")

    # --- CLEAR SCREEN FOR NEXT SWING ---
    st.divider()
    if st.button("🔄 Clear Screen for Next Swing", type="primary", use_container_width=True):
        st.rerun()


