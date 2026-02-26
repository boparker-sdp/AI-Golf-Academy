import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action

st.set_page_config(page_title="AI Golf Academy", layout="wide")

st.title("🏌️ AI Golf Diagnostic Hub")

# --- CAMERA SETUP GUIDE ---
with st.expander("📸 How to get the best AI Analysis"):
    st.markdown("""
    **1. Wrist & Plane Lab (Down-the-Line):**
    * Camera at **hip height**.
    * Aim through your **hands** toward the target.
    
    **2. Stability & Sequence Scan (Face-On):**
    * Camera at **chest height**.
    * Aim at your **belt buckle**.
    
    *Note: Perspective errors can cause the 'Anatomy Lines' to drift.*
    """)

uploaded_file = st.file_uploader("Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.video(video_path)
        
        st.subheader("Run Diagnostic Labs")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚖️ Stability & Sequence", use_container_width=True):
                with st.spinner("Analyzing Hip/Shoulder Sequence..."):
                    summary, video_out = analyze_diagnostic_swing(video_path)
                    st.session_state.summary = summary
                    st.session_state.result_video = video_out
        with c2:
            if st.button("🚀 Plane & X-Factor", use_container_width=True):
                with st.spinner("Analyzing Wrist Lag & X-Factor..."):
                    summary, video_out = analyze_wrist_action(video_path)
                    st.session_state.summary = summary
                    st.session_state.result_video = video_out

    with col_results:
        if 'result_video' in st.session_state:
            st.markdown(st.session_state.summary)
            st.video(st.session_state.result_video)
