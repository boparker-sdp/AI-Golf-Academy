import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action

st.title("🏌️ AI Golf Diagnostic Hub")

uploaded_file = st.file_uploader("Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    
    st.subheader("Choose a Diagnostic Lab:")
    col1, col2 = st.columns(2)

    with col1:
        # This solves the 'Thin/Fat' issue by tracking Head/Hip stability
        if st.button("⚖️ Run Stability Scan", use_container_width=True):
            with st.spinner("Checking Head Bobbing & Hip Sway..."):
                summary, video_out = analyze_diagnostic_swing(video_path)
                st.session_state.summary = summary
                st.session_state.result_video = video_out

    with col2:
        # This is our Anatomy-Based Plane Lab (No clicking required!)
        if st.button("🚀 Run Plane & Wrist Lab", use_container_width=True):
            with st.spinner("Analyzing Swing Plane & Wrist Lag..."):
                summary, video_out = analyze_wrist_action(video_path)
                st.session_state.summary = summary
                st.session_state.result_video = video_out

    # --- DISPLAY RESULTS ---
    if 'result_video' in st.session_state:
        st.divider()
        st.markdown(st.session_state.summary)
        st.video(st.session_state.result_video)
