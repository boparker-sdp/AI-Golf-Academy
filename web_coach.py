import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 1. INITIAL DISCOVERY
st.subheader("📝 Step 1: Tell the Coach what happened")
col_input, col_setup = st.columns([1, 1])

with col_input:
    ball_flight = st.selectbox(
        "What was the ball flight of this specific swing?",
        ["Unknown/Not Sure", "Straight Pull (Left)", "Slice (Curves Right)", "Straight/Target", "Fat (Hit Ground Early)"]
    )

with col_setup:
    with st.expander("📸 Proper Camera Setup"):
        st.write("FO (Stability): Chest height, at buckle. | DTL (Plane): Hip height, through hands.")

# 2. FILE UPLOAD
uploaded_file = st.file_uploader("Step 2: Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # Session State Management for Video Path
    if 'video_path' not in st.session_state or uploaded_file.name != st.session_state.get('last_uploaded'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.last_uploaded = uploaded_file.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        
        # Informed Scout Report based on Ball Flight
        if ball_flight == "Fat (Hit Ground Early)":
            st.error("**The Dip Detector.** Fat shots are usually caused by your head dropping or your hips swaying. Run the **Foundation Lab** to check your Stability Boxes.")
        elif ball_flight == "Straight Pull (Left)":
            st.warning("**The Shoulder-Push.** Your path is likely 'Out-to-In'. Run the **Foundation Lab** to check your Sequence Stretch.")
        elif ball_flight == "Slice (Curves Right)":
            st.warning("**The Over-the-Top.** This is a plane issue. Run the **Swing Plane Lab** to see if you are 'climbing' over the cone.")
        else:
            st.info("Analyzing geometry. Run both labs to find your power leaks.")

        st.video(st.session_state.video_path)
        
        c1, c2 = st.columns(2)
        if c1.button("⚖️ Foundation & Sequence", use_container_width=True):
            summary, v_out = analyze_foundation_sequence(st.session_state.video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Foundation"
        if c2.button("🚀 Swing Plane Lab", use_container_width=True):
            summary, v_out = analyze_swing_plane(st.session_state.video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Swing Plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader(f"AI Results: {st.session_state.mode}")
            st.video(st.session_state.v_out)
            st.markdown(st.session_state.summary)
            
            # --- CONTEXTUAL STABILITY ANALYSIS ---
            if st.session_state.mode == "Foundation":
                st.info("""
                **Coach's Deep Dive (Stability):**
                * **Head Box:**
