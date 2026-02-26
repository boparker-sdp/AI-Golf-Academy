import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action

# 1. Page Config & Professional Styling
st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 2. Friend/Operator Setup Guide
with st.expander("📸 INSTRUCTIONS FOR YOUR CAMERA OPERATOR"):
    st.markdown("""
    **To ensure the AI math is accurate, your friend should follow these rules:**

    **1. For the Wrist & Plane Lab (Standing behind the golfer):**
    * Stand directly behind the golfer looking down the target line.
    * Hold phone at **hip height** and aim at the golfer's **hands**.
    
    **2. For the Stability & Sequence Scan (Facing the golfer):**
    * Stand directly facing the golfer (belly-to-belly). 
    * Hold phone at **chest height** and aim at the **belt buckle**.

    **⚠️ CRITICAL:** Hold the phone perfectly still. Do **NOT** move the camera to 'follow' the clubhead.
    """)

# 3. Persistent File Uploader
uploaded_file = st.file_uploader("Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("Raw Swing Video")
        st.video(video_path)
        
        st.subheader("Select Diagnostic Lab")
        c1, c2 = st.columns(2)
        
        if c1.button("⚖️ Stability & Sequence", use_container_width=True):
            with st.spinner("Checking your kinematic sequence..."):
                summary, v_out = analyze_diagnostic_swing(video_path)
                st.session_state.summary = summary
                st.session_state.v_out = v_out
                st.session_state.mode = "stability"

        if c2.button("🚀 Plane & X-Factor", use_container_width=True):
            with st.spinner("Calculating plane and stretch..."):
                summary, v_out = analyze_wrist_action(video_path)
                st.session_state.summary = summary
                st.session_state.v_out = v_out
                st.session_state.mode = "plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader("AI Analysis Results")
            st.markdown(st.session_state.summary)
            st.video(st.session_state.v_out)
            
            # --- THE NEW COLOR GUIDE & EXPLAINER ---
            st.divider()
            st.markdown("#### 🟢 COLOR GUIDE")
            st.caption("🟢 **GREEN**: Within Pro tolerances. | 🔴 **RED**: Fault detected.")
            
            with st.expander("🎓 COACH'S INTEL: The Pull-Left Diagnosis"):
                if st.session_state.get("mode") == "stability":
                    st.info("**The Sequence Fault:** Even if you start with a 'Hip Lead', you may be 'Spinning Out' your shoulders early. This causes the club to cut across the ball, resulting in a pull.")
                else:
                    st.info("**The Power Link:** Your X-Factor is your potential energy. To stop the pull, keep your 'Hinge' (Lag) deep while letting your hips clear a path for your hands.")
                
                st.success("**Drill for Today:** Try the 'Step Drill'. Take your stance, start your backswing, and as you reach the top, take a small step toward the target with your lead foot *before* you swing your arms.")
