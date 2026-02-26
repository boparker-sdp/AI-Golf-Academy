import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action

# 1. Page Config & Title
st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 2. Camera Operator Guide (Always visible at top)
with st.expander("📸 INSTRUCTIONS FOR YOUR CAMERA OPERATOR"):
    st.markdown("""
    **If you are filming the golfer, please follow these rules for accurate AI math:**

    **1. For the Wrist & Plane Lab (Standing behind the golfer):**
    * **Where to stand:** Directly behind the golfer, looking down the target line.
    * **Height:** Hold the phone at the golfer's **hip height**.
    * **Target:** Center the golfer's **hands** in the screen.
    
    **2. For the Stability & Sequence Scan (Standing in front of the golfer):**
    * **Where to stand:** Facing the golfer (belly-to-belly). 
    * **Height:** Hold the phone at the golfer's **chest height**.
    * **Target:** Aim the camera at the golfer's **belt buckle**.

    **⚠️ CRITICAL:** Hold the phone perfectly still. Do **NOT** move the camera to 'follow' the club.
    """)

# 3. THE FILE UPLOADER (The anchor that was missing)
uploaded_file = st.file_uploader("Upload your swing video", type=['mp4', 'mov', 'avi'])

# 4. Analysis Logic
if uploaded_file:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Create the two-column layout
    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("Your Swing")
        st.video(video_path)
        
        st.subheader("Run Diagnostic Labs")
        c1, c2 = st.columns(2)
        
        if c1.button("⚖️ Stability Scan", use_container_width=True):
            with st.spinner("Analyzing Hip/Shoulder Sequence..."):
                summary, v_out = analyze_diagnostic_swing(video_path)
                st.session_state.summary = summary
                st.session_state.v_out = v_out
                st.session_state.mode = "stability"

        if c2.button("🚀 Plane Lab", use_container_width=True):
            with st.spinner("Analyzing Wrist Lag & X-Factor..."):
                summary, v_out = analyze_wrist_action(video_path)
                st.session_state.summary = summary
                st.session_state.v_out = v_out
                st.session_state.mode = "plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader("AI Analysis Results")
            st.markdown(st.session_state.summary)
            st.video(st.session_state.v_out)
            
            # THE COACH'S INTEL SECTION
            with st.expander("🎓 COACH'S INTEL: Putting it all together"):
                if st.session_state.get("mode") == "stability":
                    st.info("**Sequence Truth:** If you see 'SHOULDER LEAD', your upper body is rotating before your hips clear. This is the #1 cause of pull-shots.")
                else:
                    st.info("**X-Factor vs. Lag:** High X-Factor is power, but it requires high Lag (Hinge) to hold that power. If the degree number jumps to 180 too early, you're 'casting'.")
                
                st.success("**The Summary:** Your hands stayed in the cone (Good Plane), but if you are pulling it left, check the 'Shoulder Lead' in the Stability Scan. Focus on 'Belt Buckle First'!")
