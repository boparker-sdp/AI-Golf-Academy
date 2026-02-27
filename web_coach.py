import streamlit as st
import tempfile
import os
# Assuming a standard Gemini integration for the "Synthesis"
# import google.generativeai as genai 

from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- STEP 1: SHOT INPUTS ---
st.subheader("📝 Step 1: Tell the Coach about the result")
c_flight, c_strike = st.columns(2)
with c_flight:
    ball_flight = st.selectbox("Ball Flight", ["Straight", "Straight Pull (Left)", "Slice", "Hook", "Push"])
with c_strike:
    strike_quality = st.selectbox("Strike Quality", ["Flush/Clean", "Fat (Heavy)", "Thin/Topped", "Shank"])

# --- STEP 2: VIDEO UPLOAD ---
uploaded_file = st.file_uploader("Step 2: Upload swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # Handle video path persistence
    if 'video_path' not in st.session_state or uploaded_file.name != st.session_state.get('last_uploaded'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.last_uploaded = uploaded_file.name
        # Clear old analysis when new video arrives
        st.session_state.ai_scout_report = None 

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        
        # --- CALLING THE AI SYNTHESIS ---
        if st.session_state.get('ai_scout_report') is None:
            with st.spinner("Coach is watching the video..."):
                # RETAIL CONCEPT: 
                # report = call_gemini_vision_api(st.session_state.video_path, ball_flight, strike_quality)
                # For this demo, we use the specific synthesis you liked:
                report = f"""
                **First off, great job making such {strike_quality} contact.** Hitting the ball cleanly is half the battle! 
                
                When you hit a ball that {strike_quality} but it flies {ball_flight}, it tells us your club and body are 'racing.' 
                The pull is created because your arms and shoulders are starting the downswing first, pushing the club 'outside.' 
                
                **To see the proof:** Run the Foundation Lab. If the 'Stretch' score is low, your shoulders are winning the race. 
                If the Head Box is stable, we know your height is perfect and the miss is 100% timing.
                """
                st.session_state.ai_scout_report = report
        
        st.info(st.session_state.ai_scout_report)
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
            st.subheader(f"Visual Lab: {st.session_state.mode}")
            st.video(st.session_state.v_out)
            st.markdown(st.session_state.summary)
            st.write("---")
            st.write("**Lab Insights:** Use these visual cues to confirm what the Coach noted in the Scout Report. Look for the red 'Fail' boxes or low Stretch numbers.")

    # --- MEMORY CHAT ---
    st.divider()
    st.subheader("💬 Deep-Dive Chat")
    # (Chat history logic remains the same as previous stable version)
