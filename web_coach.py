import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 1. INITIAL DISCOVERY (The informed starting point)
st.subheader("📝 Step 1: Tell the Coach what happened")
col_input, col_setup = st.columns([1, 1])

with col_input:
    ball_flight = st.selectbox(
        "What was the ball flight of this specific swing?",
        ["Unknown/Not Sure", "Straight Pull (Left)", "Slice (Curves Right)", "Straight/Target", "Hook (Curves Left)"]
    )

with col_setup:
    with st.expander("📸 Proper Camera Setup"):
        st.write("FO: Chest height, at buckle. | DTL: Hip height, through hands.")

# 2. FILE UPLOAD
uploaded_file = st.file_uploader("Step 2: Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        # --- INFORMED SCOUT REPORT ---
        st.subheader("📋 Coach's Scout Report")
        
        if ball_flight == "Straight Pull (Left)":
            st.warning("**Diagnosis: The Shoulder-Push.** Since you pulled it straight left, your path is 'Out-to-In'. This is usually caused by the upper body firing before the hips. Run the **Foundation Lab** to check your Sequence Stretch.")
        elif ball_flight == "Slice (Curves Right)":
            st.warning("**Diagnosis: The Over-the-Top.** A slice means the club is wiping across the ball. This is usually a plane issue. Run the **Swing Plane Lab** to see if you are 'climbing' over the cone.")
        elif ball_flight == "Straight/Target":
            st.success("**Diagnosis: Pure Strike.** You found the slot! Run the labs to see your 'Pro Blueprint' so you can repeat this feeling.")
        else:
            st.info("**Diagnosis: Analyzing Geometry.** Without ball flight data, I will look for 'Out-to-In' movement. Run both labs to find your power leaks.")

        st.video(video_path)
        
        c1, c2 = st.columns(2)
        if c1.button("⚖️ Foundation & Sequence", use_container_width=True):
            summary, v_out = analyze_foundation_sequence(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Foundation"
        if c2.button("🚀 Swing Plane Lab", use_container_width=True):
            summary, v_out = analyze_swing_plane(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Swing Plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader(f"AI Results: {st.session_state.mode}")
            st.markdown(st.session_state.summary)
            st.video(st.session_state.v_out)
            
            # Contextual help based on ball flight
            if ball_flight == "Straight Pull (Left)" and st.session_state.mode == "Foundation":
                st.error("🚨 Focus: Your 'Stretch' score must be above 30 to stop the pull!")

    # 3. INTERACTIVE CHAT
    st.divider()
    st.subheader("💬 Follow-up with Coach")
    query = st.chat_input("Ask for drills to fix your specific ball flight...")
    if query:
        st.write(f"**You:** {query}")
        # Logic here to feed ball_flight + summary into the response
        st.write(f"**AI Coach:** Since you're dealing with a **{ball_flight}**, your primary focus in the {st.session_state.mode} should be...")
