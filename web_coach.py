import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 1. INITIAL DISCOVERY
st.subheader("📝 Step 1: Tell the Coach what happened")
ball_flight = st.selectbox(
    "What was the ball flight of this specific swing?",
    ["Unknown/Not Sure", "Straight Pull (Left)", "Slice (Curves Right)", "Straight/Target", "Hook (Curves Left)"]
)

# 2. FILE UPLOAD
uploaded_file = st.file_uploader("Step 2: Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # Save video_path to session_state so it doesn't disappear on rerun
    if 'video_path' not in st.session_state or uploaded_file.name != st.session_state.get('last_uploaded'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.last_uploaded = uploaded_file.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        if ball_flight == "Straight Pull (Left)":
            st.warning("**The Shoulder-Push.** Your path is likely 'Out-to-In' because the upper body fired before the hips. Use **Foundation Lab**.")
        elif ball_flight == "Slice (Curves Right)":
            st.warning("**The Over-the-Top.** This is a plane issue. Use **Swing Plane Lab** to see if you are 'climbing' over the cone.")
        else:
            st.info("Analyzing geometry. Run both labs to find your power leaks.")

        # USE THE SESSION STATE PATH HERE
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
            
            # COACH'S THOUGHTS
            st.info("""
            **Coach's Deep Dive:** Your movement patterns suggest a conflict between your intended target and your body's sequence. 
            When the hips stall, the shoulders are forced to 'throw' the club outward. Focus on initiating the move with a target-side hip bump.
            """)
            
            col_s1, col_s2 = st.columns(2)
            with open(st.session_state.v_out, "rb") as file:
                col_s1.download_button("💾 Save Marked Video", data=file, file_name="swing_analysis.mp4", mime="video/mp4")
            if col_s2.button("💾 Save Coach's Comments"):
                st.toast("Coach's comments saved to your profile!")

    # 3. INTERACTIVE CHAT
    st.divider()
    st.subheader("💬 Follow-up with Coach")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for drills..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Logic for automated drill suggestion
        response = f"Given your {st.session_state.get('mode')} results, I recommend the **Step-Through Drill** to fix that {ball_flight}."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
