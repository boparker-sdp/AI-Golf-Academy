import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- STEP 1: FULL DISCOVERY ---
st.subheader("📝 Step 1: Shot Results")
c_flight, c_strike = st.columns(2)

with c_flight:
    ball_flight = st.selectbox(
        "Ball Flight (Direction)",
        ["Unknown", "Straight Pull (Left)", "Slice (Curves Right)", "Straight/Target", "Hook (Curves Left)"]
    )

with c_strike:
    strike_quality = st.selectbox(
        "Strike Quality (Contact)",
        ["Flush/Clean", "Fat (Heavy)", "Thin (Bladed)", "Shank/Toe"]
    )

# --- STEP 2: FILE UPLOAD ---
uploaded_file = st.file_uploader("Step 2: Upload swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'video_path' not in st.session_state or uploaded_file.name != st.session_state.get('last_uploaded'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.last_uploaded = uploaded_file.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        st.video(st.session_state.video_path)
        
        # Informed Scout Report
        if strike_quality == "Fat (Heavy)":
            st.error("Targeting Low Point: Let's check Head Stability in the Foundation Lab.")
        elif ball_flight == "Straight Pull (Left)":
            st.warning("Targeting Path: Let's check Sequence Stretch in the Foundation Lab.")
        
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
            
            # --- THE "SMARTER" DEEP DIVE ---
            st.subheader("📖 The Coach's Plain-English Breakdown")
            
            if st.session_state.mode == "Foundation":
                # Logic: If Flush but Head moved, don't talk about Fat shots.
                if strike_quality == "Flush/Clean" and ball_flight == "Straight Pull (Left)":
                    st.info(f"Great contact! Since the strike was **{strike_quality}**, your head movement isn't hurting your low-point yet. "
                            f"However, we need to talk about that **{ball_flight}**. Your 'Stretch' score is the key here. "
                            "Even with a stable head, if your shoulders fire at the same time as your hips, you 'push' the club left. "
                            "Focus on feeling your hips turn toward the target while your back stays facing it for a split second longer.")
                elif strike_quality == "Fat (Heavy)":
                    st.info("You felt that one in the turf. Because you hit it **Fat**, that red Head Box is the smoking gun. "
                            "You're dipping into the ball, which moves your 'swing circle' into the ground too early.")
                else:
                    st.info(f"Analyzing your **{ball_flight}** and **{strike_quality}** strike. Look at the sequence timing...")
            
            else:
                st.info(f"Looking at your **{ball_flight}**. Check the hand trail against the gray cone safely zone.")

    # --- THE "MEMORY" CHAT ---
    st.divider()
    st.subheader("💬 Continuous Coaching (Chat)")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a follow-up..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # The AI now knows BOTH flight and strike
            response = (f"Acknowledged. Since you hit it **{strike_quality}** but it went **{ball_flight}**, "
                        "we know your low-point control is fine, but your direction is off. "
                        "Let's focus on your hip-shoulder separation.")
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
