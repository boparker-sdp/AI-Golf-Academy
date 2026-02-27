import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# STEP 1: SHOT RESULTS
st.subheader("📝 Step 1: Shot Results")
c_flight, c_strike = st.columns(2)
with c_flight:
    ball_flight = st.selectbox("Ball Flight", ["Straight/Target", "Straight Pull (Left)", "Slice (Fade Right)", "Hook", "Push"])
with c_strike:
    strike_quality = st.selectbox("Strike Quality", ["Flush/Clean", "Fat (Heavy)", "Thin (Bladed)", "Shank"])

# STEP 2: UPLOAD
uploaded_file = st.file_uploader("Step 2: Upload Video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'video_path' not in st.session_state or uploaded_file.name != st.session_state.get('last_uploaded'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.session_state.last_uploaded = uploaded_file.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        
        # Immediate High-Level Feedback
        if strike_quality == "Flush/Clean" and ball_flight == "Straight Pull (Left)":
            st.info("Excellent contact. Since you didn't hit it fat, your head is stable. The pull is a timing issue: your shoulders are outrunning your hips. Check the Foundation Lab.")
        elif strike_quality == "Fat (Heavy)":
            st.error("You're hitting the ground early. This is usually caused by a head dip. Check the Stability boxes in the Foundation Lab.")
        else:
            st.info(f"Analyzing your {ball_flight} shot with a {strike_quality} strike. Run a lab to see the biomechanics.")

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
            
            if st.session_state.mode == "Foundation":
                st.info("Coach's Deep Dive: Look at the Head Box. If it turns RED during the downswing, your head is dipping into the ball, causing that fat strike.")
            
            col_save1, col_save2 = st.columns(2)
            with open(st.session_state.v_out, "rb") as f:
                col_save1.download_button("💾 Save Video", f, "swing_analysis.mp4")
            st.button("💾 Save Notes", on_click=lambda: st.toast("Notes saved!"))

    # CHAT SECTION
    st.divider()
    st.subheader("💬 Continuous Coaching (Chat)")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for drills..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if "drill" in prompt.lower() or "stretch" in prompt.lower():
                response = f"To fix that {ball_flight}, try the 'Back-to-Target' drill: keep your back facing the target a split second longer during the start of the downswing. This lets your hips lead the way!"
            else:
                response = f"I'm analyzing your {strike_quality} strike. Does it feel like you're losing your posture at impact?"
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
