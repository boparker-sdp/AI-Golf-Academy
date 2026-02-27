import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- STEP 1: DISCOVERY ---
st.subheader("📝 Step 1: Shot Results")
c_flight, c_strike = st.columns(2)
with c_flight:
    ball_flight = st.selectbox("Ball Flight", ["Straight", "Straight Pull (Left)", "Slice (Curves Right)", "Hook", "Push"])
with c_strike:
    strike_quality = st.selectbox("Strike Quality", ["Flush/Clean", "Fat (Heavy)", "Thin (Bladed)", "Shank"])

# --- STEP 2: UPLOAD ---
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
        if ball_flight == "Straight Pull (Left)" and strike_quality == "Flush/Clean":
            st.info("Observation: Excellent contact, but your timing is 'bunched up.' Your shoulders are racing ahead of your hips. Check the Foundation Lab.")
        elif strike_quality == "Fat (Heavy)":
            st.error("Observation: You're hitting the ground early. We need to check your Head Stability for downward dips.")
        
        st.video(st.session_state.video_path)
        c1, c2 = st.columns(2)
        if c1.button("⚖️ Foundation"):
            summary, v_out = analyze_foundation_sequence(st.session_state.video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Foundation"
        if c2.button("🚀 Swing Plane"):
            summary, v_out = analyze_swing_plane(st.session_state.video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Swing Plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader(f"AI Results: {st.session_state.mode}")
            st.video(st.session_state.v_out)
            st.markdown(st.session_state.summary)
            
            # Contextual Deep Dive
            if st.session_state.mode == "Foundation":
                st.info("Coach's Note: If that Head Box turned RED, that's why you're hitting it fat. Stay tall and rotate like a barber pole.")
            
            col_save, col_notes = st.columns(2)
            with open(st.session_state.v_out, "rb") as f:
                col_save.download_button("💾 Save Video", f, "swing.mp4")

    # --- CHAT ENGINE ---
    st.divider()
    st.subheader("💬 AI Coaching Chat")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for drills..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if "drill" in prompt.lower():
                response = f"To fix that {ball_flight}, try the 'Step-Through' drill: step toward the target with your lead foot before you swing down. This forces your hips to lead!"
            else:
                response = f"I'm looking at your {st.session_state.mode} data. Does it feel like your shoulders are rushing?"
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
