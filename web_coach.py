import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- STEP 1: SHOT RESULTS ---
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
            
            st.subheader("📖 The Coach's Plain-English Breakdown")
            if st.session_state.mode == "Foundation":
                if strike_quality == "Flush/Clean" and ball_flight == "Straight Pull (Left)":
                    st.info(f"The strike was **{strike_quality}**, so your low-point is solid. However, that **{ball_flight}** confirms your sequence is 'bunched up.'")
                else:
                    st.info(f"Analyzing your **{ball_flight}** and **{strike_quality}** strike...")

    # --- THE IMPROVED CHAT ENGINE ---
    st.divider()
    st.subheader("💬 AI Coaching Chat")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history in an expandable container to prevent "creaking"
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for specific drills..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # --- DRILL GENERATION LOGIC ---
        with st.chat_message("assistant"):
            if "drill" in prompt.lower() or "stretch" in prompt.lower() or "separate" in prompt.lower():
                response = f"""
                Acknowledged. To fix that **{ball_flight}** and increase your **Stretch Score**, we need to keep your shoulders from 'racing' the hips. 
                
                ### 🛠️ The "Step-Through" Drill
                1. **Setup:** Take your normal stance with a mid-iron.
                2. **The Backswing:** As you reach the top, keep your back facing the target.
                3. **The Trigger:** Before you swing down, take a small step toward the target with your lead foot.
                4. **The Feel:** This forces your hips to move first, creating that 'Stretch' you see in the lab.
                
                ### 🛠️ The "Pump" Drill
                * Stop at the top of your swing.
                * Move your hips toward the target 3 times while keeping your shoulders completely still.
                * On the 4th pump, let the swing go.
                
                Does one of these feel like something you can try at the range?
                """
            else:
                response = "I'm analyzing that. Can you tell me if you feel more 'stuck' in your hips or 'rushed' in your shoulders?"
            
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
