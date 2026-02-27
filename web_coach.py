import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- INITIAL DISCOVERY ---
st.subheader("📝 Step 1: Tell the Coach what happened")
ball_flight = st.selectbox(
    "What was the ball flight of this specific swing?",
    ["Unknown/Not Sure", "Straight Pull (Left)", "Slice (Curves Right)", "Straight/Target", "Fat (Hit Ground Early)"]
)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Step 2: Upload your swing video", type=['mp4', 'mov', 'avi'])

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
            st.markdown(st.session_state.summary)
            
            # --- THE "NO-PHD" DEEP DIVE ---
            st.subheader("📖 The Coach's Plain-English Breakdown")
            if st.session_state.mode == "Foundation":
                st.info(f"Since your ball was a **{ball_flight}**, let's look at your body's timing. "
                        "A 'Fat' shot happens because your body is literally getting closer to the ground during the swing. "
                        "If that Head Box turned red, it means you're 'dipping' your chest downward—this forces the club to hit the grass "
                        "before it ever reaches the ball. To fix this, you need to feel like your head is pinned against a wall behind you "
                        "so you stay tall and rotate like a spinning top, rather than collapsing downward.")
            else:
                st.info(f"You mentioned a **{ball_flight}**. In the video, look at that gray cone—that's your 'Safe Zone.' "
                        "If your hands stay above that top black line, the club is coming from 'outside' your body. "
                        "Think of it like trying to pull a door shut from the wrong side; it forces the club to swipe across the ball. "
                        "If you keep your hands inside that gray area, the club can swing toward the target instead of across it.")

            col_s1, col_s2 = st.columns(2)
            with open(st.session_state.v_out, "rb") as file:
                col_s1.download_button("💾 Save Marked Video", data=file, file_name="golf_analysis.mp4", mime="video/mp4")
            st.button("💾 Save Coach's Notes", on_click=lambda: st.toast("Notes saved!"))

    # --- THE "MEMORY" CHAT LAB ---
    st.divider()
    st.subheader("💬 Continuous Coaching (Chat)")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a follow-up or ask for a specific drill..."):
        # 1. Save User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response (Mimicking a conversation)
        with st.chat_message("assistant"):
            # Here, we'd normally pass st.session_state.chat_history to Gemini
            response = (f"That's a great follow-up. Since we just talked about your {st.session_state.mode} "
                        f"and your {ball_flight}, let's dig deeper into that. If you're feeling 'stuck,' "
                        "it's because your hips aren't giving your arms enough room. Does that feel like what's happening?")
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun() # Forces the UI to update with the new message
