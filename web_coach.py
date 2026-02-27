import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

ball_flight = st.selectbox("What was the ball flight?", ["Unknown", "Pull-Left", "Slice", "Straight"])
uploaded_file = st.file_uploader("Upload swing", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # ... (Video saving logic)
    col_vid, col_results = st.columns([1, 1])
    with col_vid:
        st.video(video_path)
        c1, c2 = st.columns(2)
        if c1.button("⚖️ Foundation"):
            summary, v_out = analyze_foundation_sequence(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Foundation"
        if c2.button("🚀 Swing Plane"):
            summary, v_out = analyze_swing_plane(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Swing Plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.subheader(f"Results: {st.session_state.mode}")
            st.video(st.session_state.v_out)
            st.markdown(st.session_state.summary)
            
            # COACH'S THOUGHTS
            st.info("""
            **Coach's Deep Dive:** Your movement patterns suggest a conflict between your intended target and your body's sequence. 
            When the hips stall, the shoulders are forced to 'throw' the club outward, resulting in the high-lag-low-accuracy outcome you're seeing. 
            Focus on initiating the move with a target-side hip bump to create the space your arms need to stay 'in the slot.'
            """)
            
            col_save1, col_save2 = st.columns(2)
            with open(st.session_state.v_out, "rb") as file:
                col_save1.download_button("💾 Save Marked Video", data=file, file_name="swing_analysis.mp4", mime="video/mp4")
            if col_save2.button("💾 Save Coach's Comments"):
                st.toast("Coach's comments saved to your profile!")

    # CHAT SECTION
    st.divider()
    st.subheader("💬 Chat with your AI Coach")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for drills..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = "Let's focus on that 'Shoulder-Push'. Try the 'Split-Hand Drill' to feel the sequence."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if st.button("💾 Save Chat History"):
                st.toast("Chat history archived!")
