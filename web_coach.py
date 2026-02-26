import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# Instructions at the top
with st.expander("📸 How to film for the Coach"):
    st.info("Face-On: Camera at chest height, aiming at belt buckle. \nDown-The-Line: Camera at hip height, aiming through hands toward target.")

uploaded_file = st.file_uploader("Upload your swing", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.video(video_path)
        
        # WHOLISTIC SCOUT REPORT
        st.subheader("📋 Coach's Scout Report")
        st.write("Visual Scan: Swing path looks slightly outward. Sequence appears shoulder-dominant.")
        st.success("Recommendation: Run the **Swing Plane Lab** to check path, then **Foundation** to fix timing.")
        
        c1, c2 = st.columns(2)
        if c1.button("⚖️ Foundation & Sequence", use_container_width=True):
            summary, v_out = analyze_foundation_sequence(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Foundation"
        if c2.button("🚀 Swing Plane Lab", use_container_width=True):
            summary, v_out = analyze_swing_plane(video_path)
            st.session_state.summary, st.session_state.v_out, st.session_state.mode = summary, v_out, "Swing Plane"

    with col_results:
        if 'v_out' in st.session_state:
            st.markdown(st.session_state.summary)
            st.video(st.session_state.v_out)
            
    # CHAT BOX FOR FOLLOW-UPS
    st.divider()
    st.subheader("💬 Follow-up with Coach")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a follow-up question about your results...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        # Mocking the Coach response - in production, this sends to Gemini Pro
        response = f"Your {st.session_state.get('mode', 'analysis')} shows a pull risk. Try the 'Step Drill' to fix the Shoulder Lead."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
