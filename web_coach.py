import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_foundation_sequence, analyze_swing_plane

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# 1. Setup Instructions
with st.expander("📸 INSTRUCTIONS FOR YOUR CAMERA OPERATOR"):
    st.markdown("""
    **To ensure the AI math is accurate, please follow these rules:**
    * **Down-the-Line (Swing Plane):** Stand behind the golfer, phone at **hip height**, aim through **hands**.
    * **Face-On (Foundation):** Stand facing the golfer, phone at **chest height**, aim at **belt buckle**.
    * **STILLNESS:** Do NOT move the camera to 'follow' the club.
    """)

# 2. Upload
uploaded_file = st.file_uploader("Upload your swing video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_vid, col_results = st.columns([1, 1])

    with col_vid:
        st.subheader("📋 Coach's Scout Report")
        st.info("""
        **Visual Assessment:** I detect an **Out-to-In** path. 
        * If you are **Slicing** (curving right), run the **Swing Plane Lab**. 
        * If you are **Pulling** (straight left), run the **Foundation Lab** to check your Sequence Stretch.
        """)
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
            
            # --- THE BINARY DIAGNOSTIC ---
            st.divider()
            with st.expander("🔬 UNDERSTANDING YOUR 'OUT-TO-IN' PATH"):
                t1, t2 = st.tabs(["The Slice (Vertical)", "The Pull (Horizontal)"])
                with t1:
                    st.error("**The Over-the-Top (Slice)**")
                    st.write("Hands 'climb' over the plane. Look at the yellow trail in the **Swing Plane Lab**.")
                with t2:
                    st.error("**The Shoulder-Push (Pull)**")
                    st.write("Low 'Stretch.' Shoulders fire with hips. Check your Stretch score in the **Foundation Lab**.")

            st.success("💡 **Next Step:** Use the **Chat Function** below to ask the Coach for a set of drills to fix your specific error.")

    # --- THE INTERACTIVE COACH ---
    st.divider()
    st.subheader("💬 Chat with your AI Coach")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask for drills or explain your results..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # This is where you would call the Gemini API with the lab results as context
            context = f"User is in {st.session_state.mode} mode. Results: {st.session_state.summary}"
            response = "Based on your low Sequence Stretch and 'Shoulder-Push' pull, I recommend the **Step-Through Drill**: Take your stance, and as you reach the top of your backswing, step your lead foot forward before starting your downswing. This forces your hips to lead. Would you like a video description of that drill?"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
