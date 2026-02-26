import streamlit as st
import tempfile
import os
from swing_analyzer import analyze_diagnostic_swing, analyze_wrist_action

st.set_page_config(page_title="AI Golf Academy", layout="wide")
st.title("🏌️ AI Golf Diagnostic Hub")

# --- CAMERA OPERATOR GUIDE ---
with st.expander("📸 INSTRUCTIONS FOR YOUR CAMERA OPERATOR"):
    st.markdown("""
    **If you are filming the golfer, please follow these rules for accurate AI math:**

    **1. For the Wrist & Plane Lab (Standing behind the golfer):**
    * **Where to stand:** Stand directly behind the golfer so you are looking down the line they want to hit the ball.
    * **Height:** Hold the phone at the golfer's **hip height**.
    * **Target:** Line up the camera so the golfer's **hands** are in the center of the screen.
    
    **2. For the Stability & Sequence Scan (Standing in front of the golfer):**
    * **Where to stand:** Stand directly facing the golfer (belly-to-belly). 
    * **Height:** Hold the phone at the golfer's **chest height**.
    * **Target:** Aim the camera at the golfer's **belt buckle**.

    **⚠️ CRITICAL:** Do **NOT** move the camera or try to 'follow' the club. Hold it perfectly still like a tripod, or the AI will think the golfer is swaying when they aren't!
    """)

# ... (rest of the file remains the same)
