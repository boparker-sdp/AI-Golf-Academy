import streamlit as st
from google import genai
import time
import os

def vibe_coach(video_path, result_context, model_id="gemini-3-flash-preview"):
    """
    Analyzes a golf swing based on video and optional ball flight data.
    """
    # Initialize using the secret key from your vault
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 1. Upload Video
    video_file = client.files.upload(file=video_path)
    
    # 2. Poll for completion
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        return "Error: AI could not process this video."

    # 3. Flexible Prompting
    prompt = f"""
    You are a world-class PGA swing coach. 
    RESULT CONTEXT: {result_context}

    YOUR MISSION:
    1. If context is 'unknown', analyze pure technical form.
    2. If context is provided, work backwards to the flaw in the video.

    REPORT FORMAT:
    - THE BREAKDOWN: Primary flaw and its connection to the ball flight.
    - THE 'FEEL' FIX: One drill or mental cue.
    - THE TOOLBOX: Recommend 'X-Ray Diagnostic' (stability/plane) or 'Wrist Lab' (hinge/release).
    """

    # 4. Generate Content
    # SYNCED: Uses the model_id passed from web_coach.py
    response = client.models.generate_content(
        model=model_id,
        contents=[prompt, video_file]
    )
    
    return response.text