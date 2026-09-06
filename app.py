"""
app.py -- Streamlit web interface for the Object Finder Agent.
Thin UI only: everything else lives in agent.py and drawer.py.
"""

import os
import time

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from agent import run_agent

load_dotenv()  # reads GROQ_API_KEY from a .env file if present

st.set_page_config(page_title="Object Finder Agent", page_icon="🔎")
st.title("🔎 Object Finder Agent")
st.caption(
    "Upload a picture, ask for ANY object (e.g. *fire hydrant*, *dog*, *umbrella*), and the "
    "agent finds it and draws a red box. LangChain @tools + Groq — 100% free."
)

# Sidebar: API key + model picker
api_key = os.getenv("GROQ_API_KEY", "")
with st.sidebar:
    st.header("⚙️ Settings")
    if api_key:
        st.success("API key loaded from .env")
    else:
        api_key = st.text_input("Groq API key", type="password", help="Get one free at console.groq.com/keys")
    model_name = st.selectbox(
        "Model",
        [
            "qwen/qwen3.6-27b"
        ],
        index=0,
    )

# Main page: upload + query
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
query = st.text_input("What should I find?", placeholder="e.g. fire hydrant")

if uploaded and query:
    image = Image.open(uploaded).convert("RGB")

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar (or put it in a .env file).")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.image(image, use_container_width=True)

    with st.spinner("Agent is thinking and calling its tools..."):
        try:
            start = time.time()
            output_image, detection, trace = run_agent(image, query, model_name, api_key)
            elapsed = time.time() - start
        except Exception as exc:  # bad key, no internet, etc.
            st.error(f"Agent run failed: {exc}")
            st.stop()

    with col2:
        st.subheader("After")
        if output_image is not None:
            st.image(output_image, use_container_width=True)
        else:
            st.info(f"The agent could not find **{query}** in this picture.")

    with st.expander("🧠 Agent trace — tools the agent called", expanded=True):
        for step in trace:
            st.markdown(f"- `{step}`")
        if not trace:
            st.markdown("- (no tools were called)")

    st.subheader("Agent answer (raw structured output)")
    st.json(detection.model_dump() if detection else {"found": False, "label": query})
    st.caption(f"Model: {model_name} · agent finished in {elapsed:.1f}s")
else:
    st.info("👈 Upload an image, type what to look for, and the agent will box it in red.")
