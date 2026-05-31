# app.py — BrailleBridge
# Main Streamlit application
# Full implementation happens during hackathon (May 31 4PM onwards)

import streamlit as st
from braille_map import dots_to_char

# Page configuration
st.set_page_config(
    page_title="BrailleBridge",
    page_icon="👁",
    layout="wide"
)

# Session state initialization
if "hero_shown" not in st.session_state:
    st.session_state.hero_shown = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

# Placeholder — full UI built during hackathon
st.title("BrailleBridge")
st.caption("Point. Read. Understand.")
st.info("App under construction — launching May 31, 4PM IST")