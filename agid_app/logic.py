import time
import random
from ui import show_fullscreen_spinner
import streamlit as st


def detect_ai():
    """Here we will implement the logic for the model."""
    spinner_placeholder = st.empty()
    show_fullscreen_spinner(spinner_placeholder)

    ### ADD MODEL PROCESSING LOGIC HERE ###

    time.sleep(2)

    ### END OF MODEL PROCESSING LOGIC ###
    spinner_placeholder.empty()
    return random.choice([True, False])
