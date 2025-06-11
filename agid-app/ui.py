import streamlit as st
import base64
from PIL import Image
from io import BytesIO


def inject_styles():
    with open("agid-app/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_uploaded_image(uploaded_file):
    if not uploaded_file:
        st.markdown("""
        <div class="placeholder-box">Drag your 500x500 image here</div>
        """, unsafe_allow_html=True)
        return None

    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="JPG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(f"""
    <div class="placeholder-box">
        <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />
    </div>
    """, unsafe_allow_html=True)
    return image


def render_detection_result(result):
    st.markdown("---")
    if result:
        st.error("This image is likely **AI-generated**.")
    else:
        st.success("This image appears to be **authentic**.")


def show_fullscreen_spinner(placeholder):
    spinner_html = """
    <div id="fullscreen-spinner">
        <div class="spinner"></div>
        <p style="margin-top: 20px; font-size: 18px; color: #555;">AGID is thinking...</p>
    </div>
    <style>
    #fullscreen-spinner {
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(255,255,255,0.8);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .spinner {
        border: 12px solid #f3f3f3;
        border-top: 12px solid #4CAF50;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    placeholder.markdown(spinner_html, unsafe_allow_html=True)
