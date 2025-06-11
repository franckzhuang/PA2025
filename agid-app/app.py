import streamlit as st
from logic import detect_ai
from ui import display_uploaded_image, render_detection_result, inject_styles

st.set_page_config(
    page_title="AGID - Anti Generated Image Detection",
    layout="wide",
    page_icon="🔎"
)

inject_styles()

_, col1, _, col2, _ = st.columns([1, 6, 1, 6, 1])

with col1:
    with st.container():
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        image = display_uploaded_image(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.title("🔎 AGID")
    st.subheader("Anti Generated Image Detection")
    st.markdown("""
    Upload an image and press the detection button below to check if it's likely **AI-generated** or **Real photos**.
    """)

    if 'result' not in st.session_state:
        st.session_state.result = None

    detect = st.button("🚀 Run Detection", disabled=uploaded_file is None)
    if detect and uploaded_file:
        st.session_state.result = detect_ai()

    if st.session_state.result is not None:
        render_detection_result(st.session_state.result)
