# sidebar.py

import streamlit as st


def create_sidebar(data):
    st.sidebar.title("Filter Options")

    # Example filter widgets
    min_slider = st.sidebar.slider("Minimum Value", min(data), max(data), min(data))
    max_slider = st.sidebar.slider("Maximum Value", min(data), max(data), max(data))

    # Checkbox for additional filter
    include_checkbox = st.sidebar.checkbox("Include Additional Filter")

    # File upload button
    uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])

    return min_slider, max_slider, include_checkbox, uploaded_file
