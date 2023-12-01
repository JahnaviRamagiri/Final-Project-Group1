# main.py

import streamlit as st
from sidebar import create_sidebar
import base64
import pydeck as pdk


def filter_data(data, min_value, max_value, include_additional_filter):
    # Example data filtering logic based on selected filters
    filtered_data = list(filter(lambda x: (x > min_value) & (x < max_value), data))

    if include_additional_filter:
        # Apply additional filter logic
        filtered_data = filtered_data*10

    return filtered_data


def display_pdf(uploaded_file):
    if uploaded_file is not None:
        # Display PDF content using pydeck
        pdf_contents = uploaded_file.read()
        st.subheader("PDF Viewer:")
        pdf_base64 = base64.b64encode(pdf_contents).decode("utf-8")
        pdf_display = f'<embed src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" type="application/pdf">'
        st.write(pdf_display, unsafe_allow_html=True)

# Store uploaded files in a list
uploaded_files = []

def main():
    # Load or generate example data
    data = [i for i in range(6)]

    # Create sidebar and get filter values
    min_value, max_value, include_additional_filter, uploaded_file = create_sidebar(data)

    # Filter data based on sidebar values
    filtered_data = filter_data(data, min_value, max_value, include_additional_filter)

    st.title("Filtered Data")
    st.write(filtered_data)

    # Main content layout
    main_column, pdf_viewer_column = st.columns([2, 3])

    # Display filtered data on the left side
    with main_column:
        # File history table
        file_history = st.table()

    # Display uploaded files in the table
    for file in uploaded_files:
        file_history.add_rows([[file]])

    # Check if a new file is uploaded
    if uploaded_file is not None:
        # Add the new file to the list
        uploaded_files.append(uploaded_file)

        # Update the file history table
        for file in uploaded_files:
            file_history.add_rows([[file.name]])

    # Display PDF content on the right side
    with pdf_viewer_column:
        st.title("PDF Viewer")
        display_pdf(uploaded_file)


if __name__ == "__main__":
    main()
