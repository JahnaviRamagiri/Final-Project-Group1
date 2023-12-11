import streamlit as st
import helperfunctions as hf
import numpy as np


def main():
    st.title("Resume Scanner Tool")

    # Upload and parse resume
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx"])
    if uploaded_file is not None:

        resume_text, pages = hf.convert_pdf_to_txt_file(uploaded_file)

        # Metrics visualization with sliders
        st.sidebar.subheader("Job Role Confidence Threshold")
        confidence_threshold = st.sidebar.slider(label="", min_value=0, max_value=100, value=80, step=5)
        st.sidebar.write(f"Showing job roles with confidence => {confidence_threshold}%")

        # Job Role Classification
        labels, preds = hf.predict_label(resume_text)

        threshold = confidence_threshold/100 * 0.5  # You can adjust this based on your preference
        preds_binary = (np.array(preds) > threshold).astype(int)

        job_profiles = [label for value, label in zip(preds_binary[0], labels) if value]
        st.subheader(f"Predicted Job Role: {', '.join(job_profiles)}")

        if st.button("Summarize Resume"):
            st.subheader("Resume Summary:")
            resume_summary = hf.summarize_text(resume_text)
            st.markdown(resume_summary[0]['summary_text'], unsafe_allow_html=True)


        # Skillset Identification
        skills = hf.identify_skillsets(resume_text)
        st.subheader("Identified Skills:")
        st.markdown(", ".join(skills))

        # Job Description Comparison
        st.subheader("Resume - Job Description Comparison")
        job_description = st.text_area("Enter Job Description:")
        if st.button("Compare"):
            similarity_score = hf.compare_job_description(resume_text, job_description)
            st.subheader(f"Similarity Score with Job Description: {similarity_score:.2%}")


if __name__ == "__main__":
    main()
