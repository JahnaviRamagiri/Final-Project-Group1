import streamlit as st
import helperfunctions as hf


def main():
    st.title("Resume Scanner Tool")

    # Upload and parse resume
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx"])
    if uploaded_file is not None:
        resume_text, pages = hf.convert_pdf_to_txt_file(uploaded_file)
        resume_summary = hf.summarize_text(resume_text)

        # Job Role Classification
        # job_role, confidence = hf.predict_label(resume_text)
        # st.subheader(f"Predicted Job Role: {job_role} (Confidence: {confidence:.2%})")

        job_role = hf.predict_label(resume_text)
        st.subheader(f"Predicted Job Role: {job_role})")

        # Skillset Identification
        skills = hf.identify_skillsets(resume_text)
        st.subheader("Identified Skills:")
        st.write(", ".join(skills))

        # Job Description Comparison
        job_description = st.text_area("Enter Job Description:")
        if st.button("Compare"):
            similarity_score = hf.compare_job_description(resume_text, job_description)
            st.subheader(f"Similarity Score with Job Description: {similarity_score:.2%}")

        # Visualize relevant information using sliders
        st.sidebar.subheader("Visualization Settings")
        display_summary = st.sidebar.checkbox("Display Resume Summary", value=True)
        display_metrics = st.sidebar.checkbox("Display Metrics", value=True)

        if display_summary:
            st.subheader("Resume Summary:")
            st.text(resume_summary)

        if display_metrics:
            # Metrics visualization with sliders
            confidence_threshold = st.slider("Job Role Confidence Threshold", 0.0, 1.0, 0.8, 0.01)
            st.write(f"Showing job roles with confidence >= {confidence_threshold:.2%}")


if __name__ == "__main__":
    main()
