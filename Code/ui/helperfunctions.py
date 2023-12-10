import streamlit as st
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from transformers import pipeline


@st.cache_data
def classify_job_role(resume_text):
    # Placeholder for job role classification model
    return "Software Engineer", 0.85


@st.cache_data
def identify_skillsets(resume_text):
    # Placeholder for skillset identification model
    return ["Python", "Machine Learning", "Data Analysis"]


@st.cache_data
def compare_job_description(resume_text, job_description):
    # Placeholder for job description comparison model
    return 0.75


@st.cache_data
def convert_pdf_to_txt_file(path):

    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    interpreter = PDFPageInterpreter(rsrcmgr, device)

    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))

    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()

    device.close()
    retstr.close()
    return t, nbPages


@st.cache_data
def summarize_text(text_input):
    summarizer = pipeline("summarization")
    summarized_output = summarizer(text_input,max_length= 100)
    return summarized_output

@st.cache_data
def summarize_resume(file):

    # Extract text from PDF
    resume_txt, pages = convert_pdf_to_txt_file(file)

    # Summarize resume
    summarised_text = summarize_text(resume_txt)

    return summarised_text

