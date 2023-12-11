import streamlit as st
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from transformers import pipeline
import re

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

model_path = '../../Model/'
device = "cpu"

labels = np.load(model_path+'labels.npy', allow_pickle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10).to(device)
model.load_state_dict(torch.load(model_path+'resume_label.pth'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

summarizer = pipeline("summarization")

def predict_label(text):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits)

    preds = preds.detach().cpu().numpy()

    threshold = 0.2  # You can adjust this based on your preference
    preds_binary = (np.array(preds) > threshold).astype(int)

    job_profiles = [label for value, label in zip(preds_binary[0], labels) if value]

    return job_profiles


@st.cache_data
def classify_job_role(resume_text):
    # Placeholder for job role classification model
    return "Software Engineer", 0.85

@st.cache_data
def read_skillset():
    df = pd.read_csv("../../Data/Cleaned/clean_norm_skillset.csv")
    skills = ','.join(df['Skill'])
    skillset = skills.split(',')
    skillset = set(skillset)
    skill_list = [element.strip() for element in skillset if len(element.split()) < 3]

    return skill_list


@st.cache_data
def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

@st.cache_data
def identify_skillsets(resume_text):
    skillset = read_skillset()
    skills = extract_skills_from_resume(resume_text, skillset)
    return skills


@st.cache_data()
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
    summarized_output = summarizer(text_input,max_length=200)
    return summarized_output
    return text_input.summary_text


