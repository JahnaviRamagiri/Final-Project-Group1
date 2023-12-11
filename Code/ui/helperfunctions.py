import streamlit as st
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from transformers import pipeline
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertForSequenceClassification
import torch
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


model_path = '../../Model/'

device = "cpu"

stop_words = set(stopwords.words("english"))

labels = np.load(model_path+'labels.npy', allow_pickle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10).to(device)
model.load_state_dict(torch.load(model_path+'resume_label.pth', map_location=torch.device(device)))

summarizer = pipeline("summarization")

vectorizer = TfidfVectorizer()

# Load pre-trained BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model and tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

@st.cache_data
def predict_label(text):
    model.eval()
    encoding = bert_tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits)

    preds = preds.detach().cpu().numpy()

    return labels, preds


def compare_job_description1(resume_text, job_description):
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


def compare_job_description(resume_text, job_description):

    resume_text = " ".join(word for word in resume_text.split() if word.lower() not in stop_words)
    job_description = " ".join(word for word in job_description.split() if word.lower() not in stop_words)

    # Tokenize and get embeddings for the first document
    inputs_doc1 = roberta_tokenizer(resume_text, return_tensors='pt', max_length=512, truncation=True)
    outputs_doc1 = roberta_model(**inputs_doc1)
    embeddings_doc1 = outputs_doc1.last_hidden_state.mean(dim=1)  # Using mean pooling for simplicity

    # Tokenize and get embeddings for the second document
    inputs_doc2 = roberta_tokenizer(job_description, return_tensors='pt', max_length=512, truncation=True)
    outputs_doc2 = roberta_model(**inputs_doc2)
    embeddings_doc2 = outputs_doc2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity(
        embeddings_doc1.detach().numpy(),
        embeddings_doc2.detach().numpy(),
    )[0][0]

    return (similarity-0.9)*10


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
    summarized_output = summarizer(text_input,max_length=300)
    print(type(summarized_output))
    return summarized_output
