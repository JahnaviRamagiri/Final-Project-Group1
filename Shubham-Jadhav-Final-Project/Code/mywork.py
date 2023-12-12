def remove_repeating_neighbors(self, tokens, n=3):
    dedup_tokens = []
    for i, word in enumerate(tokens):
        if word not in tokens[i + 1: i + n + 1]:
            dedup_tokens.append(word)
    return dedup_tokens


def clean_resume_text(self, text):
    """
    Clean and preprocess individual resume text.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # lowercasiing
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove duplicate neighbors
    dedup_tokens = self.remove_repeating_neighbors(tokens)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_wo_sw = [token for token in dedup_tokens if token not in stop_words]
    # Concatenate tokens into cleaned text
    cleaned_text = ' '.join(tokens_wo_sw)

    return cleaned_text


def preprocess_resume_df(self):
    """
    Preprocess "Resume".
    """
    print_title("Preprocessing Resumes", '-', 20)
    self.cleaned_df['Resume'] = self.uncleaned_df['Resume'].apply(self.clean_resume_text)
    # Add additional preprocessing steps or features if needed

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
    return summarized_output


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

resume_text, pages = hf.convert_pdf_to_txt_file(uploaded_file)

# Job Role Classification
labels, preds = hf.predict_label(resume_text)

threshold = confidence_threshold / 100 * 0.5  # You can adjust this based on your preference
preds_binary = (np.array(preds) > threshold).astype(int)

resume_summary = hf.summarize_text(resume_text)
st.markdown(resume_summary[0]['summary_text'], unsafe_allow_html=True)

similarity_score = hf.compare_job_description(resume_text, job_description)