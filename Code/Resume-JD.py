from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
doc1 = "I am Data Scientist"
doc2 = "I am looking for Software Engineer"

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([doc1, doc2])
similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

print(f"Cosine Similarity using TFIDF: {similarity*100:.2f}%")

# Load pre-trained BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model and tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

for model, tokenizer, name in [(bert_model, bert_tokenizer, 'BERT'), (roberta_model, roberta_tokenizer, 'ROBERTA')]:

    # Tokenize and get embeddings for the first document
    inputs_doc1 = tokenizer(doc1, return_tensors='pt', max_length=512, truncation=True)
    outputs_doc1 = model(**inputs_doc1)
    embeddings_doc1 = outputs_doc1.last_hidden_state.mean(dim=1)  # Using mean pooling for simplicity

    # Tokenize and get embeddings for the second document
    inputs_doc2 = tokenizer(doc2, return_tensors='pt', max_length=512, truncation=True)
    outputs_doc2 = model(**inputs_doc2)
    embeddings_doc2 = outputs_doc2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity(
        embeddings_doc1.detach().numpy(),
        embeddings_doc2.detach().numpy(),
    )[0][0]

    print(f"Cosine Similarity using {name}: {similarity*100:.2f}%")
