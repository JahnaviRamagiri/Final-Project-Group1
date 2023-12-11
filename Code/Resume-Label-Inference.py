import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

model_path = '../Model/'

device = "cpu"

labels = np.load(model_path+'labels.npy', allow_pickle=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10).to(device)
model.load_state_dict(torch.load(model_path+'resume_label.pth'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def predict_label(text, model, tokenizer, device="cpu", max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits)

    preds = preds.detach().cpu().numpy()

    threshold = 0.4  # You can adjust this based on your preference
    preds_binary = (np.array(preds) > threshold).astype(int)

    job_profiles = [label for value, label in zip(preds_binary[0], labels) if value]

    return job_profiles


data = pd.read_csv('../Data/Cleaned/clean_resume_lbl.csv')
data = data.dropna()
resume = data.loc[345].Resume

preds = predict_label(resume, model, tokenizer, device)

print(preds)

