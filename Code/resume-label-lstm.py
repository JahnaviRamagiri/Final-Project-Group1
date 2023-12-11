import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

model_path = '../Model/'

# Load your custom data
data = pd.read_csv('../Data/Cleaned/clean_resume_lbl.csv')
data = data.dropna()
# data = data[:1000]

print(data.shape)

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
max_len = 128  # You can adjust this based on your dataset and resource constraints // change this


# Preprocess the data
class Transformer(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32),
        }

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

mlb = MultiLabelBinarizer()

# Prepare the data loaders
train_dataset = Transformer(
    texts=train_data['Resume'].values,
    labels=mlb.fit_transform(train_data['Label'].apply(lambda x: x.split(','))),
    tokenizer=tokenizer,
    max_len=max_len,
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = Transformer(
    texts=val_data['Resume'].values,
    labels=mlb.fit_transform(val_data['Label'].apply(lambda x: x.split(','))),
    tokenizer=tokenizer,
    max_len=max_len,
)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Number of classes: {len(mlb.classes_)}")

# Define the model
# model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased',
#     num_labels=len(mlb.classes_),
# )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# model.to(device)

class BertLSTMModel(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_size=256, num_layers=1, bidirectional=True):
        super(BertLSTMModel, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        lstm_output, _ = self.lstm(pooled_output)
        # lstm_output = lstm_output[:, -1, :]
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)
        return logits

# Fine-tune BERT model with CNN head
model = BertLSTMModel(BertModel.from_pretrained('bert-base-uncased'), num_labels = len(mlb.classes_)).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# criterion = torch.nn.BCEWithLogitsLoss()


# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

epochs = 10
train_losses = []
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the validation set
model.eval()
val_losses = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Validation'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())

        preds = torch.sigmoid(outputs)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Calculate evaluation metrics (e.g., precision, recall, F1 score)
from sklearn.metrics import precision_score, recall_score, f1_score

threshold = 0.4  # You can adjust this based on your preference
preds_binary = (np.array(all_preds) > threshold).astype(int)
labels_binary = np.array(all_labels)

precision = precision_score(labels_binary, preds_binary, average='micro')
recall = recall_score(labels_binary, preds_binary, average='micro')
f1 = f1_score(labels_binary, preds_binary, average='micro')

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

torch.save(model.state_dict(), model_path+'resume_label_lstm.pth')
np.save(model_path+'labels_lstm.npy', mlb.classes_)
print(train_losses, val_losses)
