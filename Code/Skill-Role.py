import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification  # Updated import
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your custom data
data = pd.read_csv('../Data/Cleaned/clean_norm_skillset.csv')
data = data.dropna()
# data = data[:1000]

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
max_len = 128  # Adjust this based on your dataset and resource constraints

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Use the same MultiLabelBinarizer instance for both train and validation
mlb = MultiLabelBinarizer()

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


# Prepare the data loaders
train_dataset = Transformer(
    texts=train_data['Skill'].values,
    labels=mlb.fit_transform(train_data['Label'].apply(lambda x: x.split(','))),
    tokenizer=tokenizer,
    max_len=max_len,
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Use the same MultiLabelBinarizer instance for validation
val_dataset = Transformer(
    texts=val_data['Skill'].values,
    labels=mlb.transform(val_data['Label'].apply(lambda x: x.split(','))),  # Note: use transform instead of fit_transform
    tokenizer=tokenizer,
    max_len=max_len,
)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define the model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(mlb.classes_),  # Number of unique labels
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 1  # Adjust this based on your dataset and resource constraints
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
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

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        val_losses.append(loss.item())

        preds = torch.sigmoid(outputs.logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics (e.g., precision, recall, F1 score)
threshold = 0.5  # Adjust this based on your preference
preds_binary = (np.array(all_preds) > threshold).astype(int)
labels_binary = np.array(all_labels)

precision = precision_score(labels_binary, preds_binary, average='micro')
recall = recall_score(labels_binary, preds_binary, average='micro')
f1 = f1_score(labels_binary, preds_binary, average='micro')

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
