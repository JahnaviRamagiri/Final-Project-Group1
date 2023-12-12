from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Load your custom data
resume_df = pd.read_csv('../Data/Cleaned/clean_resume_lbl.csv')
resume_df = resume_df.dropna()

data = resume_df['Resume']
labels = resume_df['Label'].apply(lambda x: x.split(','))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=6312)

# Convert labels to binary format
mlb = MultiLabelBinarizer()
y_train_binary = mlb.fit_transform(y_train)
y_test_binary = mlb.transform(y_test)

# Convert text data to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create Binary Relevance model with Multinomial Naive Bayes as the base classifier
classifier = BinaryRelevance(MultinomialNB())
classifier.fit(X_train_vectorized, y_train_binary)

# Make predictions on the train set
predictions_tr = classifier.predict(X_train_vectorized)

# Evaluate the model
accuracy_tr = accuracy_score(y_train_binary, predictions_tr)
precision_tr = precision_score(y_train_binary, predictions_tr, average='micro')
recall_tr = recall_score(y_train_binary, predictions_tr, average='micro')
f1_tr = f1_score(y_train_binary, predictions_tr, average='micro')

print("Training Metrics")
print(f'Precision: {precision_tr:.4f}, Recall: {recall_tr:.4f}')
print(f", F1 Score: {f1_tr:.4f}', Accuracy: {accuracy_tr:.4f}")

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test_binary, predictions)
precision = precision_score(y_test_binary, predictions, average='micro')
recall = recall_score(y_test_binary, predictions, average='micro')
f1 = f1_score(y_test_binary, predictions, average='micro')

print("\nTesting Metrics")
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
print(f", F1 Score: {f1:.4f}', Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:\n", classification_report(y_test_binary, predictions, target_names=mlb.classes_))
