import pandas as pd

file_path = "resume_samples.txt"

try:
    df = pd.read_csv(file_path, delimiter="\t", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, delimiter="\t", encoding='latin1')

print(df.head())