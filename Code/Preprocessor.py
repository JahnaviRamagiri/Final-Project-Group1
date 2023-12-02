import pandas as pd
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import codecs
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, data_path, input_file_path, cleaned_file_path):
        """
        Input Files: resume_samples.txt
        Creating Files:
            1) resume_samples.csv
        """
        self.data_path = data_path
        self.input_file_path = self.data_path + input_file_path
        self.cleaned_file_path = self.data_path + cleaned_file_path
        self.input_data = codecs.open(self.input_file_path, "rU", encoding='utf-8', errors='ignore')

        self.resume_df = self.read_dataset()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.cleaned_df = pd.DataFrame()

    def print_title(self, title, pattern="*", pattern_length=20, num_blank_lines=1):
        """

        :param title: String to be printed
        :param pattern: Pattern preceeding and Succeding the String in the title
        :param pattern_length: Length of pattern
        :param num_blank_lines: Total blank lines before and after the Title
        """
        print((num_blank_lines // 2 + 1) * "\n", pattern_length * pattern, title, pattern_length * pattern,
              num_blank_lines // 2 * "\n")

    def read_dataset(self):
        """

        """
        self.print_title("Reading Dataset")
        self.input_data.seek(0)
        resume_lst = self.input_data.read().splitlines()

        print(f"Resume list size: {len(resume_lst)}\n")

        skill_set = []
        resumes = []
        for resume in resume_lst:
            x = resume.split(':::')
            if len(x) == 3:
                skill_set.append(x[1])
                resumes.append(x[2])

        resume_df = pd.DataFrame({
            'Skillset': skill_set,
            'Resume': resumes
        })

        print(f"Resume data dimensions: {resume_df.shape}\n")
        print(resume_df.head())

        return resume_df

    def basic_statistics(self):
        """
        Calculate basic statistics for the dataset.
        """
        self.print_title("Statistics", '`', 20)
        print("Total number of resumes:", len(self.cleaned_df))
        print(f"Average length of resumes: {round(self.cleaned_df['Resume'].apply(len).mean())} words")
        print("Most common skills:")
        # TODO: Get most common skills
        # Process skillset
        # print(self.resume_df['Skillset'].value_counts().head())

    def remove_repeating_neighbors(self, tokens, n=3):

        dedup_tokens = []
        for i, word in enumerate(tokens):
            if word not in tokens[i+1: i+n+1]:
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

    def clean_skillset(self, skillset):
        """
        Clean and preprocess individual skillset.
        """
        pattern = re.compile(r'\([^\)]*\)')
        matched_skillset = re.sub(pattern, '', skillset)
        # Tokenization
        skills = matched_skillset.split(';')
        # Lowercasing
        skills = [skill.lower() for skill in skills]
        # Remove duplicates
        skills = list(set(skills))
        # Sorting
        skills.sort()
        # Join skills into a cleaned skillset
        cleaned_skillset = skills

        return cleaned_skillset

    def preprocess_skillset_df(self):
        """
        Preprocess "Skillset".
        """
        self.print_title("Preprocessing Skillset", '-', 20)
        self.cleaned_df['Skillset'] = self.resume_df['Skillset'].apply(self.clean_skillset)
        # TODO: work on experience mentioned in skillset brackets

    def preprocess_resume_df(self):
        """
        Preprocess "Resume".
        """
        self.print_title("Preprocessing Resumes", '-', 20)
        self.cleaned_df['Resume'] = self.resume_df['Resume'].apply(self.clean_resume_text)
        # Add additional preprocessing steps or features if needed
        self.cleaned_df.to_csv(self.cleaned_file_path, index=False)

    def read_cleaned_dataset(self):
        self.cleaned_df = pd.read_csv(self.cleaned_file_path)

    def generate_wordcloud(self, resume_index):
        """
        Generate a word cloud for a specific resume record.
        :param resume_index: Input the Resume Index number
        """
        text = self.cleaned_df['Resume'][resume_index]

        # Generate WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the WordCloud using Matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Resume #{resume_index}')
        plt.show()
