import pandas as pd
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
import nltk
from utils import print_title
from random import randint

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, input_file_path, cleaned_file_path):
        """
        Input Files: resume_samples.txt
        Creating Files:
            1) resume_samples.csv
        """
        self.uncleaned_df = pd.read_csv(input_file_path)
        self.cleaned_file_path = cleaned_file_path
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.cleaned_df = pd.DataFrame()

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

        if skillset != skillset:
            return ''

        skillset = skillset.lower()
        pattern = re.compile(r'\([^\)]*\)')
        matched_skillset = re.sub(pattern, '', skillset)

        # Tokenization
        skill_list = [item.strip() for item in matched_skillset.replace('/', ';').split(';')]
        # Remove duplicates
        skills = list(set(skill_list))
        # Sorting
        skills.sort()
        # Join skills into a cleaned skillset
        cleaned_skillset = skills

        return cleaned_skillset

    def clean_skills_norm(self, skillset):
        """
        Clean and preprocess individual skillset.
        """
        skill_ = []
        if skillset != skillset:
            return ''

        skillset = skillset.lower()
        pattern = re.compile(r'\([^\)]*\)')
        matched_skillset = re.sub(pattern, '', skillset)
        # skill_list = [skill.strip() for skill in skills.split('/')]
        # Tokenization
        skill_list = [item.strip() for item in matched_skillset.replace('/', ',').split(',')]
        # Remove duplicates
        skills = list(set(skill_list))
        # Sorting
        skills.sort()
        # Join skills into a cleaned skillset
        cleaned_skillset = ', '.join(skills)

        return cleaned_skillset

    def preprocess_skillset_df(self):
        """
        Preprocess "Skillset".
        """
        print_title("Preprocessing Skillset", '-', 20)
        self.cleaned_df['Skill'] = self.uncleaned_df['Skill'].apply(self.clean_skillset)
        # TODO: work on experience mentioned in skillset brackets

    def preprocess_resume_df(self):
        """
        Preprocess "Resume".
        """
        print_title("Preprocessing Resumes", '-', 20)
        self.cleaned_df['Resume'] = self.uncleaned_df['Resume'].apply(self.clean_resume_text)
        # Add additional preprocessing steps or features if needed
    def preprocess_normalised_skills(self):
        print_title("Preprocessing Normalised Skills", '-', 20)
        self.cleaned_df['Skill'] = self.uncleaned_df['Skill'].apply(self.clean_skills_norm)



    def save_df(self):
        self.cleaned_df.to_csv(self.cleaned_file_path, index=False)

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


def generate_wordcloud(data, resume_index):
    """
    Generate a word cloud for a specific resume record.
    :param resume_index: Input the Resume Index number
    """
    text = data['Resume'][resume_index]

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the WordCloud using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Resume #{resume_index}')
    plt.show()


if __name__ == "__main__":

    if (not os.path.isfile("../Data/Cleaned/clean_resume_skill.csv")):
        resume_skill = Preprocessor("../Data/Uncleaned/uncleaned_resume_skill.csv", "../Data/Cleaned/clean_resume_skill.csv")
        resume_skill.preprocess_skillset_df()
        resume_skill.preprocess_resume_df()
        resume_skill.save_df()
        resume_skill.generate_wordcloud(randint(0, len(resume_skill.cleaned_df)))

    if (not os.path.isfile("../Data/Cleaned/clean_resume_lbl.csv")):
        resume_lbl = Preprocessor("../Data/Uncleaned/uncleaned_resume_lbl.csv", "../Data/Cleaned/clean_resume_lbl.csv")
        resume_lbl.preprocess_resume_df()
        resume_lbl.cleaned_df['Label'] = resume_lbl.uncleaned_df['Label']
        resume_lbl.save_df()
        resume_lbl.generate_wordcloud(randint(0, len(resume_lbl.cleaned_df)))

    if (not os.path.isfile("../Data/Cleaned/clean_norm_skillset.csv")):
        norm_skill = Preprocessor("../Data/Uncleaned/uncleaned_skill_class.csv", "../Data/Cleaned/clean_norm_skillset.csv")
        norm_skill.preprocess_normalised_skills()
        norm_skill.cleaned_df['Label'] = norm_skill.uncleaned_df['Class']
        norm_skill.save_df()
        # norm_skill.generate_wordcloud(randint(0, len(norm_skill.cleaned_df)))
