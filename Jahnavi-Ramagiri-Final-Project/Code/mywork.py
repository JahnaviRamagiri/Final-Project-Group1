def read_skillset():
    df = pd.read_csv("../../Data/Cleaned/clean_norm_skillset.csv")
    skills = ','.join(df['Skill'])
    skillset = skills.split(',')
    skillset = set(skillset)
    skill_list = [element.strip() for element in skillset if len(element.split()) < 3]
    skill_list = [x for x in skill_list if x]
    return skill_list


skillset = read_skillset()


@st.cache_data
def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return set(skills)


@st.cache_data
def identify_skillsets(resume_text):
    skills = extract_skills_from_resume(resume_text, skillset)
    return skills

def main():
    st.title("Resume Scanner Tool")

    # Upload and parse resume
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx"])
    if uploaded_file is not None:

        # Metrics visualization with sliders
        st.sidebar.subheader("Job Role Confidence Threshold")
        confidence_threshold = st.sidebar.slider(label="", min_value=0, max_value=100, value=80, step=5)
        st.sidebar.write(f"Showing job roles with confidence => {confidence_threshold}%")
        st.subheader(f"Predicted Job Role: {', '.join(job_profiles)}")

        if st.button("Summarize Resume"):
            st.subheader("Resume Summary:")


        # Skillset Identification
        skills = hf.identify_skillsets(resume_text)
        st.subheader("Identified Skills:")
        st.markdown(", ".join(skills))

        # Job Description Comparison
        st.subheader("Resume - Job Description Comparison")
        job_description = st.text_area("Enter Job Description:")
        if st.button("Compare"):
            st.subheader(f"Similarity Score with Job Description: {similarity_score:.2%}")


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

    if (not os.path.isfile("../Data/Cleaned/clean_resume_skill1.csv")):
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

    if (not os.path.isfile("../Data/Cleaned/clean_norm_skillset1.csv")):
        norm_skill = Preprocessor("../Data/Uncleaned/uncleaned_skill_class.csv", "../Data/Cleaned/clean_norm_skillset.csv")
        norm_skill.preprocess_normalised_skills()
        norm_skill.cleaned_df['Label'] = norm_skill.uncleaned_df['Class']
        norm_skill.save_df()
        # norm_skill.generate_wordcloud(randint(0, len(norm_skill.cleaned_df)))
