import pandas as pd
import codecs
import os
from utils import print_title


class DataGathering:
    def __init__(self, raw_data_path, uncleaned_file_path):
        """
        Input Files: resume_samples.txt
        Creating Files:
            1) resume_samples.csv
        """
        self.data_path = raw_data_path
        self.uncleaned_file_path = uncleaned_file_path
        self.df = pd.DataFrame()

    def collect_resume_lbl_data(self):
        file_list = []
        for x in os.listdir(self.data_path):
            if x.endswith(".txt"):
                file_list.append(x.split('.')[0])
        return file_list

    def create_resume_lbl_df(self, file_list):
        resume_list = []
        lbl_list = []

        print_title("Creating Resume-Label Dataset")

        for file_name in file_list:
            resume = codecs.open(self.data_path+file_name+".txt", "rU", encoding='utf-8', errors='ignore')
            resume.seek(0)
            resume_list.append(resume.read())

            lbl = codecs.open(self.data_path+file_name+".lab", "rU", encoding='utf-8', errors='ignore')
            lbl.seek(0)
            lbl_list.append(lbl.read().replace("\n", ","))

        self.df['Resume'] = resume_list
        self.df['Label'] = lbl_list

        self.df.to_csv(self.uncleaned_file_path, index=False)

    def create_resume_skill_data(self):
        """

        """
        print_title("Creating Resume-Skill Dataset")
        input_data = codecs.open(self.data_path, "rU", encoding='utf-8', errors='ignore')
        input_data.seek(0)
        resume_lst = input_data.read().splitlines()

        skill_set = []
        resumes = []
        for resume in resume_lst:
            x = resume.split(':::')
            if len(x) == 3:
                skill_set.append(x[1])
                resumes.append(x[2])

        self.df['Skill'] = skill_set
        self.df['Resume'] = resumes

        self.df.to_csv(self.uncleaned_file_path, index=False)

    def basic_statistics(self):
        """
        Calculate basic statistics for the dataset.
        """
        print_title("Statistics", '`', 20)
        print("Total number of resumes:", len(self.df))
        print(f"Average length of resumes: {round(self.df['Resume'].apply(len).mean())} words")


if __name__ == "__main__":
    resume_lbl = DataGathering("../Data/Raw/resumes_corpus/", "../Data/Uncleaned/uncleaned_resume_lbl.csv")
    file_lst = resume_lbl.collect_resume_lbl_data()
    resume_lbl.create_resume_lbl_df(file_lst)
    resume_lbl.basic_statistics()

    resume_skill = DataGathering("../Data/Raw/resume_samples.txt", "../Data/Uncleaned/uncleaned_resume_skill.csv")
    resume_skill.create_resume_skill_data()
    resume_lbl.basic_statistics()
