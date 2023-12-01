import codecs
import pandas as pd


class Preprocessor:
    def __init__(self, input_file_path, input_csv_path):
        """
        Input Files: resume_samples.txt
        Creating Files:
            1) resume_samples.csv
        """
        self.input_file_path = input_file_path
        self.input_csv_path = input_csv_path
        self.input_data = codecs.open(self.input_file_path, "rU", encoding='utf-8', errors='ignore')


    def print_title(self, title, pattern="*", pattern_length=20, num_blank_lines=1):
        """

        :param title: String to be printed
        :param pattern: Pattern preceeding and Succeding the String in the title
        :param pattern_length: Length of pattern
        :param num_blank_lines: Total blank lines before and after the Title
        :return: Null
        """
        print((num_blank_lines // 2 + 1) * "\n", pattern_length * pattern, title, pattern_length * pattern,
              num_blank_lines // 2 * "\n")

    def read_dataset(self):
        """
        :param self:
        :param sample:
        :return:
        """

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
        resume_df.to_csv(self.input_csv_path, index = False)

        return resume_df