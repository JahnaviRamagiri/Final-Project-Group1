import codecs
import pandas as pd

data_file = codecs.open("../Data/resume_samples.txt", "rU", encoding='utf-8', errors='ignore')
data_file.seek(0)
resume_lst = data_file.read().splitlines()

print(f"Resume list size: {len(resume_lst)}\n")

skill_set = []
resumes = []
for resume in resume_lst:
    x = resume.split(':::')
    if len(x) == 3:
        skill_set.append(x[1])
        resumes.append(x[2])

resume_df = pd.DataFrame({
    'skillset': skill_set,
    'resume': resumes
})

print(f"Resume data dimensions: {resume_df.shape}\n")
print(resume_df.head())
