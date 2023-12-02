import Preprocessor as prep
import os.path

data_path = "../Data/"
preprocess = 1

if __name__ == "__main__":
    p = prep.Preprocessor(data_path, "resume_samples.txt", "cleaned_dataset.csv")
    # TODO: Handle Folder Structures - Give file paths accordingly

    if (not os.path.isfile(p.cleaned_file_path)) or preprocess:
        resume_df = p.resume_df
        p.preprocess_skillset_df()
        p.preprocess_resume_df()
        print(p.resume_df.head())

    # Preprocessed Dataset
    p.read_cleaned_dataset()    # Contains only cleaned_resume ??
    p.basic_statistics()
    print(p.cleaned_df.head())

    p.generate_wordcloud(0)
    p.generate_wordcloud(50)
    p.generate_wordcloud(500)

