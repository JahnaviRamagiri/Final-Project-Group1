import Preprocessor as prep


if __name__ == "__main__":
    p = prep.Preprocessor("resume_samples.txt", "cleaned_dataset.csv")
    # TODO: Handle Folder Structures - Give file paths accordingly
    resume_df = p.resume_df
    p.basic_statistics()
    p.preprocess_skillset_df()
    p.preprocess_resume_df()
    print(p.resume_df.head())

    # Preprocessed Dataset
    p.read_cleaned_dataset()    # Contains only cleaned_resume ??
    p.generate_wordcloud(0)
    p.generate_wordcloud(50)
    p.generate_wordcloud(500)

