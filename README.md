# Final-Project-Group1
Final Project Repository for DATS 6312 Natural Language Processing


## MODEL CREATION
The Code folder contains all the project programs
The order of files to execute for successful project compilation
1. DataGathering.py - Accumulate raw txt files and export usable data into csv files
2. Preprocessor.py - Program to clean resumes files
3. Base.py - Program for base case model i.e., Multinomial Naive Bayes
4. Resume-Label.py - Program for multi label classfication using BERT and Linear Head for Job Roles
5. resume-label-cnn.py - Program for multi label classfication using BERT and CNN Head for Job Roles
6. resume-label-lstm.py - Program for multi label classfication using BERT and LSTM Head for Job Roles

## UI
Code/ui contains all programs to run streamlit ui
1. main.py - Main file to start streamlit server that contains all code for tool
2. helperfunctions - File with methods that support ui such as resume to job role inference, etc. (Not required to execute)


## Miscellaneous
Following are the files that were used to test the codes that are used in streamlit and also to experiment if additional analysis can be done on the existing data to extract more information
1. Skill-Role.py - Program to test classification for Skillset and Job Roles
2. Resume-Skill.py - Program to test classification for Skillset and Resume
3. Resume-Label-Inference - Program to run test cases on single resume to infer job roles from saved model
4. Resume-JD.py - Program to test similarity scores between resume and job description using TFIDF, BERT, and RoBERTa
5. utils.py - File with utility functions
6. nlp_ui.py, sidebar.py, test.py - Initial draft of ui codes to test streamlit functionality
