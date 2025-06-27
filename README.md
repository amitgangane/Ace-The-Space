Ace the Space: AI Powered Resume Job Recommendation System

Overview
"Ace the Space" is a machine learning project focused on revolutionizing the job market by providing an AI-powered job role prediction system. 
It aims to simplify the process of matching candidates with suitable roles by analyzing skills and job descriptions, offering a smart, data-driven approach to hiring


Project Objective
The primary objective of "Ace the Space" is to develop a recommendation system that analyzes resumes against job market requirements. 
This system will provide actionable insights for improving resume alignment with job demand and generate scores based on an algorithm.


Scope
The project reviews and compares resumes with job profiles that are currently in demand


Future Scope
The system will offer recommendations for improving resumes based on current job market trends.


Key Features
  Resume Analysis: Evaluates technical skills, experience, education, and other key elements of a resume. 
  Market Trends Comparison: Compares the resume against current job trends and employer requirements. 
  Personalized Recommendations: Suggests areas for improvement, including skills to acquire or highlight.


Technologies Used
The project utilizes the following technologies:
  Python 
  Pandas (for data manipulation) 
  NumPy 
  Scikit-learn (for preprocessing, feature extraction, and model evaluation metrics like 
  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report) 
  NLTK (for text cleaning and tokenization) 
  SpaCy (for advanced NLP tasks) 
  Sentence Transformers (specifically all-MiniLM-L6-v2 for generating embeddings) 
  Matplotlib (for visualization) 
  Seaborn (for enhanced visualizations) 


Dataset
The project uses two primary datasets:
  srd.csv (Resume Dataset) 
  jd.csv (Job Description Dataset)


Data Preprocessing
Data preprocessing steps included:
Handling Missing Values:
    Certification column in the Resume Dataset: Missing values were filled with "No Certification". 
  Job Description Dataset: Rows with missing 
    description were dropped. 
  Columns with too many missing values (
    max_salary, currency, normalized_salary, fips) were dropped from the Job Dataset. 
Text Cleaning & Preprocessing: This involved tasks such as converting text to lowercase, removing punctuation, stop words, and numbers, and performing lemmatization.


Model Building (Methodology)
The core methodology involves:
  Feature Engineering: Extracting skills, education, experience, and certifications from resumes and job descriptions. 
  Text Embedding: Using a pre-trained SentenceTransformer model (all-MiniLM-L6-v2) to convert resume and job description text into numerical embeddings. 
  Similarity Calculation: Computing cosine similarity scores between resume embeddings and job description embeddings to find the most suitable matches. 
  Matching Score: Assigning a matching score for resumes to jobs based on similarity. 


Model Evaluation
  The model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score. 
  Accuracy: The model achieved 89% accuracy. 
  Precision: With 79.39% precision, the model is often accurate when it suggests a designation. 
  Recall: With 75.49% recall, the model successfully identifies a significant portion of relevant cases. 


Results and Key Findings
The system provides a modern way of acing the job market by leveraging Natural Language Processing (NLP) and Machine Learning to accurately match candidates with suitable roles. 
The project is impactful due to:
  Efficiency: It automates manual job-matching efforts. 
  Scalability: It can be expanded for multiple industries and roles. 
  Future-Ready: With advancements like BERT and Deep Learning, accuracy and relevance can be further improved.


Challenges Encountered
  Selecting the proper dataset. 
  Choosing the right model. 
  Low generalization due to dataset limitation. 
  Case sensitivity or abbreviations can affect searching. 

Future Enhancements
  Enhancing Recommendation System: Integrate personalized job recommendations aligned with market demands and individual career trajectories. 
  Exploring Deep Learning Models: Integrate deep learning models for more advanced textual analysis to enhance accuracy and efficiency in job role prediction and resume matching. 
  Expanding Dataset Size: Increase the dataset size to ensure the model learns from a diverse and comprehensive set of resumes and job descriptions, leading to better pattern recognition and robust decision-making. 
  Improve Skill Extraction: Utilize NLP for better skill extraction
