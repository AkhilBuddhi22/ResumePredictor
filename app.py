import streamlit as st
import PyPDF2
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = [token.text for token in doc.ents if token.label_ in ["ORG", "PRODUCT", "PERSON"]]
    return list(set(skills))

# Function to calculate similarity
def calculate_similarity(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    corpus = resume_texts + [job_description]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    return similarity_scores.flatten()

# Streamlit UI
st.title("AI Resume Screening App")

# Job Description Input
job_description = st.text_area("Paste the Job Description:")

# Resume Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Screen Resumes"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and provide a job description.")
    else:
        resume_texts = []
        resume_data = []
        
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)
            skills = extract_skills(resume_text)
            resume_texts.append(resume_text)
            resume_data.append({"Name": uploaded_file.name, "Skills": ", ".join(skills), "Text": resume_text})
        
        # Compute similarity scores
        scores = calculate_similarity(resume_texts, job_description)
        
        # Display results
        results = pd.DataFrame(resume_data)
        results["Relevance Score"] = scores
        results = results.sort_values(by="Relevance Score", ascending=False)
        
        st.subheader("Screening Results")
        st.dataframe(results[["Name", "Skills", "Relevance Score"]])
