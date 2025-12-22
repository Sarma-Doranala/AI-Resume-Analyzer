import streamlit as st
import spacy
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load AI model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("NLP Model not found. Run: python -m spacy download en_core_web_sm")

def extract_text(file):
    if file.name.endswith('.pdf'):
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return None

def is_valid_resume(text):
    """Checks if the document looks like a resume based on common headers."""
    # List of common resume keywords
    resume_indicators = ['experience', 'education', 'skills', 'projects', 'summary', 'work history', 'qualification']
    text_lower = text.lower()
    # Check if at least 2 indicators are present
    matches = [indicator for indicator in resume_indicators if indicator in text_lower]
    return len(matches) >= 2

def extract_job_role(text):
    patterns = [
        r"(?i)job title[:\s]+([^\n]+)",
        r"(?i)position[:\s]+([^\n]+)",
        r"(?i)role[:\s]+([^\n]+)",
        r"(?i)looking for a\s+([^\n,.]+)",
        r"(?i)hiring for\s+([^\n,.]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    doc = nlp(text[:200])
    for chunk in doc.noun_chunks:
        if any(word in chunk.text.lower() for word in ['engineer', 'manager', 'developer', 'analyst', 'lead']):
            return chunk.text.title()
    return "Generic Professional Role"

def calculate_ats_score(resume_text, jd_text, match_percentage):
    score = match_percentage * 0.7 
    sections = ["experience", "education", "skills", "summary"]
    found_sections = [s for s in sections if s in resume_text.lower()]
    score += (len(found_sections) / len(sections)) * 15
    contact_info = [r'\d{10}', r'[\w\.-]+@[\w\.-]+']
    found_contact = [re.search(p, resume_text) for p in contact_info]
    score += (len([c for c in found_contact if c]) / len(contact_info)) * 15
    return min(score, 100)

# --- UI Layout ---
st.set_page_config(page_title="ATS Pro Analyzer", layout="wide")
st.title("📊 ATS Score & Resume Optimizer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 1. Your Resume")
    resume_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key="res")

with col2:
    st.subheader("📋 2. Job Description")
    jd_option = st.radio("How will you provide the JD?", ("Paste Text", "Upload JD File"), horizontal=True)
    jd_content = ""
    if jd_option == "Paste Text":
        jd_content = st.text_area("Paste the Job Description here", height=150)
    else:
        jd_file = st.file_uploader("Upload JD Document", type=["pdf", "docx"], key="jd_upload")
        if jd_file:
            jd_content = extract_text(jd_file)

if st.button("Generate Detailed Analysis"):
    if resume_file and jd_content:
        resume_text = extract_text(resume_file)
        
        # --- NEW: RESUME VALIDATION CHECK ---
        if not is_valid_resume(resume_text):
            st.error("❌ Invalid File: The uploaded document does not appear to be a correct resume. Please upload a valid resume containing Experience, Education, or Skills.")
        else:
            # Continue with Analysis
            detected_role = extract_job_role(jd_content)
            st.divider()
            st.markdown(f"### 🎯 Detected Role: **{detected_role}**")
            
            # Match Logic
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform([resume_text, jd_content])
            base_match = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100
            final_score = calculate_ats_score(resume_text, jd_content, base_match)
            
            st.header(f"Overall ATS Score: {int(final_score)}/100")
            st.progress(int(final_score))

            # Recruiter Focus (Skills AI)
            st.subheader("🎯 Recruiter Focus Area")
            job_doc = nlp(jd_content.lower())
            resume_doc = nlp(resume_text.lower())
            job_keys = set([t.text.upper() for t in job_doc if t.pos_ in ["PROPN", "NOUN"] and not t.is_stop])
            res_keys = set([t.text.upper() for t in resume_doc if t.pos_ in ["PROPN", "NOUN"] and not t.is_stop])
            missing = job_keys - res_keys
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**Keywords Matched:** {len(job_keys.intersection(res_keys))}")
            with c2:
                st.error(f"**Keywords Missing:** {len(missing)}")

            # Modification Box
            st.subheader("🛠️ Changes Needed to Improve Score")
            with st.container(border=True):
                if missing:
                    st.markdown(f"**1. Missing Skills:** Recruiters are looking for these terms. Add them to your Skills section: `{', '.join(list(missing)[:10])}`")
                st.markdown(f"**2. Keyword Integration:** Integrate the role title **'{detected_role}'** into your summary/headline.")
                if "@" not in resume_text:
                    st.markdown("**3. Contact Info:** No email detected. Ensure your contact details are at the very top.")
    else:
        st.warning("Please provide both documents.")