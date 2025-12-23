import streamlit as st
import spacy
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. CONFIGURATION & AI LOAD ---
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("AI Model missing. Please ensure your requirements.txt includes the spacy model link.")

# --- 2. MODULAR UTILITY FUNCTIONS ---

def extract_text(file):
    """Handles text extraction from PDF and DOCX files."""
    if file.name.endswith('.pdf'):
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def is_valid_resume(text):
    """Verifies document structure to ensure it is a resume."""
    indicators = ['experience', 'education', 'skills', 'projects', 'summary', 'history']
    text_lower = text.lower()
    matches = [i for i in indicators if i in text_lower]
    return len(matches) >= 2

def analyze_job_identity(text):
    """
    Advanced Module: Detects Job Title AND Category (Sales, Dev, etc.).
    """
    text_lower = text.lower()
    # Define department categories
    categories = {
        "Development": ["software", "developer", "engineer", "frontend", "backend", "fullstack", "coder", "programming", "python", "java", "react"],
        "Sales": ["sales", "account executive", "business development", "inside sales", "outreach", "revenue", "client"],
        "Marketing": ["marketing", "seo", "social media", "content", "branding", "digital", "copywriter"],
        "Data & AI": ["data scientist", "data analyst", "machine learning", "ai", "sql", "tableau", "statistics"],
        "HR & Admin": ["hr", "human resources", "recruiter", "talent", "admin", "office", "operations"]
    }
    
    # 1. Detect Category
    detected_category = "Other Professional"
    for cat, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            detected_category = cat
            break

    # 2. Detect Specific Title using Regex patterns
    job_title = "Generic Professional Role"
    patterns = [
        r"(?i)job title[:\s]+([^\n]+)", 
        r"(?i)position[:\s]+([^\n]+)", 
        r"(?i)role[:\s]+([^\n]+)", 
        r"(?i)looking for a\s+([^\n,.]+)"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            job_title = match.group(1).strip().title()
            break
    
    # Fallback if no specific title found
    if job_title == "Generic Professional Role" and detected_category != "Other Professional":
        job_title = f"{detected_category} Specialist"
        
    return job_title, detected_category

def calculate_ats_score(resume_text, jd_text, match_percentage):
    """Calculates weighted ATS score based on keyword match and formatting."""
    score = match_percentage * 0.7 
    sections = ["experience", "education", "skills", "summary"]
    found_sections = [s for s in sections if s in resume_text.lower()]
    score += (len(found_sections) / len(sections)) * 15
    contact_info = [r'\d{10}', r'[\w\.-]+@[\w\.-]+']
    found_contact = [re.search(p, resume_text) for p in contact_info]
    score += (len([c for c in found_contact if c]) / len(contact_info)) * 15
    return min(score, 100)

# --- 3. STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Pro ATS Analyzer", layout="wide")
st.title("📊 AI Resume & Job Matcher PRO")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 1. Your Resume")
    resume_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key="res")

with col2:
    st.subheader("📋 2. Job Description")
    jd_option = st.radio("Input Method:", ("Paste Text", "Upload File"), horizontal=True)
    jd_content = ""
    if jd_option == "Paste Text":
        jd_content = st.text_area("Paste JD here...", height=150)
    else:
        jd_file = st.file_uploader("Upload JD Document", type=["pdf", "docx"], key="jd_upload")
        if jd_file:
            jd_content = extract_text(jd_file)

# --- 4. MAIN EXECUTION ENGINE ---

if st.button("Generate Detailed Analysis"):
    if resume_file and jd_content:
        resume_text = extract_text(resume_file)
        
        # Validation Check
        if not is_valid_resume(resume_text):
            st.error("❌ Invalid File: The uploaded document does not appear to be a correct resume. Please upload a file containing Experience, Education, or Skills.")
        else:
            # Identity Detection
            title, category = analyze_job_identity(jd_content)
            
            st.divider()
            st.markdown(f"### 🎯 Detected Role: **{title}**")
            st.markdown(f"#### 🏢 Category: **{category}**")
            st.info(f"The recruiter is looking for a professional in **{category}**. Analysis below is tailored for this track.")

            # ATS Scoring & Logic
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform([resume_text, jd_content])
            base_match = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100
            final_score = calculate_ats_score(resume_text, jd_content, base_match)
            
            st.header(f"Overall ATS Score: {int(final_score)}/100")
            st.progress(int(final_score))

            # NLP Keyword Extraction
            job_doc = nlp(jd_content.lower())
            resume_doc = nlp(resume_text.lower())
            job_keys = set([t.text.upper() for t in job_doc if t.pos_ in ["PROPN", "NOUN"] and not t.is_stop])
            res_keys = set([t.text.upper() for t in resume_doc if t.pos_ in ["PROPN", "NOUN"] and not t.is_stop])
            missing = job_keys - res_keys
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Keywords Matched:** {len(job_keys.intersection(res_keys))}")
            with c2:
                st.error(f"**Keywords Missing:** {len(missing)}")

            # Modification Box
            st.subheader("🛠️ Required Resume Modifications")
            with st.container(border=True):
                if missing:
                    st.markdown(f"**1. Category Keywords:** As this is a **{category}** role, add these terms: `{', '.join(list(missing)[:10])}`")
                
                st.markdown(f"**2. Title Integration:** Ensure the title **'{title}'** appears in your professional summary.")
                
                if "@" not in resume_text:
                    st.markdown("**3. Contact Info:** No email detected. Ensure your contact details are at the very top.")
    else:
        st.warning("Please provide both documents.")