import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Corrected Imports
from modules.ai_engine import run_ats_check, infer_target_path, get_reddit_strategy
from modules.scraper import fetch_live_jobs

# --- 1. INITIALIZATION & ACCESS GATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="PlacementIQ Secure Access", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #ff7a00;'>🔒 Student Passkey</h1>", unsafe_allow_html=True)
    password = st.text_input("Enter Passkey", type="password")
    if st.button("Unlock Dashboard"):
        if password == "1234":
            st.session_state.authenticated = True
            st.rerun()
    st.stop()

# --- 2. GLOBAL SETTINGS ---
st.set_page_config(page_title="PlacementIQ Pro | Adaptive 3D", layout="wide")
st.session_state.theme = 'dark'

if 'scan_jobs' not in st.session_state:
    st.session_state.scan_jobs = False

# --- 3. 3D GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    .stApp { background: #0b0e11; color: #ffffff; font-family: 'Inter', sans-serif; }
    .feature-card {
        background: rgba(28, 33, 39, 0.85); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px; padding: 30px; backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    h1, h2, h3, p, label { color: #ffffff !important; }
    .apply-btn {
        display: inline-block; padding: 12px 24px;
        background: linear-gradient(45deg, #ff7a00, #ff9500);
        color: white !important; font-weight: 800; text-decoration: none;
        border-radius: 12px; text-align: center; transition: 0.3s;
    }
    </style>

            
    """, unsafe_allow_html=True)

# --- 4. ENGINE LOAD ---
try:
    model = joblib.load("data/placement_model.pkl")
except:
    st.error("Engine Offline: Ensure 'placement_model.pkl' is in the 'data/' folder.")

# --- 5. SIDEBAR: CAREER SUITE ---
with st.sidebar:
    st.markdown("### 🧠 Career Suite")
    
    if 'current_target' in st.session_state:
        st.info(f"📍 Market Path: **{st.session_state.current_target}**")

    if 'missing_skills' in st.session_state and st.session_state.missing_skills:
        with st.expander("🔑 Show Missing Skill Keywords"):
            st.markdown("#### Industry Gaps Detected:")
            for skill in st.session_state.missing_skills:
                st.markdown(f"""
                    <div style="background: rgba(255, 122, 0, 0.1); border: 1px solid #ff7a00; 
                                border-radius: 5px; padding: 5px 10px; margin-bottom: 5px; 
                                color: #ff7a00; font-weight: bold; font-size: 0.85rem;">
                        ⚠️ {skill.upper()}
                    </div>
                """, unsafe_allow_html=True)
            st.caption("Add these to your resume projects to boost matching.")
    
    st.markdown("---")
    if st.button("🚀 Find 5 Live Matches"):
        st.session_state.scan_jobs = True

# --- 6. TOP NAVIGATION ---
st.markdown("<h1 style='text-align: center; color: #ff7a00;'>⚡ PLACEMENT IQ PRO: ADAPTIVE COMMAND CENTER</h1>", unsafe_allow_html=True)

# --- 7. MAIN DASHBOARD CONTENT ---
col_in, col_viz = st.columns([1, 1.2], gap="large")

with col_in:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("📝 Dynamic Data Ingestion")
    cgpa = st.slider("University CGPA", 5.0, 10.0, 7.8)
    internships = st.selectbox("Internships Completed", [0, 1, 2, 3])
    projects = st.slider("Major Projects", 0, 10, 3)
    comm = st.slider("Comm. Skill (1-10)", 1, 10, 7)
    
    # FIXED Labels
    resume_input = st.text_area("📄 Resume Content", placeholder="Paste resume text...", height=150)
    jd_input = st.text_area("🎯 Job Description", placeholder="Paste target JD for Auto-Detection...", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 8. ADAPTIVE INTELLIGENCE ENGINE & 3D MAPPING ---
if jd_input and resume_input:
    detected_role = infer_target_path(jd_input)
    st.session_state.current_target = detected_role
    
    # Run ATS Check
    ats_score, found_skills, missing_skills = run_ats_check(resume_input, jd_input, detected_role)
    st.session_state.missing_skills = missing_skills
    
    # ML Prediction Logic
    try:
        input_data = np.array([[cgpa, (1 if internships > 0 else 0), projects, ats_score, comm]])
        prob_raw = model.predict_proba(input_data)[0][1]
        final_prob = round(min(20 + (prob_raw * 75), 98.2), 2)
    except:
        final_prob = ats_score

    with col_viz:
        st.subheader(f"📊 3D Intelligence: {detected_role}")
        
        # Radar mapping
        radar_data = pd.DataFrame(dict(
            r=[cgpa, projects*2, comm, ats_score/10, internships*3],
            theta=['Academic', 'Innovation', 'Soft Skills', 'ATS Score', 'Exp']
        ))
        
        fig = px.line_polar(radar_data, r='r', theta='theta', line_close=True, template="plotly_dark")
        fig.update_traces(fill='toself', line_color='#ff7a00')
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Readiness Index", f"{final_prob}%", f"{round(final_prob-50, 1)}%")

        # Subjective Priority Roadmap (Reddit Logic)
        if missing_skills:
            st.markdown("---")
            st.markdown("### 🛠️ Subjective Priority Roadmap (Reddit Logic)")
            roadmap_advice = get_reddit_strategy(missing_skills, detected_role)
            for advice in roadmap_advice:
                st.info(advice)

# --- 9. REAL-TIME EXTRACTED MATCHES ---
if st.session_state.get('scan_jobs'):
    st.markdown("---")
    current_target = st.session_state.get('current_target', 'SDE')
    st.subheader(f"🎯 Live {current_target} Opportunities Extracted")
    
    job_data = fetch_live_jobs(current_target)
    
    if not job_data.empty:
        for i in range(min(20, len(job_data))):
            row = job_data.iloc[i]
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); padding:20px; border-radius:15px; margin-bottom:15px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div><h3 style="margin:0; color:#ff7a00;">{row['Title']}</h3><p style="margin:0; opacity:0.8;">{row['Company']}</p></div>
                        <a href="{row['URL']}" class="apply-btn" target="_blank">Apply Now</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)