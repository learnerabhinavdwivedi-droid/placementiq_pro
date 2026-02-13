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
    st.set_page_config(page_title="PlacementIQ Pro | Access", layout="centered")
    st.markdown("""
        <style>
        .stApp { background: #0b0e11; color: white; }
        .auth-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 122, 0, 0.3);
            border-radius: 24px; padding: 40px; text-align: center;
            backdrop-filter: blur(12px); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.8);
        }
        </style>
        <div class="auth-card">
            <h1 style='color: #ff7a00;'>⚡ PLACEMENT IQ</h1>
            <p>Secure Engineering Portal</p>
        </div>
    """, unsafe_allow_html=True)
    
    password = st.text_input("Enter Passkey", type="password")
    if st.button("Unlock Dashboard"):
        if password == "1234":
            st.session_state.authenticated = True
            st.rerun()
    st.stop()

# --- 2. GLOBAL SETTINGS ---
st.set_page_config(page_title="PlacementIQ Pro | Adaptive 3D", layout="wide")

# --- 3. ENHANCED GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    .stApp { 
        background: radial-gradient(circle at top right, #1a1f25, #0b0e11); 
        color: #ffffff; 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 32px;
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 24px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .feature-card:hover {
        border: 1px solid rgba(255, 122, 0, 0.4);
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 122, 0, 0.1);
    }

    [data-testid="stMetricValue"] { color: #ff7a00 !important; font-weight: 800; }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff7a00, #ff9500);
        color: white !important; border-radius: 12px; border: none;
        padding: 10px 24px; font-weight: 700; width: 100%;
        box-shadow: 0 4px 15px rgba(255, 122, 0, 0.2);
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
    st.markdown("<h2 style='color:#ff7a00'>🧠 Career Suite</h2>", unsafe_allow_html=True)
    
    if 'current_target' in st.session_state:
        st.markdown(f"""
            <div style='background:rgba(255,122,0,0.1); padding:15px; border-radius:12px; border:1px solid #ff7a00;'>
                <small style='color:#ff7a00'>ACTIVE PATH</small><br>
                <b>{st.session_state.current_target}</b>
            </div>
        """, unsafe_allow_html=True)

    if 'missing_skills' in st.session_state and st.session_state.missing_skills:
        with st.expander("🔑 Skill Gap Explorer", expanded=True):
            for skill in st.session_state.missing_skills:
                st.markdown(f"""
                    <div style='margin-bottom:8px; padding:6px 12px; background:rgba(255,50,50,0.1); 
                         border-left:3px solid #ff3232; border-radius:4px; font-size:0.8rem;'>
                        MISSING: {skill.upper()}
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔍 Scan Live Opportunities"):
        st.session_state.scan_jobs = True

# --- 6. TOP NAVIGATION ---
st.markdown("<h1 style='text-align: center; letter-spacing: -1px; font-weight:800;'>⚡ PLACEMENT<span style='color:#ff7a00'>IQ</span> PRO</h1>", unsafe_allow_html=True)

# --- 7. MAIN DASHBOARD CONTENT ---
col_in, col_viz = st.columns([1, 1.2], gap="large")

with col_in:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("📝 Intelligence Ingestion")
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        cgpa = st.number_input("University CGPA", 0.0, 10.0, 7.8)
    with sub_col2:
        internships = st.selectbox("Internships", [0, 1, 2, 3])
        
    projects = st.slider("Innovation Projects", 0, 10, 3)
    comm = st.select_slider("Communication Mastery", options=range(1, 11), value=7)
    
    resume_input = st.text_area("📄 Digital Resume Content", placeholder="Paste resume text...", height=180)
    jd_input = st.text_area("🎯 Target Job Description", placeholder="Paste JD for auto-detection...", height=180)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 8. ADAPTIVE INTELLIGENCE ENGINE & 3D MAPPING ---
if jd_input and resume_input:
    detected_role = infer_target_path(jd_input)
    st.session_state.current_target = detected_role
    
    ats_score, found_skills, missing_skills = run_ats_check(resume_input, jd_input, detected_role)
    st.session_state.missing_skills = missing_skills
    
    try:
        input_data = np.array([[cgpa, (1 if internships > 0 else 0), projects, ats_score, comm]])
        prob_raw = model.predict_proba(input_data)[0][1]
        final_prob = round(min(20 + (prob_raw * 75), 98.2), 2)
    except:
        final_prob = ats_score

    with col_viz:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader(f"📊 3D Mapping: {detected_role}")
        
        radar_data = pd.DataFrame(dict(
            r=[cgpa, projects*2, comm, ats_score/10, internships*3],
            theta=['Academic', 'Innovation', 'Soft Skills', 'ATS Score', 'Exp']
        ))
        
        fig = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself', fillcolor='rgba(255, 122, 0, 0.2)', line_color='#ff7a00')
        fig.update_layout(
            polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False), 
                       angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
            showlegend=False, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Readiness Index", f"{final_prob}%", f"{round(final_prob-50, 1)}%")
        st.markdown('</div>', unsafe_allow_html=True)

        if missing_skills:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🛠️ Strategic Roadmap (Reddit Insight)")
            roadmap_advice = get_reddit_strategy(missing_skills, detected_role)
            for advice in roadmap_advice:
                st.markdown(f"**•** {advice}")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 9. JOB EXTRACTION ---
if st.session_state.get('scan_jobs'):
    st.markdown("---")
    current_target = st.session_state.get('current_target', 'SDE')
    st.subheader(f"🎯 Live {current_target} Openings Extracted")
    
    job_data = fetch_live_jobs(current_target)
    
    if not job_data.empty:
        cols = st.columns(2)
        for idx, row in job_data.iterrows():
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class="feature-card" style='margin-bottom:20px; padding:20px;'>
                        <h4 style='color:#ff7a00; margin:0;'>{row['Title']}</h4>
                        <p style='margin:5px 0 15px 0; opacity:0.7;'>{row['Company']}</p>
                        <a href="{row['URL']}" class="apply-btn" target="_blank">View Position</a>
                    </div>
                """, unsafe_allow_html=True)