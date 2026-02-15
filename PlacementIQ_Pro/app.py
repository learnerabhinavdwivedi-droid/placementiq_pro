import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import os

# Core Logic Imports
from modules.ai_engine import run_ats_check, infer_target_path, get_reddit_strategy
from modules.scraper import fetch_live_jobs

# --- 1. INITIALIZATION & ACCESS GATE ---
st.set_page_config(page_title="PlacementIQ Pro | Adaptive 3D", layout="wide", initial_sidebar_state="expanded")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
        <style>
        .stApp { background: #0b0e11; color: white; }
        .auth-card {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 122, 0, 0.3);
            border-radius: 24px; padding: 40px; text-align: center;
            backdrop-filter: blur(12px); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.8);
            margin-top: 100px;
        }
        </style>
        <div class="auth-card">
            <h1 style='color: #ff7a00; font-weight: 800;'>⚡ PLACEMENT IQ</h1>
            <p style='font-weight: 600;'>Secure Engineering Portal</p>
        </div>
    """, unsafe_allow_html=True)
    password = st.text_input("Enter Passkey", type="password")
    if st.button("Unlock Dashboard"):
        if password == "1234":
            st.session_state.authenticated = True
            st.rerun()
    st.stop()

# --- 2. PERMANENT 3D DESIGN (Mechanical Theme) ---
# Rotating Wireframe Gear in the corner
three_js_code = """
<div id="container" style="position: fixed; bottom: 0; right: 0; width: 300px; height: 300px; z-index: 0; pointer-events: none; opacity: 0.4;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(300, 300);
    document.getElementById('container').appendChild(renderer.domElement);
    const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xff7a00, wireframe: true, transparent: true, opacity: 0.5 });
    const gear = new THREE.Mesh(geometry, material);
    scene.add(gear);
    camera.position.z = 30;
    function animate() { requestAnimationFrame(animate); gear.rotation.x += 0.005; gear.rotation.y += 0.01; renderer.render(scene, camera); }
    animate();
</script>
"""
components.html(three_js_code, height=0)

# --- 3. GLOBAL CSS (Sidebar, Navigation & High-Glow) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    .stApp { background: #0b0e11; color: #ffffff; font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3, b, label, .stMetricValue { font-weight: 800 !important; }

    /* Sidebar Fix and Toggle Layering */
    [data-testid="stSidebar"] { 
        background-color: #0b0e11 !important; 
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
        z-index: 1000 !important;
    }

    /* Horizontal Radio Rail */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        flex-direction: row !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 10px 8px !important;
        min-width: 80px !important;
        position: relative;
        transition: 0.3s;
    }

    /* Active Selection Glow & Indicator Arrow */
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        border: 2px solid #ff7a00 !important;
        background: rgba(255, 122, 0, 0.1) !important;
        box-shadow: 0 0 15px rgba(255, 122, 0, 0.4);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::after {
        content: '⌄';
        position: absolute;
        bottom: -25px;
        left: 50%;
        transform: translateX(-50%);
        color: #ff7a00;
        font-weight: 900;
        font-size: 1.4rem;
    }

    /* Hide Native Elements & Style Text */
    [data-testid="stSidebar"] div[role="radiogroup"] label div:first-child { display: none !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.65rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        color: white !important;
        text-align: center;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px; padding: 25px; margin-bottom: 20px;
        box-shadow: 0 4px 24px 0 rgba(0, 0, 0, 0.4);
    }

    .dev-credit {
        position: fixed; bottom: 15px; left: 20px; font-size: 0.7rem; font-weight: 800;
        color: rgba(255, 255, 255, 0.3); letter-spacing: 1px; text-transform: uppercase;
    }
    
    .stApp header { visibility: visible !important; background: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# --- 4. ENGINE LOAD ---
try:
    model = joblib.load("data/placement_model.pkl")
except:
    st.sidebar.warning("🤖 Model Offline. Run train_model.py first.")

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    # Logo & Name
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px;">
            <div style="background: linear-gradient(45deg, #ff7a00, #ff9500); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(255, 122, 0, 0.4);">
                <span style="color: white; font-weight: 800; font-size: 1.1rem;">IQ</span>
            </div>
            <h1 style='color: #ffffff; font-size: 1.5rem; letter-spacing: -1px; margin: 0;'>Placement<span style='color: #ff7a00;'>IQ.</span></h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #ff7a00; font-weight: 800; margin-bottom: 10px; font-size: 0.8rem;'>🧭 NAVIGATE PAGES</p>", unsafe_allow_html=True)
    page_selection = st.radio(label="Nav", options=["INGESTION", "ROADMAP", "FEED"], label_visibility="collapsed")
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    if 'current_target' in st.session_state:
        st.markdown(f'''
            <div style="background: rgba(255, 122, 0, 0.05); border: 1px solid rgba(255, 122, 0, 0.4); border-radius: 12px; padding: 15px;">
                <small style="color: #ff7a00; font-weight: 800;">TARGET PATH</small>
                <h3 style="margin: 0; color: white; font-size: 1.1rem;">{st.session_state.current_target}</h3>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="dev-credit">MADE BY @ABHINAV_DVD</div>', unsafe_allow_html=True)

# --- 6. PAGE CONTENT LOGIC ---

if page_selection == "INGESTION":
    st.markdown("<h2 style='color: #ff7a00;'>🖥️ INGESTION ZONE</h2>", unsafe_allow_html=True)
    col_in, col_stats = st.columns([1, 1], gap="large")
    
    with col_in:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📝 Profile Input")
        cgpa = st.number_input("University CGPA", 0.0, 10.0, 7.8)
        projects = st.slider("Innovation Projects", 0, 10, 3)
        resume_text = st.text_area("📄 Digital Resume", height=150, placeholder="Paste text content...")
        jd_text = st.text_area("🎯 Job Description", height=150, placeholder="Paste JD here...")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        if jd_text and resume_text:
            role = infer_target_path(jd_text)
            st.session_state.current_target = role
            score, found, missing = run_ats_check(resume_text, jd_text, role)
            st.session_state.missing_skills = missing
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader(f"🤖 AI Audit: {role}")
            st.metric("Readiness Score", f"{score}%")
            st.progress(score/100)
            st.write(f"**Key Skills Found:** {', '.join(found[:5])}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Input Resume and JD to begin AI Audit.")

elif page_selection == "ROADMAP":
    st.markdown("<h2 style='color: #ff7a00;'>📊 READINESS & ROADMAP</h2>", unsafe_allow_html=True)
    if 'current_target' in st.session_state:
        col_map, col_strat = st.columns([1.2, 1], gap="large")
        with col_map:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            radar_data = pd.DataFrame(dict(r=[8, 7, 9, 6, 5], theta=['Tech', 'Design', 'Soft Skills', 'ATS', 'Projects']))
            fig = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
            fig.update_traces(fill='toself', fillcolor='rgba(255, 122, 0, 0.2)', line_color='#ff7a00')
            fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False)), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_strat:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("🛠️ Strategic Roadmap")
            strategies = get_reddit_strategy(st.session_state.get('missing_skills', []), st.session_state.current_target)
            for s in strategies:
                st.markdown(f"<div style='border-left: 3px solid #ff7a00; padding-left: 15px; margin-bottom: 15px; font-weight: 600;'>{s}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif page_selection == "FEED":
    st.markdown("<h2 style='color: #ff7a00;'>🎯 OPPORTUNITY FEED</h2>", unsafe_allow_html=True)
    if 'current_target' in st.session_state:
        jobs = fetch_live_jobs(st.session_state.current_target)
        if not jobs.empty:
            for idx, row in jobs.iterrows():
                st.markdown(f"""
                    <div class="feature-card">
                        <h4 style="color: #ff7a00; margin: 0;">{row['Title']}</h4>
                        <p style="opacity: 0.8; margin-bottom: 15px;">{row['Company']}</p>
                        <a href="{row['URL']}" target="_blank" style="background: #ff7a00; color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: 800; font-size: 0.8rem;">APPLY NOW →</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Searching for roles...")