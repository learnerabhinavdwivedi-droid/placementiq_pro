import streamlit as st
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_nlp_model():
    """Loads the SBERT model for semantic matching."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def infer_target_path(jd_text):
    """Automatically detects the career track based on JD content."""
    model = load_nlp_model()
    
   # Expanded clusters covering major LinkedIn job categories
    clusters = {
    "SDE": "Software development, backend, frontend, full stack, web applications, APIs, microservices, git, OOP, system design",

    "Data Science": "Machine learning, data analysis, statistics, python, pandas, numpy, data visualization, predictive modeling, AI",

    "Data Analyst": "SQL, Excel, dashboards, Power BI, Tableau, reporting, business intelligence, data cleaning, ETL",

    "DevOps": "CI/CD, Docker, Kubernetes, AWS, Azure, GCP, infrastructure as code, Jenkins, monitoring, cloud deployment",

    "Cyber Security": "Network security, penetration testing, vulnerability assessment, SIEM, firewalls, ethical hacking, SOC",

    "Cloud Engineer": "Cloud architecture, AWS, Azure, GCP, serverless, cloud security, infrastructure automation",

    "Mobile Developer": "Android, iOS, Flutter, React Native, Swift, Kotlin, mobile UI/UX, app deployment",

    "UI/UX Designer": "User research, wireframing, prototyping, Figma, Adobe XD, usability testing, interaction design",

    "Product Manager": "Product lifecycle, roadmap, stakeholder management, agile, scrum, market research, KPIs",

    "Project Manager": "Project planning, risk management, budgeting, stakeholder communication, agile, waterfall, PMP",

    "Business Analyst": "Requirements gathering, stakeholder analysis, process improvement, documentation, gap analysis",

    "Marketing": "Digital marketing, SEO, SEM, social media, content marketing, email marketing, branding, analytics",

    "Sales": "Lead generation, CRM, negotiation, B2B, B2C, revenue growth, client acquisition",

    "Human Resources": "Recruitment, onboarding, payroll, employee engagement, HR policies, performance management",

    "Finance": "Financial analysis, accounting, budgeting, forecasting, taxation, auditing, Excel, SAP",

    "Operations": "Supply chain, logistics, inventory management, vendor coordination, process optimization",

    "Mechanical Design": "Mechanical engineering, CAD, SolidWorks, AutoCAD, FEA, thermodynamics, product design, GD&T",

    "Electrical Engineer": "Circuit design, PCB, embedded systems, power systems, MATLAB, control systems",

    "Civil Engineer": "Structural design, construction management, AutoCAD, site supervision, project estimation",

    "Quality Assurance": "Manual testing, automation testing, Selenium, test cases, regression testing, bug tracking",

    "Customer Support": "Customer service, ticketing systems, communication skills, troubleshooting, CRM"
    }

    labels = list(clusters.keys())
    descriptions = list(clusters.values())
    
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    cluster_embeddings = model.encode(descriptions, convert_to_tensor=True)
    
    scores = util.pytorch_cos_sim(jd_embedding, cluster_embeddings)[0]
    best_match_idx = scores.argmax().item()
    
    return labels[best_match_idx]

def run_ats_check(resume, jd, inferred_role):
    """Calculates semantic match and skill gaps for the 3D Analyzer."""
    model = load_nlp_model()
    
    # 1. Base Semantic Match
    emb = model.encode([resume, jd], convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(emb[0], emb[1])
    ats_score = round(float(cos_sim) * 100, 2)
    
    # 2. Expanded Hard-Skill Gap Analysis
    # Syncing this library with your expanded infer_target_path clusters
    SKILL_LIBRARY = {
        "SDE": ["python", "java", "git", "sql", "rest api", "oops"],
        "Data Science": ["python", "machine learning", "pandas", "numpy", "statistics"],
        "Data Analyst": ["sql", "excel", "power bi", "tableau", "etl"],
        "Mechanical Design": ["autocad", "solidworks", "fea", "thermodynamics", "ansys", "gd&t"],
        "Mechanical Engineer": ["thermodynamics", "fluid mechanics", "cad", "manufacturing"],
        "Quality Assurance": ["selenium", "manual testing", "test cases", "bug tracking"],
        "DevOps": ["docker", "kubernetes", "aws", "ci/cd", "jenkins"],
        "UI/UX Designer": ["figma", "adobe xd", "wireframing", "prototyping"]
    }
    
    # Get target skills or default to an empty list
    target_skills = SKILL_LIBRARY.get(inferred_role, [])
    
    # Standardize both resume and skills to lowercase for accurate matching
    resume_lower = resume.lower()
    found = [s for s in target_skills if s.lower() in resume_lower]
    missing = [s for s in target_skills if s.lower() not in resume_lower]
    
    return ats_score, found, missing

def get_reddit_strategy(missing_skills, role):
    """Provides subjective advice based on common Reddit engineering logic."""
    strategies = {
        "SOLIDWORKS": "Reddit ME Strategy: Build a CAD assembly of 50+ parts. Document the GD&T and stress analysis in a portfolio to prove mastery beyond basic shapes.",
        "AUTOCAD": "Reddit Design Strategy: Focus on 2D speed and precision. Mention specific industry standards (like ISO or ASME) you used during drafting.",
        "PYTHON": "Reddit Dev Strategy: Avoid 'tutorial hell.' Automate a real-world task (like data entry or file sorting) and host the script on GitHub.",
        "GIT": "Reddit SDE Strategy: Learn to resolve merge conflicts via command line. A clean commit history on a project is a huge green flag.",
        "FEA": "Reddit Analysis Strategy: Discuss your boundary conditions and validation methods. Results are useless if you can't explain why they are accurate.",
        "MATLAB": "Reddit Engineering Strategy: Focus on toolboxes relevant to your niche (e.g., Simulink for controls or Signal Processing)."
    }

    subjective_advice = []
    for skill in missing_skills:
        # Standardize key for lookup
        advice = strategies.get(skill.upper(), f"Reddit Strategy: Create a 'Proof of Competence' project for {skill}. Show it, don't just list it.")
        subjective_advice.append(advice)
    
    return subjective_advice

