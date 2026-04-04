"""
MSF FamilyGuard — Predictive Triage & Intervention System
==========================================
TWO distinct user areas:

  FOR USERS (social workers / FSC staff):
    📋 Case Triage — single case assessment (6 inputs only) + batch CSV upload & triage

  FOR PROJECT (capstone / supervisors):
    📊 Overview Dashboard
    🗺️  Cluster Intelligence
    📚 How the Model Works

Single case assessment requires only:
  Victim_Category, Age_Group, Sex, Reporting_Source,
  Household_Size, Prior_Social_Service_Contact
All engineered features are auto-derived from these 6 inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

# This gets the exact folder where app.py lives (the 'app' folder)
APP_DIR = Path(__file__).parent
# This goes up one level to the root of your repository
ROOT_DIR = APP_DIR.parent 
# Define exactly where your models and data live relative to the root
MODEL_DIR = ROOT_DIR / "models" # <-- Change "models" if your folder is named differently
DATA_DIR = ROOT_DIR / "data"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSF FamilyGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Cluster colour scheme */
.c0{background:#fffbeb;border-left:5px solid #f59e0b;}
.c1{background:#fef2f2;border-left:5px solid #ef4444;}
.c2{background:#f5f3ff;border-left:5px solid #8b5cf6;}
.c3{background:#fff1f2;border-left:5px solid #ec4899;}
.ccard{border-radius:10px;padding:14px 16px;margin-bottom:10px;}
.ctitle{font-weight:700;font-size:14px;margin-bottom:4px;}
/* Risk score display */
.risk-score{font-size:3.2rem;font-weight:800;text-align:center;line-height:1.1;}
.risk-lbl{text-align:center;font-size:14px;font-weight:600;margin-top:4px;}
/* Badges */
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;margin:2px;}
.b-red   {background:#fee2e2;color:#b91c1c;}
.b-amber {background:#fef3c7;color:#92400e;}
.b-green {background:#d1fae5;color:#065f46;}
.b-blue  {background:#dbeafe;color:#1e40af;}
.b-purple{background:#ede9fe;color:#5b21b6;}
.b-grey  {background:#f1f5f9;color:#475569;}
/* Intervention cards */
.icard{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;margin-top:6px;}
/* Section headers */
.sec-hdr{font-size:17px;font-weight:700;border-bottom:2px solid #e2e8f0;padding-bottom:5px;margin:1.2rem 0 0.7rem;}
/* User/Project tabs styling */
div[data-testid="stTabs"] button[data-baseweb="tab"]{font-size:15px;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_CFG = {
    0: dict(name="Silent Pressure Cooker", emoji="🫧",  color="#f59e0b", css="c0",
            badge="b-green",  badge_txt="Preventive",
            profile="No financial stress, Spouse + VA victims, latent risk. Families are not reaching services despite underlying vulnerability.",
            signal="Low ecological risk. Danger is hidden — under-reporting is common.",
            who="Community Development Councils, RCs, FSC outreach workers",
            urgency="Preventive — act within 2–4 weeks",
            interventions=[
                ("Grassroots outreach",   "Engage through CCs and RCs to build trust before crisis escalates",              "b-blue"),
                ("Financial screening",   "Link to ComCare, Workfare, utility assistance",                                  "b-green"),
                ("VA welfare checks",     "Schedule routine home visits for elderly/disabled household members",             "b-amber"),
                ("PPO awareness",         "Ensure female spouses know their legal rights under the Women's Charter",         "b-purple"),
            ]),
    1: dict(name="Ecological Hotspot Child", emoji="🏘️", color="#ef4444", css="c1",
            badge="b-red",    badge_txt="High Alert",
            profile="100% financial stress, zero prior contact, all child victims, high community incidence. Risk is environmental — these families are undetected.",
            signal="63% high-risk. First contact is the critical intervention window.",
            who="School counsellors, Allied Educators, MOE-FSC liaisons, KidSTART",
            urgency="High alert — triage within 48 hours",
            interventions=[
                ("School monitoring",      "Equip AEDs and counsellors to detect unexplained injuries and absenteeism",     "b-red"),
                ("Childcare alerts",       "Train childcare centre staff on physical abuse indicators for 0-6 year olds",    "b-red"),
                ("Estate campaign",        "Parenting workshops and KidSTART in high-incidence community estates",           "b-amber"),
                ("Reporter training",      "Refresh CYPA mandatory reporting obligations for teachers and GPs",              "b-blue"),
            ]),
    2: dict(name="Chronic Spousal Cycle",     emoji="🔄", color="#8b5cf6", css="c2",
            badge="b-purple", badge_txt="Chronic",
            profile="100% financial stress, 26% prior contact, mostly female spouses. Lower severity but recurring — patterns repeat across years.",
            signal="Recidivism risk 0.38. Women stay due to financial dependency or fear.",
            who="PSC Protection Officers, PAVE, TRANS-SAFE Centre, PPO unit",
            urgency="Moderate — structured follow-up within 1 week",
            interventions=[
                ("Long-term counselling",  "Assign dedicated case worker — minimum 12-month engagement plan",                "b-purple"),
                ("DVERT referral",         "Fast-track to Domestic Violence Emergency Response Team for police cases",        "b-red"),
                ("Financial independence", "Connect to WSG employment support, skills training, childcare subsidies",        "b-green"),
                ("Safety planning",        "Ensure victim has PPO, safe word, and emergency contacts documented",            "b-amber"),
                ("Perpetrator programme",  "Mandatory Counselling Order assessment for repeat incidents",                    "b-blue"),
            ]),
    3: dict(name="Chronic Family Cycle",      emoji="🔁", color="#ec4899", css="c3",
            badge="b-red",    badge_txt="URGENT",
            profile="100% prior contact, 100% financial stress, 100% intergenerational risk, all child victims. Highest recidivism (0.87). Past intervention did not resolve the root cause.",
            signal="67% high-risk. Violence is entrenched across generations.",
            who="MSF PSV, CPSCs, Foster Care caseworkers, School DSAs",
            urgency="URGENT — escalate within 24 hours",
            interventions=[
                ("Statutory review",       "Escalate to MSF PSV for CYPA investigation and consider supervision order",     "b-red"),
                ("Out-of-home assessment", "Evaluate foster/kinship care placement if home remains unsafe",                  "b-red"),
                ("Intensive therapy",      "Multisystemic Therapy (MST) or Family Preservation Services",                   "b-amber"),
                ("Parenting programme",    "Triple P or Circle of Security — mandatory if case stays at home",               "b-blue"),
                ("School welfare check",   "Weekly welfare checks via school; flag absenteeism immediately",                 "b-purple"),
            ]),
}

ABUSE_PROXY = {"Physical Abuse":0.8,"Sexual Abuse":1.0,"Neglect":0.5,"Emotional & Psychological Abuse":0.6, "Self-Neglect": 0.5}
AGE_ORDINAL = {"0-6 years":0,"7-12 years":1,"13-16 years":2,"17-18 years":3,
               "18-29 years":4,"18-64 years":5,"30-39 years":6,"40-49 years":7,
               "50-64 years":8,"65+ years":9}
SINGAPORE_ESTATES = [
    "Ang Mo Kio","Bedok","Bishan","Bukit Batok","Bukit Merah","Bukit Panjang",
    "Bukit Timah","Central Area","Choa Chu Kang","Clementi","Geylang","Hougang",
    "Jurong East","Jurong West","Kallang","Marine Parade","Novena","Pasir Ris",
    "Punggol","Queenstown","Sembawang","Sengkang","Serangoon","Tampines",
    "Toa Payoh","Woodlands","Yishun",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_arts():
    required = ["msf_cluster_model.pkl","scaler_cluster.pkl","model_meta.pkl",
                "scaler.pkl","label_encoders.pkl"]
    
    # Check if files exist inside the MODEL_DIR
    for f in required:
        file_path = MODEL_DIR / f
        if not file_path.exists():
            st.error(f"Missing: {file_path}  →  Please check your GitHub folder structure.")
            st.stop()
            
    # Search for the HRF and IIN models inside the MODEL_DIR
    hrf_cands = sorted(MODEL_DIR.glob("msf_hrf_model_*.pkl"))
    iin_cands = sorted(MODEL_DIR.glob("msf_iin_model_*.pkl"))
    
    if not hrf_cands: st.error(f"Missing msf_hrf_model_*.pkl in {MODEL_DIR}"); st.stop()
    if not iin_cands: st.error(f"Missing msf_iin_model_*.pkl in {MODEL_DIR}"); st.stop()
    
    return {
        "hrf":     joblib.load(hrf_cands[0]),
        "iin":     joblib.load(iin_cands[0]),
        "km":      joblib.load(MODEL_DIR / "msf_cluster_model.pkl"),
        "sc_cl":   joblib.load(MODEL_DIR / "scaler_cluster.pkl"),
        "sc_mod":  joblib.load(MODEL_DIR / "scaler.pkl"),
        "meta":    joblib.load(MODEL_DIR / "model_meta.pkl"),
        "le":      joblib.load(MODEL_DIR / "label_encoders.pkl"),
    }

arts = load_arts()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATA (no processed_data.csv — cluster derived on-the-fly)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_base():
    # Use DATA_DIR instead of hardcoded "../data"
    cands = list(DATA_DIR.glob("*v3*.csv")) + list(DATA_DIR.glob("final_refined*.csv"))
    if not cands:
        st.error(f"Cannot find the dataset CSV. Looked in: {DATA_DIR}")
        st.stop()
        
    df = pd.read_csv(sorted(cands)[-1], low_memory=False)
    df.loc[df["Victim_Category"]=="Non-Elderly VA","Age_Group"] = "18-64 years"

    # Re-derive engineered features from raw columns
    fsi = ((df["Community_Incidence_Rate"]>=1.65)|
           ((df["Victim_Category"]=="Spouse")&(df["Prior_Social_Service_Contact"]==1)&(df["Household_Size"]>=4))).astype(int)
    df["Financial_Stress_Index"] = fsi
    igf = ((df["Victim_Category"]=="Child")&(df["Prior_Social_Service_Contact"]==1)&(fsi==1)).astype(int)
    df["Intergenerational_Risk_Flag"] = igf
    df["Gender_Power_Imbalance_Flag"] = (
        ((df["Victim_Category"]=="Spouse")&(df["Sex"]=="Female")&(df["Severity_Score"]>=3.0))|
        ((df["Victim_Category"]=="Elderly VA")&(df["Sex"]=="Female"))
    ).astype(int)
    df["Recidivism_Risk_Score"] = (
        (df["Prior_Social_Service_Contact"]*2+fsi*1+igf*1+(df["Household_Size"]>=5).astype(int)*1)/5.0
    ).clip(0,1).round(2)
    ap = df["Type_of_Abuse"].map(ABUSE_PROXY)
    df["Ecological_Risk_Index"] = (
        ap*0.30+((df["Household_Size"]-2)/4.0).clip(0,1)*0.20+
        (df["Community_Incidence_Rate"]/3.0).clip(0,1)*0.25+fsi*0.15+df["Prior_Social_Service_Contact"]*0.10
    ).round(3)

    # Assign cluster via saved model
    meta  = arts["meta"]; le_cl = meta["le_cl"]
    df_cl = df.copy()
    for col in meta["cluster_cat_features"]:
        df_cl[f"{col}_cl_enc"] = le_cl[col].transform(df_cl[col])
    X_cl = df_cl[meta["cluster_feature_cols"]].values.astype(float)
    df["Cluster"] = arts["km"].predict(arts["sc_cl"].transform(X_cl))
    return df

base_df = load_base()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def derive_features(vc, age, sex, src, hs, psc, sev=3.0, cir=2.4):
    """Derive all 7 engineered features from the 6 user inputs + defaults."""
    if vc == "Non-Elderly VA": age = "18-64 years"
    fsi = int(cir>=1.65 or (vc=="Spouse" and psc==1 and hs>=4))
    igf = int(vc=="Child" and psc==1 and fsi==1)
    gpf = int((vc=="Spouse" and sex=="Female" and sev>=3.0) or (vc=="Elderly VA" and sex=="Female"))
    rss = round(min((psc*2+fsi+igf+(1 if hs>=5 else 0))/5.0,1.0),2)
    ap  = ABUSE_PROXY.get("Physical Abuse",0.8)
    eri = round(ap*0.30+min((hs-2)/4.0,1)*0.20+min(cir/3.0,1)*0.25+fsi*0.15+psc*0.10,3)
    return dict(fsi=fsi,igf=igf,gpf=gpf,rss=rss,eri=eri,age=age,sev=sev,cir=cir)

def get_cluster(vc, age, hs, psc, fsi, igf, rss, eri, sev=3.0, cir=2.4):
    meta = arts["meta"]; le_cl = meta["le_cl"]
    cl_row = {}
    for col in meta["cluster_cat_features"]:
        val = {"Victim_Category":vc,"Type_of_Abuse":"Physical Abuse","Age_Group":age}.get(col,"Child")
        cl_row[f"{col}_cl_enc"] = int(le_cl[col].transform([val])[0]) if val in le_cl[col].classes_ else 0
    num_vals = {"Financial_Stress_Index":fsi,"Household_Size":hs,"Prior_Social_Service_Contact":psc,
                "Community_Incidence_Rate":cir,"Severity_Score":sev,"Recidivism_Risk_Score":rss,
                "Ecological_Risk_Index":eri,"Intergenerational_Risk_Flag":igf}
    for col in meta["cluster_num_features"]: cl_row[col]=num_vals.get(col,0)
    X = np.array([[cl_row.get(f,0) for f in meta["cluster_feature_cols"]]],dtype=float)
    return int(arts["km"].predict(arts["sc_cl"].transform(X))[0])

def predict(vc, age, sex, src, hs, psc, fsi, igf, gpf, rss, eri, cluster, sev=3.0, cir=2.4, year=2024):
    meta = arts["meta"]; le = arts["le"]
    feat_row = {}
    for col in meta["feature_cols"]:
        if col == "Cluster": feat_row[col]=cluster
        elif col == "Age_Group": feat_row[col]=AGE_ORDINAL.get(age,4)
        elif col in le:
            val={"Victim_Category":vc,"Sex":sex,"Reporting_Source":src}.get(col,le[col].classes_[0])
            feat_row[col]=int(le[col].transform([val])[0]) if val in le[col].classes_ else 0
        else:
            feat_row[col]={"Year":year,"Household_Size":hs,"Prior_Social_Service_Contact":psc,
                           "Severity_Score":sev,"Community_Incidence_Rate":cir,
                           "Financial_Stress_Index":fsi,"Intergenerational_Risk_Flag":igf,
                           "Gender_Power_Imbalance_Flag":gpf,"Recidivism_Risk_Score":rss,
                           "Ecological_Risk_Index":eri}.get(col,0)
    X = np.array([[feat_row[c] for c in meta["feature_cols"]]],dtype=float)
    Xs = arts["sc_mod"].transform(X)
    hrf_prob = float(arts["hrf"].predict_proba(Xs)[0,1])
    iin_prob = float(arts["iin"].predict_proba(Xs)[0,1]) if hrf_prob>=0.50 else None
    return hrf_prob, iin_prob

def tier(prob):
    if prob>=0.70: return "🔴 Critical",  "#ef4444", "b-red"
    if prob>=0.45: return "🟡 Elevated",  "#f59e0b", "b-amber"
    return              "🟢 Routine",   "#10b981", "b-green"

def render_interventions(cfg):
    for title,desc,badge_cls in cfg["interventions"]:
        st.markdown(
            f'<div class="icard"><span class="badge {badge_cls}">{title}</span>'
            f'<span style="font-size:13px;margin-left:10px">{desc}</span></div>',
            unsafe_allow_html=True)

def cluster_card(cid, n=None, hrf=None):
    cfg = CLUSTER_CFG[cid]
    extra = ""
    if n is not None:
        extra = f'<div style="font-size:12px;color:#64748b">n={n} ({n/len(base_df):.0%})</div>'
    hrf_part = ""
    if hrf is not None:
        hrf_part = f'<div style="font-size:20px;font-weight:800;color:{cfg["color"]}">{hrf:.0%} high risk</div>'
    st.markdown(
        f'<div class="ccard {cfg["css"]}">'
        f'<div class="ctitle">{cfg["emoji"]} {cfg["name"]}</div>'
        f'{hrf_part}{extra}'
        f'<span class="badge {cfg["badge"]}">{cfg["badge_txt"]}</span>'
        f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — two separate navigation areas
# ─────────────────────────────────────────────────────────────────────────────
def set_nav_user():
    st.session_state.last_group = "user"
    st.session_state.radio_proj = None

def set_nav_proj():
    st.session_state.last_group = "proj"
    st.session_state.radio_user = None

if "last_group" not in st.session_state:
    st.session_state.last_group = "user"
    st.session_state.radio_user = "📋 Case Triage"
    st.session_state.radio_proj = None

with st.sidebar:
    st.markdown("## 🛡️ MSF FamilyGuard")
    st.caption("Family Violence Predictive Triage System")
    st.divider()

    st.markdown("**For social workers / FSC staff**")
    user_page = st.radio(
        "user_nav", ["📋 Case Triage"],
        label_visibility="collapsed",
        on_change=set_nav_user,
        key="radio_user"
    )
    st.divider()

    st.markdown("**Project information**")
    proj_page = st.radio(
        "proj_nav",
        ["📊 Overview Dashboard", "🗺️ Cluster Intelligence", "📚 How the Model Works"],
        label_visibility="collapsed",
        on_change=set_nav_proj,
        key="radio_proj"
    )
    st.divider()
    st.caption(f"Best Layer 2: {arts['meta']['hrf_model_name']}")
    st.caption(f"Best Layer 3: {arts['meta']['iin_model_name']}")
    st.caption("K-Means k=4 · Temporal split 2021→2024")

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING — detect which radio was changed
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.last_group == "user":
    active_page = st.session_state.get("radio_user") or "📋 Case Triage"
else:
    active_page = st.session_state.get("radio_proj") or "📊 Overview Dashboard"


# ─────────────────────────────────────────────────────────────────────────────
# ══ USER PAGE: Case Triage ═══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if active_page == "📋 Case Triage":
    st.title("📋 Case Triage")
    st.caption("Assess individual cases or upload a CSV for batch triage.")

    tab_single, tab_batch = st.tabs(["🔍 Single Case Assessment", "📥 Batch Upload & Triage"])

    # ── SINGLE CASE ──────────────────────────────────────────────────────────
    with tab_single:
        st.markdown("Enter the 6 case fields below. All risk scores are computed automatically.")

        st.markdown("**Step 1: Who is the victim?**")
        c1, c2, c3 = st.columns(3)
        vc  = c1.selectbox("Victim category *", ["Child","Spouse","Elderly VA","Non-Elderly VA"])
        sex = c2.selectbox("Sex *", ["Female","Male"])
        src = c3.selectbox("Reporting source *",
                           ["Police","School","Hospital","Social Service","Hotline","Community"])

        age_opts = {"Child":["0-6 years","7-12 years","13-16 years","17-18 years"],
                    "Spouse":["20-64 years"],"Elderly VA":["65+ years"],
                    "Non-Elderly VA":["18-64 years"]}
        age = st.selectbox("Age group *", age_opts[vc])

        st.markdown("**Step 2: Household information**")
        c4, c5 = st.columns(2)
        hs  = c4.slider("Household size *", 2, 6, 4,
                         help="Number of people living in the same household")
        psc = c5.selectbox("Prior social service contact *",
                           [0, 1], format_func=lambda x: "Yes — has prior contact" if x else "No — first contact",
                           help="Has this household previously been known to any social service agency?")

        submitted = st.button("Run Risk Assessment", use_container_width=True, type="primary")

        if submitted or st.session_state.get("case_assessed", False):
            st.session_state.case_assessed = True
            feats = derive_features(vc, age, sex, src, hs, psc)
            cid   = get_cluster(vc, feats["age"], hs, psc, feats["fsi"], feats["igf"],
                                 feats["rss"], feats["eri"])
            hrf_prob, iin_prob = predict(vc, feats["age"], sex, src, hs, psc,
                                          feats["fsi"], feats["igf"], feats["gpf"],
                                          feats["rss"], feats["eri"], cid)
            t_lbl, t_col, t_badge = tier(hrf_prob)
            cfg = CLUSTER_CFG[cid]

            st.divider()

            # ── Score + Archetype row ─────────────────────────────────────────
            col_score, col_arch, col_iin = st.columns([1, 1.4, 1])

            with col_score:
                st.markdown(
                    f'<div style="text-align:center;padding:18px;background:#f8fafc;'
                    f'border-radius:12px;border:2px solid {t_col}">'
                    f'<div class="risk-score" style="color:{t_col}">{hrf_prob:.0%}</div>'
                    f'<div class="risk-lbl" style="color:{t_col}">{t_lbl}</div>'
                    f'<div style="font-size:12px;color:#64748b;margin-top:6px">High Risk probability</div>'
                    f'</div>', unsafe_allow_html=True)

            with col_arch:
                st.markdown(
                    f'<div class="ccard {cfg["css"]}" style="height:100%">'
                    f'<div class="ctitle" style="font-size:16px">{cfg["emoji"]} {cfg["name"]}</div>'
                    f'<span class="badge {cfg["badge"]}">{cfg["badge_txt"]}</span>'
                    f'<div style="font-size:12px;color:#64748b;margin-top:8px">'
                    f'<b>Urgency:</b> {cfg["urgency"]}</div>'
                    f'<div style="font-size:12px;color:#475569;margin-top:6px">{cfg["signal"]}</div>'
                    f'</div>', unsafe_allow_html=True)

            with col_iin:
                if iin_prob is not None:
                    iin_col = "#ef4444" if iin_prob>=0.5 else "#f59e0b"
                    iin_txt = "URGENT — same day" if iin_prob>=0.5 else "Schedule within 1 week"
                    st.markdown(
                        f'<div style="text-align:center;padding:18px;background:#f8fafc;'
                        f'border-radius:12px;border:1.5px solid {iin_col}">'
                        f'<div style="font-size:2.2rem;font-weight:800;color:{iin_col}">{iin_prob:.0%}</div>'
                        f'<div style="font-size:13px;font-weight:600;color:{iin_col}">{iin_txt}</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:6px">Intervention urgency</div>'
                        f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="text-align:center;padding:18px;background:#f8fafc;'
                        'border-radius:12px;border:1.5px solid #e2e8f0;">'
                        '<div style="font-size:1.6rem;color:#94a3b8">—</div>'
                        '<div style="font-size:13px;color:#94a3b8">Not applicable</div>'
                        '<div style="font-size:12px;color:#cbd5e1;margin-top:6px">Urgency only computed for High Risk cases</div>'
                        '</div>', unsafe_allow_html=True)

            # ── Intervention strategy ─────────────────────────────────────────
            st.markdown(
                f'<div class="sec-hdr">Recommended intervention — {cfg["emoji"]} {cfg["name"]}</div>',
                unsafe_allow_html=True)
            st.caption(f"**Lead agency:** {cfg['who']}  |  **Urgency:** {cfg['urgency']}")
            render_interventions(cfg)

            # ── Radar chart ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">Risk factor radar</div>', unsafe_allow_html=True)
            cats = ["Ecological Risk","Recidivism","Financial Stress","Community Rate","Household","Prior Contact"]
            vals = [feats['eri'], feats['rss'], feats['fsi'],
                    min(feats['cir']/2.8,1), (hs-2)/4, float(psc)]
            r_c, g_c, b_c = int(t_col[1:3], 16), int(t_col[3:5], 16), int(t_col[5:7], 16)
            fig = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself', fillcolor=f'rgba({r_c},{g_c},{b_c},0.2)', line_color=t_col))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                              showlegend=False, height=300,
                              margin=dict(t=20,b=20,l=40,r=40))
            st.plotly_chart(fig, use_container_width=True)

    # ── BATCH UPLOAD ─────────────────────────────────────────────────────────
    with tab_batch:
        st.markdown("Upload a CSV of cases. The model scores each row and assigns a risk tier and archetype.")

        # Template download
        template_cols = ["Victim_Category","Age_Group","Sex","Reporting_Source",
                         "Household_Size","Prior_Social_Service_Contact"]
        st.download_button(
            "⬇ Download CSV template (6 columns)",
            pd.DataFrame(columns=template_cols).to_csv(index=False),
            "msf_case_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload cases CSV", type="csv")

        if uploaded:
            raw = pd.read_csv(uploaded)
            if "Family_ID" not in raw.columns:
                raw.insert(0,"Family_ID",[f"FAM-{i:04d}" for i in range(1,len(raw)+1)])
            if "Estate" not in raw.columns:
                raw["Estate"] = np.random.choice(SINGAPORE_ESTATES, size=len(raw))

            results = []
            progress = st.progress(0, text="Scoring cases...")
            for idx, row in raw.iterrows():
                r = row.to_dict()
                vc_  = str(r.get("Victim_Category","Child"))
                age_ = str(r.get("Age_Group","0-6 years"))
                sex_ = str(r.get("Sex","Female"))
                src_ = str(r.get("Reporting_Source","School"))
                hs_  = int(r.get("Household_Size",4))
                psc_ = int(r.get("Prior_Social_Service_Contact",0))
                try:
                    f_   = derive_features(vc_,age_,sex_,src_,hs_,psc_)
                    cid_ = get_cluster(vc_,f_["age"],hs_,psc_,f_["fsi"],f_["igf"],f_["rss"],f_["eri"])
                    hp,ip= predict(vc_,f_["age"],sex_,src_,hs_,psc_,
                                   f_["fsi"],f_["igf"],f_["gpf"],f_["rss"],f_["eri"],cid_)
                    tlbl,_,_= tier(hp)
                    cfg_  = CLUSTER_CFG[cid_]
                    results.append({
                        "Family_ID"      : r.get("Family_ID","—"),
                        "Estate"         : r.get("Estate","—"),
                        "Victim_Category": vc_,
                        "Age_Group"      : age_,
                        "Sex"            : sex_,
                        "Reporting_Source": src_,
                        "Household_Size" : hs_,
                        "Prior_Contact"  : "Yes" if psc_ else "No",
                        "Risk_Score"     : f"{hp:.0%}",
                        "_prob"          : hp,
                        "Tier"           : tlbl,
                        "Archetype"      : f"{cfg_['emoji']} {cfg_['name']}",
                        "IIN"            : f"{ip:.0%}" if ip else "—",
                        "Urgency"        : cfg_["urgency"],
                    })
                except Exception as e:
                    results.append({"Family_ID":r.get("Family_ID","?"),"Tier":"⚠ Error","_prob":0,
                                    "Risk_Score":"—","Archetype":str(e)[:40],"IIN":"—","Urgency":"—",
                                    "Estate":"—","Victim_Category":vc_,"Age_Group":age_,
                                    "Sex":sex_,"Reporting_Source":src_,"Household_Size":hs_,"Prior_Contact":"?"})
                progress.progress((idx+1)/len(raw), text=f"Scoring {idx+1}/{len(raw)}...")

            progress.empty()
            res_df = pd.DataFrame(results).sort_values("_prob",ascending=False).reset_index(drop=True)

            # KPIs
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total cases",       len(res_df))
            k2.metric("Critical (≥70%)",   (res_df["Tier"]=="🔴 Critical").sum())
            k3.metric("Elevated (45-69%)", (res_df["Tier"]=="🟡 Elevated").sum())
            k4.metric("Routine (<45%)",    (res_df["Tier"]=="🟢 Routine").sum())

            # Display table
            display_cols = ["Family_ID","Estate","Victim_Category","Age_Group","Sex",
                            "Risk_Score","Tier","Archetype","IIN","Urgency"]
            st.dataframe(res_df[[c for c in display_cols if c in res_df.columns]],
                         hide_index=True, use_container_width=True)

            # ── Case detail inspector ─────────────────────────────────────────
            st.markdown('<div class="sec-hdr">Case detail inspector</div>', unsafe_allow_html=True)
            sel_id = st.selectbox("Select a Family ID to view full detail:", res_df["Family_ID"].tolist())
            sel = res_df[res_df["Family_ID"]==sel_id].iloc[0]
            sel_prob = sel["_prob"]
            t_lbl2,t_col2,_ = tier(sel_prob)
            cfg2 = CLUSTER_CFG[[c for c in CLUSTER_CFG if CLUSTER_CFG[c]["name"] in sel.get("Archetype","")][0]] if any(CLUSTER_CFG[c]["name"] in sel.get("Archetype","") for c in CLUSTER_CFG) else CLUSTER_CFG[0]

            d1,d2 = st.columns([1,1.5])
            with d1:
                st.markdown(
                    f'<div style="text-align:center;padding:16px;background:#f8fafc;'
                    f'border-radius:10px;border:2px solid {t_col2}">'
                    f'<div style="font-size:2.8rem;font-weight:800;color:{t_col2}">{sel_prob:.0%}</div>'
                    f'<div style="font-size:13px;font-weight:600;color:{t_col2}">{t_lbl2}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown(f"**Archetype:** {sel.get('Archetype','—')}")
                st.markdown(f"**Urgency:** {sel.get('Urgency','—')}")
                st.markdown(f"**Immediate Intervention:** {sel.get('IIN','—')}")
            with d2:
                st.markdown(f'<div class="sec-hdr">Intervention steps</div>', unsafe_allow_html=True)
                st.caption(f"**Lead agency:** {cfg2['who']}")
                render_interventions(cfg2)

            # ── Estate-level risk chart ───────────────────────────────────────
            if "Estate" in res_df.columns and res_df["Estate"].notna().any():
                estate_df = res_df.groupby("Estate")["_prob"].mean().reset_index()
                estate_df.columns = ["Estate","Avg Risk Score"]
                estate_df = estate_df.sort_values("Avg Risk Score",ascending=False)
                fig_e = px.bar(estate_df,x="Estate",y="Avg Risk Score",
                               title="Average risk score by estate / region",
                               color="Avg Risk Score",color_continuous_scale="RdYlGn_r",
                               text=estate_df["Avg Risk Score"].apply(lambda v:f"{v:.0%}"))
                fig_e.update_traces(textposition="outside")
                fig_e.update_layout(height=380,margin=dict(t=40,b=80),
                                    xaxis_tickangle=-40,coloraxis_showscale=False)
                st.plotly_chart(fig_e,use_container_width=True)

            st.download_button("⬇ Export triage results CSV",
                               res_df.drop(columns=["_prob"],errors="ignore").to_csv(index=False),
                               "msf_triage_results.csv",mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# ══ PROJECT PAGE: Overview Dashboard ════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_page == "📊 Overview Dashboard":
    st.title("📊 Overview Dashboard")
    st.caption("Training population snapshot — 2,000 cases, 2021–2024 | MSF DV Trends Report 2025")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total cases",       f"{len(base_df):,}")
    c2.metric("High Risk (HRF=1)", f"{base_df['High_Risk_Flag'].sum():,} ({base_df['High_Risk_Flag'].mean():.1%})")
    c3.metric("Urgent (IIN=1)",    f"{base_df['Immediate_Intervention_Needed'].sum():,} ({base_df['Immediate_Intervention_Needed'].mean():.1%})")
    c4.metric("Best model AUC",    f"{arts['meta']['hrf_model_name']}")

    st.markdown('<div class="sec-hdr">Risk distribution</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca:
        vc_r = base_df.groupby("Victim_Category")["High_Risk_Flag"].agg(["sum","count"]).reset_index()
        vc_r.columns = ["Category","High Risk","Total"]
        vc_r["Rate"] = (vc_r["High Risk"]/vc_r["Total"]*100).round(1)
        fig1 = px.bar(vc_r,x="Category",y="Rate",color="Rate",
                      color_continuous_scale="RdYlGn_r",text="Rate",
                      title="High Risk rate by victim category (%)")
        fig1.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig1.update_layout(showlegend=False,coloraxis_showscale=False,height=300,margin=dict(t=40,b=20))
        st.plotly_chart(fig1,use_container_width=True)
    with cb:
        yr = base_df.groupby("Year")["High_Risk_Flag"].agg(["sum","count"]).reset_index()
        yr.columns=["Year","High Risk","Total"]; yr["Rate"]=(yr["High Risk"]/yr["Total"]*100).round(1)
        fig2=px.line(yr,x="Year",y="Rate",markers=True,title="High Risk rate 2021–2024 (%)")
        fig2.update_traces(line_color="#ef4444",marker_size=8)
        fig2.update_layout(height=300,margin=dict(t=40,b=20))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="sec-hdr">The 4 risk archetypes</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i,cid in enumerate(range(4)):
        with cols[i]:
            sub = base_df[base_df["Cluster"]==cid]
            cluster_card(cid, n=len(sub), hrf=sub["High_Risk_Flag"].mean())

    st.markdown('<div class="sec-hdr">Abuse type × cluster high-risk heatmap</div>', unsafe_allow_html=True)
    heat = base_df.groupby(["Cluster","Type_of_Abuse"])["High_Risk_Flag"].mean().unstack(fill_value=0).round(3)
    heat.index = [f"C{i}: {CLUSTER_CFG[i]['name']}" for i in heat.index]
    fig3 = px.imshow(heat,text_auto=".0%",color_continuous_scale="RdYlGn_r",
                     title="High Risk rate by cluster × abuse type",aspect="auto")
    fig3.update_layout(height=280,margin=dict(t=40,b=20))
    st.plotly_chart(fig3,use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ PROJECT PAGE: Cluster Intelligence ══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_page == "🗺️ Cluster Intelligence":
    st.title("🗺️ Cluster Intelligence")
    st.caption("Explore the 4 risk archetypes — data profiles, trends, and intervention strategies")

    sel = st.selectbox("Select archetype",
                       [f"Cluster {i}: {CLUSTER_CFG[i]['name']}" for i in range(4)])
    cid  = int(sel.split(":")[0].replace("Cluster","").strip())
    cfg  = CLUSTER_CFG[cid]
    sub  = base_df[base_df["Cluster"]==cid]

    left,right = st.columns([1,1.2])
    with left:
        st.markdown(
            f'<div class="ccard {cfg["css"]}">'
            f'<div class="ctitle" style="font-size:17px">{cfg["emoji"]} {cfg["name"]}</div>'
            f'<span class="badge {cfg["badge"]}">{cfg["badge_txt"]}</span>'
            f'<p style="font-size:13px;margin-top:8px">{cfg["profile"]}</p>'
            f'<p style="font-size:12px;color:#64748b"><b>Key signal:</b> {cfg["signal"]}</p>'
            f'</div>', unsafe_allow_html=True)

        stats = {
            "Cases in cluster"  : f"{len(sub):,} ({len(sub)/len(base_df):.0%} of all cases)",
            "High Risk rate"    : f"{sub['High_Risk_Flag'].mean():.1%}",
            "Need intervention" : f"{sub['Immediate_Intervention_Needed'].mean():.1%}",
            "Financial stress"  : f"{sub['Financial_Stress_Index'].mean():.0%}",
            "Prior contact"     : f"{sub['Prior_Social_Service_Contact'].mean():.0%}",
            "Avg severity"      : f"{sub['Severity_Score'].mean():.2f}",
            "Avg recidivism"    : f"{sub['Recidivism_Risk_Score'].mean():.2f}",
            "Ecological risk"   : f"{sub['Ecological_Risk_Index'].mean():.3f}",
            "Intergenerational" : f"{sub['Intergenerational_Risk_Flag'].mean():.0%}",
        }
        for k,v in stats.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                f'border-bottom:1px solid #f1f5f9;font-size:13px">'
                f'<span style="color:#64748b">{k}</span><span style="font-weight:600">{v}</span></div>',
                unsafe_allow_html=True)

    with right:
        vc_c = sub["Victim_Category"].value_counts()
        fig_d = px.pie(values=vc_c.values,names=vc_c.index,title="Victim category",hole=0.4)
        fig_d.update_layout(height=270,margin=dict(t=40,b=0)); st.plotly_chart(fig_d,use_container_width=True)

        ta_c = sub["Type_of_Abuse"].value_counts()
        fig_ta = px.bar(x=ta_c.values,y=ta_c.index,orientation="h",
                        title="Abuse type",color=ta_c.values,color_continuous_scale="RdPu")
        fig_ta.update_layout(height=230,margin=dict(t=40,b=10,l=10,r=10),
                              coloraxis_showscale=False,showlegend=False)
        st.plotly_chart(fig_ta,use_container_width=True)

    yr_t = sub.groupby("Year").agg(Cases=("High_Risk_Flag","count"),HR=("High_Risk_Flag","sum")).reset_index()
    yr_t["Rate"]=yr_t["HR"]/yr_t["Cases"]
    fig_yr = make_subplots(specs=[[{"secondary_y":True}]])
    fig_yr.add_trace(go.Bar(x=yr_t["Year"],y=yr_t["Cases"],name="Cases",marker_color="#e2e8f0"),secondary_y=False)
    fig_yr.add_trace(go.Scatter(x=yr_t["Year"],y=yr_t["Rate"],name="High Risk rate",mode="lines+markers",
                                line=dict(color=cfg["color"],width=3),marker_size=8),secondary_y=True)
    fig_yr.update_layout(title=f"Trend for {cfg['name']}",height=290,margin=dict(t=40,b=20),
                         legend=dict(orientation="h",yanchor="bottom",y=1.02))
    fig_yr.update_yaxes(title_text="Cases",secondary_y=False)
    fig_yr.update_yaxes(title_text="High Risk rate",secondary_y=True,tickformat=".0%")
    st.plotly_chart(fig_yr,use_container_width=True)

    st.markdown(f'<div class="sec-hdr">Intervention strategy</div>', unsafe_allow_html=True)
    st.caption(f"**Lead agency:** {cfg['who']}  |  **Urgency:** {cfg['urgency']}")
    for i,(title,desc,badge_cls) in enumerate(cfg["interventions"],1):
        st.markdown(
            f'<div class="icard"><span style="color:#94a3b8;font-size:12px">Step {i}</span><br>'
            f'<span class="badge {badge_cls}">{title}</span>'
            f'<span style="font-size:13px;margin-left:10px">{desc}</span></div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ PROJECT PAGE: How the Model Works ════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_page == "📚 How the Model Works":
    st.title("📚 How the Model Works")

    st.markdown("### Three-layer predictive pipeline")
    st.markdown("""
| Layer | Method | Question answered |
|-------|--------|------------------|
| 1 — Archetypes | K-Means k=4 | *What type of family is this?* |
| 2 — High Risk | 5 classifiers compared | *Is this family Tier 2 high risk?* |
| 3 — Urgency | Cascade on HRF=1 cases | *Does this case need same-day intervention?* |
""")

    st.markdown("### Feature engineering (all derived inside the notebook)")
    st.markdown("""
| Feature | Derived from | Formula |
|---------|-------------|---------|
| `Financial_Stress_Index` | CIR + Victim_Category + PSC + HH | 1 if CIR ≥ 1.65 or (Spouse + PSC=1 + HH≥4) |
| `Intergenerational_Risk_Flag` | Victim_Category + PSC + FSI | 1 if Child AND PSC=1 AND FSI=1 |
| `Gender_Power_Imbalance_Flag` | Victim_Category + Sex + Severity | 1 if (Female Spouse + sev≥3) or Female Elderly VA |
| `Recidivism_Risk_Score` | PSC + FSI + IGF + HH | (PSC×2 + FSI + IGF + [HH≥5]) ÷ 5 |
| `Ecological_Risk_Index` | Type_of_Abuse + HH + CIR + FSI + PSC | Bronfenbrenner 4-level weighted composite |
""")

    st.markdown("### Train/Test accuracy gap — how to read the results")
    st.markdown("""
The **Acc_Gap = Train_Accuracy − Test_Accuracy** is the primary generalisation metric.  
A smaller gap means the model learned real patterns, not just memorised the training data.

| Acc_Gap | Interpretation |
|---------|---------------|
| < 0.05 | Excellent — generalises well |
| 0.05–0.10 | Acceptable |
| 0.10–0.15 | Mild overfitting — consider regularisation |
| > 0.15 | Significant overfitting |
""")

    st.markdown("### Triage thresholds")
    st.markdown("""
| Tier | Probability | Action |
|------|------------|--------|
| 🔴 Critical | ≥ 70% | Deploy outreach within 24 hours |
| 🟡 Elevated | 45–69% | Add to watchlist, visit within 1 week |
| 🟢 Routine | < 45% | Standard community support pathway |
""")

    st.markdown("### Files needed to run this application")
    st.code("""
app.py                                      ← this file
final_refined_family_at_risk_ml_dataset_v3.csv  ← dataset
msf_hrf_model_*.pkl                         ← Layer 2 model
msf_iin_model_*.pkl                         ← Layer 3 model
msf_cluster_model.pkl                       ← K-Means model
scaler.pkl                                  ← StandardScaler (prediction)
scaler_cluster.pkl                          ← StandardScaler (clustering)
label_encoders.pkl                          ← categorical encoders
model_meta.pkl                              ← feature lists + cluster info

Run with:  streamlit run app.py
""")

    st.warning("""
**Limitations**
- Trained on synthetic data aligned to MSF DV Trends Report 2025. Validate on real records before deployment.
- A 75% risk score means this *type* of case has historically been 75% likely to be Tier 2 — not a guarantee.
- Social worker professional judgement must override model output in all cases.
- Retrain annually as population patterns shift.
""")

