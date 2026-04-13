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
import datetime
import csv

warnings.filterwarnings("ignore")

# This gets the exact folder where app.py lives (the 'app' folder)
APP_DIR = Path(__file__).parent
# This goes up one level to the root of your repository
ROOT_DIR = APP_DIR.parent 
# Define exactly where your models and data live relative to the root
MODEL_DIR = ROOT_DIR / "models" # <-- Change "models" if your folder is named differently
DATA_DIR = ROOT_DIR / "data"
LOG_PATH  = ROOT_DIR / "logs" / "predictions.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Audit log columns ─────────────────────────────────────────────────────────
LOG_COLS = [
    "timestamp_sgt", "model_version", "model_file", "source",
    "victim_category", "age_group", "sex", "reporting_source",
    "household_size", "prior_social_service_contact",
    "fsi", "igf", "gpf", "rss", "eri",
    "cluster_id", "cluster_name", "risk_score_pct", "tier",
]

def _get_model_version():
    """Return (version_tag, filename) of the active HRF model for audit logs."""
    hrf_cands = sorted(MODEL_DIR.glob("msf_hrf_model_*.pkl"))
    if not hrf_cands:
        return "unknown", "unknown"
    fname = hrf_cands[0].name
    mtime = datetime.datetime.fromtimestamp(hrf_cands[0].stat().st_mtime)
    return mtime.strftime("%Y%m%d"), fname

def log_prediction(source, vc, age, sex, src, hs, psc, feats, cid, cluster_name, hrf_prob, t_lbl):
    """Append one prediction record to the CSV audit log.
    Never raises — a log failure must never break the prediction UI.
    """
    try:
        model_version, model_file = _get_model_version()
        now_sgt = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        row = {
            "timestamp_sgt"               : now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version"               : model_version,
            "model_file"                  : model_file,
            "source"                      : source,
            "victim_category"             : vc,
            "age_group"                   : age,
            "sex"                         : sex,
            "reporting_source"            : src,
            "household_size"              : hs,
            "prior_social_service_contact": psc,
            "fsi"                         : feats.get("fsi", ""),
            "igf"                         : feats.get("igf", ""),
            "gpf"                         : feats.get("gpf", ""),
            "rss"                         : feats.get("rss", ""),
            "eri"                         : feats.get("eri", ""),
            "cluster_id"                  : cid,
            "cluster_name"                : cluster_name,
            "risk_score_pct"              : f"{hrf_prob:.4f}",
            "tier"                        : t_lbl,
        }
        write_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass  # Never let logging break the UI

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
    # ── C0: Chronic Spousal Cycle ────────────────────────────────────────────
    # Actual data: PSC=100%, all Spouse, HRF=8%, Recidivism=0.65, ERI=0.66
    # 100% prior contact — chronic repeat pattern known to MSF but not yet statutory.
    0: dict(
        name="Chronic Spousal Cycle", emoji="🔄", color="#f59e0b", css="c0",
        profile="100% prior social service contact, all Spouse victims, chronic repeat pattern. High ERI (0.66) and Recidivism (0.65) but below statutory threshold. Women remain due to financial dependency — risk is accumulating.",
        signal="Chronic repeat pattern. 100% prior contact means the system has been here before — intervention has not broken the cycle yet.",
        who="PSC Protection Officers, PAVE, TRANS-SAFE Centre, PPO unit, FSC long-term caseworkers",
        tiers={
            "critical": dict(
                badge="b-red", badge_txt="CRITICAL",
                urgency="CRITICAL — assess within 24 hours",
                interventions=[
                    ("Immediate home visit",    "Dispatch outreach worker for same-day safety assessment — do not delay despite low prior contact",         "b-red"),
                    ("Emergency safety plan",   "Document safe word, emergency contacts, and evacuation plan with victim on first visit",                   "b-red"),
                    ("PPO/MPO application",     "Assist victim to file for Personal Protection Order or Mandatory Counselling Order immediately",           "b-amber"),
                    ("Crisis service referral", "Refer to PAVE or TRANS-SAFE Centre crisis line; link to ComCare emergency assistance",                    "b-purple"),
                    ("Case registration",       "Register with FSC immediately — establish the ongoing relationship this family has never had",             "b-blue"),
                ],
            ),
            "elevated": dict(
                badge="b-amber", badge_txt="Priority",
                urgency="Priority — act within 1 week",
                interventions=[
                    ("Priority home visit",     "Schedule home visit within 5 working days; conduct structured safety screening",                          "b-amber"),
                    ("Financial safety net",    "Link to ComCare, Workfare, utility assistance to reduce economic dependency vulnerability",                "b-green"),
                    ("PPO awareness",           "Ensure victim knows legal rights under Women's Charter and how to apply for PPO",                          "b-purple"),
                    ("Service warm handoff",    "Facilitate direct referral to FSC counsellor — do not leave with a leaflet only",                         "b-blue"),
                ],
            ),
            "routine": dict(
                badge="b-green", badge_txt="Preventive",
                urgency="Preventive — act within 2–4 weeks",
                interventions=[
                    ("Grassroots outreach",     "Engage through CCs and RCs to build trust — this family has never engaged services before",               "b-blue"),
                    ("Financial screening",     "Link to ComCare, Workfare, utility assistance",                                                            "b-green"),
                    ("PPO awareness",           "Ensure female spouses know their legal rights under the Women's Charter",                                  "b-purple"),
                    ("GP sensitisation",        "Alert family GP to screen for family violence indicators at routine visits",                               "b-grey"),
                ],
            ),
        },
    ),
    # ── C1: Silent Pressure Cooker ───────────────────────────────────────────
    # Actual data: PSC=0%, all Spouse, HRF=5%, Recidivism=0.24, ERI=0.42
    # Zero prior contact — these families have NEVER reached services.
    # Lowest HRF and Recidivism — risk is latent, not yet surfaced.
    1: dict(
        name="Silent Pressure Cooker", emoji="🫧", color="#ef4444", css="c1",
        profile="Zero prior social service contact, all Spouse victims, lowest HRF (5%) and Recidivism (0.24). Risk is hidden — these families have never reached services. Danger accumulates silently beneath the surface.",
        signal="No prior contact means no early warning system. This is a first-contact window — engage before crisis escalates.",
        who="Community Development Councils, RCs, GP clinics, FSC outreach workers, polyclinics",
        tiers={
            "critical": dict(
                badge="b-red", badge_txt="CRITICAL",
                urgency="CRITICAL — statutory response same day",
                interventions=[
                    ("CYPA mandatory report",   "File immediate report under Children and Young Persons Act — do not wait for confirmation",                "b-red"),
                    ("MSF CPS referral",        "Escalate to MSF Child Protective Service for urgent investigation and safety determination",               "b-red"),
                    ("Emergency placement",     "Assess suitability of temporary foster or kinship care if home environment is unsafe",                    "b-red"),
                    ("School immediate alert",  "Notify school principal and AED; increase daily welfare checks and document all observations",            "b-amber"),
                    ("Multi-agency meeting",    "Convene Child Protection Conference within 72 hours with school, MSF, and medical if applicable",         "b-purple"),
                ],
            ),
            "elevated": dict(
                badge="b-red", badge_txt="High Alert",
                urgency="High alert — triage within 48 hours",
                interventions=[
                    ("School monitoring",       "Equip AEDs and counsellors to detect unexplained injuries and absenteeism — first-contact family",        "b-red"),
                    ("Childcare alerts",        "Train childcare centre staff on physical abuse indicators for 0–6 year olds",                             "b-red"),
                    ("Priority casework",       "Assign dedicated caseworker; schedule home visit within 48 hours — this is the critical window",          "b-amber"),
                    ("Reporter training",       "Refresh CYPA mandatory reporting obligations for teachers and GPs in high-incidence estates",              "b-blue"),
                ],
            ),
            "routine": dict(
                badge="b-amber", badge_txt="Monitor",
                urgency="Monitor — act within 2 weeks",
                interventions=[
                    ("School monitoring",       "Equip AEDs and counsellors to detect unexplained injuries and absenteeism",                               "b-amber"),
                    ("Estate campaign",         "Parenting workshops and KidSTART in high-incidence community estates",                                    "b-blue"),
                    ("Reporter training",       "Refresh CYPA mandatory reporting obligations for teachers and GPs",                                       "b-blue"),
                    ("Hotspot mapping",         "Flag estate in community risk register; increase proactive outreach frequency in district",                "b-grey"),
                ],
            ),
        },
    ),
    # ── C2: Chronic Family Cycle ─────────────────────────────────────────────
    # Actual data: PSC=100%, all Child, HRF=66%, Recidivism=0.73, FSI=0.63
    # Highest HRF and Recidivism — most urgent cluster.
    # 100% prior contact — prior intervention has NOT broken the cycle.
    2: dict(
        name="Chronic Family Cycle", emoji="🔁", color="#8b5cf6", css="c2",
        profile="100% prior social service contact, all Child victims, highest HRF (66%) and Recidivism (0.73). Financial stress is high (FSI=0.63). Past intervention has not resolved the root cause — violence is entrenched and intergenerational.",
        signal="Prior intervention failed. Highest HRF rate across all clusters. Statutory escalation is likely required — community management has already been attempted.",
        who="MSF Child Protective Service, CPSCs, Foster Care caseworkers, School DSAs, MSF PSV",
        tiers={
            "critical": dict(
                badge="b-red", badge_txt="CRITICAL",
                urgency="CRITICAL — safety plan within 24 hours",
                interventions=[
                    ("DVERT emergency referral","Fast-track to Domestic Violence Emergency Response Team — activate immediately for police cases",           "b-red"),
                    ("Emergency PPO",           "Assist victim to file for emergency PPO and expedited court hearing same day if possible",                  "b-red"),
                    ("Safe house assessment",   "Assess suitability for refuge placement at PAVE or TRANS-SAFE Centre if home is unsafe",                   "b-red"),
                    ("Crisis safety plan",      "Document safe word, escape route, emergency contacts, and financial access before worker leaves",           "b-amber"),
                    ("Perpetrator alert",       "Notify police for immediate perpetrator interview; assess Mandatory Counselling Order eligibility",          "b-purple"),
                ],
            ),
            "elevated": dict(
                badge="b-purple", badge_txt="Priority",
                urgency="Priority — structured follow-up within 1 week",
                interventions=[
                    ("Dedicated caseworker",    "Assign dedicated case worker — minimum 12-month engagement plan given chronic repeat pattern",             "b-purple"),
                    ("DVERT referral",          "Fast-track to Domestic Violence Emergency Response Team for police cases",                                  "b-red"),
                    ("Safety planning",         "Ensure victim has PPO, safe word, and emergency contacts documented",                                       "b-amber"),
                    ("Financial independence",  "Connect to WSG employment support, skills training, childcare subsidies to reduce dependency",             "b-green"),
                ],
            ),
            "routine": dict(
                badge="b-purple", badge_txt="Chronic",
                urgency="Chronic management — structured follow-up within 2 weeks",
                interventions=[
                    ("Long-term counselling",   "Assign dedicated case worker — minimum 12-month engagement plan",                                          "b-purple"),
                    ("Financial independence",  "Connect to WSG employment support, skills training, childcare subsidies",                                   "b-green"),
                    ("Safety planning",         "Ensure victim has PPO, safe word, and emergency contacts documented",                                       "b-amber"),
                    ("Perpetrator programme",   "Mandatory Counselling Order assessment for repeat incidents; anger management referral",                    "b-blue"),
                ],
            ),
        },
    ),
    # ── C3: Ecological Hotspot Child ────────────────────────────────────────
    # Actual data: PSC≈0%, all Child, HRF=64%, Recidivism=0.16, ERI=0.42
    # Almost no prior contact — these families are UNDETECTED.
    # High HRF despite no history — risk is driven by environment, not case history.
    3: dict(
        name="Ecological Hotspot Child", emoji="🏘️", color="#ec4899", css="c3",
        profile="Almost zero prior social service contact (≈9%), all Child victims, high HRF (64%) despite no case history. Financial stress is high (FSI=0.60). Risk is environmental — these families are undetected. First contact is the critical intervention window.",
        signal="High risk, no prior contact. Risk is driven by community environment (high CIR), not accumulated case history. The 48-hour window after first detection is critical.",
        who="School counsellors, Allied Educators, MOE-FSC liaisons, KidSTART, childcare centres",
        tiers={
            "critical": dict(
                badge="b-red", badge_txt="CRITICAL",
                urgency="CRITICAL — escalate within 24 hours",
                interventions=[
                    ("Statutory review",        "Escalate immediately to MSF Child Protective Service for CYPA investigation — prior intervention has not resolved root cause",  "b-red"),
                    ("Emergency placement",     "Initiate foster/kinship care placement assessment if home remains unsafe for children",                     "b-red"),
                    ("Multi-agency crisis",     "Convene Child Protection Conference within 24 hours; include school, medical, police, and FSC",             "b-red"),
                    ("School welfare alert",    "Flag all children in household for immediate daily welfare checks; report any injuries same day",           "b-amber"),
                    ("Intensive therapy order", "Apply for court-ordered Multisystemic Therapy (MST) or Family Preservation Services",                      "b-purple"),
                ],
            ),
            "elevated": dict(
                badge="b-red", badge_txt="URGENT",
                urgency="URGENT — escalate within 24 hours",
                interventions=[
                    ("Statutory review",        "Escalate to MSF Child Protective Service for CYPA investigation and consider supervision order",           "b-red"),
                    ("Out-of-home assessment",  "Evaluate foster/kinship care placement — prior contact means community management has been tried",         "b-red"),
                    ("Intensive therapy",       "Multisystemic Therapy (MST) or Family Preservation Services",                                              "b-amber"),
                    ("School welfare check",    "Weekly welfare checks via school; flag absenteeism and unexplained injuries immediately",                  "b-purple"),
                ],
            ),
            "routine": dict(
                badge="b-amber", badge_txt="Priority",
                urgency="Priority — structured engagement within 1 week",
                interventions=[
                    ("Intensive therapy",       "Multisystemic Therapy (MST) or Family Preservation Services",                                              "b-amber"),
                    ("Parenting programme",     "Triple P or Circle of Security — mandatory if case remains in the home",                                   "b-blue"),
                    ("School welfare check",    "Weekly welfare checks via school; flag absenteeism immediately",                                            "b-purple"),
                    ("Case conference",         "Quarterly multi-agency case conference to review progress and escalation triggers",                         "b-grey"),
                ],
            ),
        },
    ),
}

def get_tier_cfg(cid, prob):
    """Return merged dict of base cluster config + tier-specific urgency/interventions/badge."""
    base = CLUSTER_CFG[cid]
    if prob >= 0.70:   key = "critical"
    elif prob >= 0.45: key = "elevated"
    else:              key = "routine"
    return {**base, **base["tiers"][key]}


ABUSE_PROXY = {"Physical Abuse":0.7,"Sexual Abuse":0.8,"Neglect":0.5,"Emotional & Psychological Abuse":0.4,"Self-Neglect":0.5}

AGE_ORDINAL = {"0-6 years":0,"7-12 years":1,"13-16 years":2,"17-18 years":3,
               "18-29 years":4,"30-39 years":5,"40-49 years":6,
               "50-64 years":7,"65+ years":8}

# Reporting source stress weights — Step 3.1 (FSI component)
REPORT_STRESS = {
    "Police": 0.4, "Hospital": 0.4, "Hotline": 0.2,
    "Community": 0.0, "School": 0.0, "Social Service": 0.0,
}
# Mesosystem Isolation proxy — Step 3.6 (ERI component) & Step 3.7 (SIC)
MESO_MAP = {
    "Police": 1.0, "Hospital": 1.0, "Hotline": 0.6,
    "School": 0.0, "Social Service": 0.0, "Community": 0.0,
}
CIR_MAX = 2.8  # max(Community_Incidence_Rate) from training data — used to normalise FSI_cir

SINGAPORE_ESTATES = [
    "Ang Mo Kio","Bedok","Bishan","Bukit Batok","Bukit Merah","Bukit Panjang",
    "Bukit Timah","Central Area","Choa Chu Kang","Clementi","Geylang","Hougang",
    "Jurong East","Jurong West","Kallang","Marine Parade","Novena","Pasir Ris",
    "Punggol","Queenstown","Sembawang","Sengkang","Serangoon","Tampines",
    "Toa Payoh","Woodlands","Yishun",
]

# ── Batch input normalization maps (case-insensitive → exact dataset value) ──
_VC_MAP  = {v.lower(): v for v in ["Child", "Spouse", "Non-Elderly VA", "Elderly VA"]}
_AGE_MAP = {a.lower(): a for a in ["0-6 years", "7-12 years", "13-16 years", "17-18 years",
                                    "18-29 years", "30-39 years", "40-49 years",
                                    "50-64 years", "65+ years"]}
_SEX_MAP = {"female": "Female", "male": "Male", "f": "Female", "m": "Male"}
_SRC_MAP = {s.lower(): s for s in ["Community", "Hospital", "Hotline", "Police",
                                    "School", "Social Service"]}

def _norm_cat(val, mapping, default):
    """Case-insensitive + whitespace-stripped lookup; returns default if not found."""
    return mapping.get(str(val).strip().lower(), default)

def _norm_psc(val):
    """Accept 1/0, True/False, Yes/No, Y/N."""
    return 1 if str(val).strip().lower() in {"1", "yes", "true", "y"} else 0

def _norm_hs(val, lo=2, hi=6):
    """Parse household size robustly; clamp to [lo, hi]."""
    try:
        return max(lo, min(hi, int(float(val))))
    except (ValueError, TypeError):
        return 4


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

    if not hrf_cands: st.error(f"Missing msf_hrf_model_*.pkl in {MODEL_DIR}"); st.stop()

    return {
        "hrf":     joblib.load(hrf_cands[0]),
        "km":      joblib.load(MODEL_DIR / "msf_cluster_model.pkl"),
        "sc_cl":   joblib.load(MODEL_DIR / "scaler_cluster.pkl"),
        "sc_mod":  joblib.load(MODEL_DIR / "scaler.pkl"),
        "meta":    joblib.load(MODEL_DIR / "model_meta.pkl"),
        "le":      joblib.load(MODEL_DIR / "label_encoders.pkl"),
    }

arts = load_arts()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_base():
    # Use DATA_DIR instead of hardcoded "../data"
    cands = list(DATA_DIR.glob("*v3*.csv")) + list(DATA_DIR.glob("final_refined*.csv"))
    if not cands:
        st.error(f"Cannot find the dataset CSV. Looked in: {DATA_DIR}")
        st.stop()
        
    df = pd.read_csv(sorted(cands)[-1], low_memory=False)

    # Re-derive engineered features from raw columns — formulas must match notebook Steps 3.1–3.7
    # FSI: 3-component weighted index (Step 3.1)
    fsi_cir     = df["Community_Incidence_Rate"] / CIR_MAX
    fsi_report  = df["Reporting_Source"].map(REPORT_STRESS).fillna(0.0)
    fsi_density = ((df["Household_Size"] - 2) / 4).clip(0, 1)
    df["Financial_Stress_Index"] = (0.50*fsi_cir + 0.30*fsi_report + 0.20*fsi_density).clip(0,1).round(3)
    fsi = df["Financial_Stress_Index"]

    # IGF: Child + household size >= 3 + prior contact (Step 3.3)
    igf = ((df["Victim_Category"]=="Child") &
           (df["Household_Size"]>=3) &
           (df["Prior_Social_Service_Contact"]==1)).astype(int)
    df["Intergenerational_Risk_Flag"] = igf

    # GPF: Female Spouse only (Step 3.4)
    gpf = ((df["Sex"]=="Female") & (df["Victim_Category"]=="Spouse")).astype(int)
    df["Gender_Power_Imbalance_Flag"] = gpf

    # RSS: PSC*2 + FSI + IGF + GPF, divided by 5 (Step 3.5)
    df["Recidivism_Risk_Score"] = (
        (df["Prior_Social_Service_Contact"]*2 + fsi + igf + gpf) / 5.0
    ).clip(0,1).round(3)

    # ERI: 5-layer Bronfenbrenner composite (Step 3.6)
    meso = df["Reporting_Source"].map(MESO_MAP).fillna(0.0)
    df["Ecological_Risk_Index"] = (
        df["Prior_Social_Service_Contact"]                 * 0.25 +
        (df["Community_Incidence_Rate"] / 3.0).clip(0, 1) * 0.25 +
        ((df["Household_Size"] - 2) / 4.0).clip(0, 1)     * 0.20 +
        meso                                               * 0.20 +
        fsi_report                                         * 0.10
    ).round(3)

    # Stress_Isolation_Cross: FSI × Mesosystem Isolation (Step 3.7)
    df["Stress_Isolation_Cross"] = (fsi * meso).round(3)

    # Assign cluster via saved model
    meta  = arts["meta"]; le_cl = meta["le_cl"]
    df_cl = df.copy()
    df_cl["Age_Group_ord"] = df_cl["Age_Group"].map(AGE_ORDINAL)  # ordinal encoding matches notebook
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
    """Derive all engineered features from the 6 user inputs + defaults.
    Formulas mirror notebook Steps 3.1–3.7 exactly."""
    # FSI: 3-component weighted index (Step 3.1)
    fsi_cir     = cir / CIR_MAX
    fsi_report  = REPORT_STRESS.get(src, 0.0)
    fsi_density = min((hs - 2) / 4.0, 1.0)
    fsi = round(min(0.50*fsi_cir + 0.30*fsi_report + 0.20*fsi_density, 1.0), 3)
    # IGF: Child + hs>=3 + prior contact (Step 3.3)
    igf = int(vc == "Child" and hs >= 3 and psc == 1)
    # GPF: Female Spouse only (Step 3.4)
    gpf = int(sex == "Female" and vc == "Spouse")
    # RSS: PSC*2 + FSI + IGF + GPF, divided by 5 (Step 3.5)
    rss = round(min((psc*2 + fsi + igf + gpf) / 5.0, 1.0), 3)
    # Mesosystem Isolation intermediate (Step 3.6)
    meso = MESO_MAP.get(src, 0.0)
    # ERI: 5-layer Bronfenbrenner composite (Step 3.6)
    eri = round(
        psc                    * 0.25 +
        min(cir / 3.0, 1)      * 0.25 +
        min((hs - 2) / 4.0, 1) * 0.20 +
        meso                   * 0.20 +
        fsi_report             * 0.10, 3)
    # Stress_Isolation_Cross: FSI x Mesosystem (Step 3.7)
    sic = round(fsi * meso, 3)
    return dict(fsi=fsi, igf=igf, gpf=gpf, rss=rss, eri=eri, sic=sic,
                age=age, sev=sev, cir=cir)

def get_cluster(vc, age, hs, psc, fsi, igf, rss, eri, sev=3.0, cir=2.4):
    meta = arts["meta"]; le_cl = meta["le_cl"]
    cl_row = {}
    for col in meta["cluster_cat_features"]:
        val = {"Victim_Category":vc,"Type_of_Abuse":"Physical Abuse","Age_Group":age}.get(col,"Child")
        cl_row[f"{col}_cl_enc"] = int(le_cl[col].transform([val])[0]) if val in le_cl[col].classes_ else 0
    num_vals = {"Financial_Stress_Index":fsi,"Household_Size":hs,"Prior_Social_Service_Contact":psc,
                "Community_Incidence_Rate":cir,"Severity_Score":sev,"Recidivism_Risk_Score":rss,
                "Ecological_Risk_Index":eri,"Intergenerational_Risk_Flag":igf,
                "Age_Group_ord":AGE_ORDINAL.get(age, 4)}
    for col in meta["cluster_num_features"]: cl_row[col]=num_vals.get(col,0)
    X = np.array([[cl_row.get(f,0) for f in meta["cluster_feature_cols"]]],dtype=float)
    return int(arts["km"].predict(arts["sc_cl"].transform(X))[0])

def predict(vc, age, sex, src, hs, psc, fsi, igf, gpf, rss, eri, sic, sev=3.0, cir=2.4, year=2024):
    meta = arts["meta"]; le = arts["le"]
    feat_row = {}
    for col in meta["feature_cols"]:
        if col == "Age_Group": feat_row[col]=AGE_ORDINAL.get(age,4)
        elif col in le:
            val={"Victim_Category":vc,"Sex":sex,"Reporting_Source":src}.get(col,le[col].classes_[0])
            feat_row[col]=int(le[col].transform([val])[0]) if val in le[col].classes_ else 0
        else:
            feat_row[col]={"Year":year,"Household_Size":hs,"Prior_Social_Service_Contact":psc,
                           "Severity_Score":sev,"Community_Incidence_Rate":cir,
                           "Financial_Stress_Index":fsi,"Intergenerational_Risk_Flag":igf,
                           "Gender_Power_Imbalance_Flag":gpf,"Recidivism_Risk_Score":rss,
                           "Ecological_Risk_Index":eri,"Stress_Isolation_Cross":sic}.get(col,0)
    X = np.array([[feat_row[c] for c in meta["feature_cols"]]],dtype=float)
    Xs = arts["sc_mod"].transform(X)
    hrf_prob = float(arts["hrf"].predict_proba(Xs)[0,1])
    return hrf_prob

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
    cfg = {**CLUSTER_CFG[cid], **CLUSTER_CFG[cid]["tiers"]["routine"]}
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
        "user_nav", ["📋 Case Triage", "📋 Audit Log"],
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
    st.caption(f"Best model: {arts['meta']['hrf_model_name']}")
    st.caption("K-Means k=4 · 80/20 temporal split")

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
        vc  = c1.selectbox("Victim category *", ["Child","Spouse","Non-Elderly VA","Elderly VA",])
        sex = c2.selectbox("Sex *", ["Female","Male"])
        src = c3.selectbox("Reporting source *",
                           ["Police","School","Hospital","Social Service","Hotline","Community"])

        age_opts = {"Child":["0-6 years","7-12 years","13-16 years","17-18 years"],
                    "Spouse":["18-29 years","30-39 years","40-49 years","50-64 years"],
                    "Non-Elderly VA":["18-29 years","30-39 years","40-49 years","50-64 years"],       "Elderly VA":["65+ years"]}
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
            hrf_prob = predict(vc, feats["age"], sex, src, hs, psc,
                               feats["fsi"], feats["igf"], feats["gpf"],
                               feats["rss"], feats["eri"], feats["sic"])
            t_lbl, t_col, t_badge = tier(hrf_prob)
            cfg = get_tier_cfg(cid, hrf_prob)
            # ── Audit log ─────────────────────────────────────────────────────────────────────
            log_prediction("single", vc, age, sex, src, hs, psc,
                           feats, cid, cfg["name"], hrf_prob, t_lbl)

            st.divider()

            # ── Score + Archetype row ─────────────────────────────────────────
            col_score, col_arch = st.columns([1, 1.4])

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
            raw.columns = raw.columns.str.strip()  # strip whitespace from column names
            if "Family_ID" not in raw.columns:
                raw.insert(0,"Family_ID",[f"FAM-{i:04d}" for i in range(1,len(raw)+1)])
            if "Estate" not in raw.columns:
                raw["Estate"] = np.random.choice(SINGAPORE_ESTATES, size=len(raw))

            results = []
            progress = st.progress(0, text="Scoring cases...")
            for idx, row in raw.iterrows():
                r = row.to_dict()
                vc_  = _norm_cat(r.get("Victim_Category","Child"),  _VC_MAP,  "Child")
                age_ = _norm_cat(r.get("Age_Group","0-6 years"),    _AGE_MAP, "0-6 years")
                sex_ = _norm_cat(r.get("Sex","Female"),             _SEX_MAP, "Female")
                src_ = _norm_cat(r.get("Reporting_Source","School"),_SRC_MAP, "School")
                hs_  = _norm_hs(r.get("Household_Size", 4))
                psc_ = _norm_psc(r.get("Prior_Social_Service_Contact", 0))
                try:
                    f_   = derive_features(vc_,age_,sex_,src_,hs_,psc_)
                    cid_ = get_cluster(vc_,f_["age"],hs_,psc_,f_["fsi"],f_["igf"],f_["rss"],f_["eri"])
                    hp= predict(vc_,f_["age"],sex_,src_,hs_,psc_,
                                f_["fsi"],f_["igf"],f_["gpf"],f_["rss"],f_["eri"],f_["sic"])
                    tlbl,_,_= tier(hp)
                    cfg_  = get_tier_cfg(cid_, hp)
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
                        "Urgency"        : cfg_["urgency"],
                    })
                    # ── Audit log ────────────────────────────────────────────────────
                    log_prediction("batch", vc_, age_, sex_, src_, hs_, psc_,
                                   f_, cid_, cfg_["name"], hp, tlbl)
                except Exception as e:
                    results.append({"Family_ID":r.get("Family_ID","?"),"Tier":"⚠ Error","_prob":0,
                                    "Risk_Score":"—","Archetype":str(e)[:40],"Urgency":"—",
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
                            "Risk_Score","Tier","Archetype","Urgency"]
            st.dataframe(res_df[[c for c in display_cols if c in res_df.columns]],
                         hide_index=True, use_container_width=True)

            # ── Case detail inspector ─────────────────────────────────────────
            st.markdown('<div class="sec-hdr">Case detail inspector</div>', unsafe_allow_html=True)
            sel_id = st.selectbox("Select a Family ID to view full detail:", res_df["Family_ID"].tolist())
            sel = res_df[res_df["Family_ID"]==sel_id].iloc[0]
            sel_prob = sel["_prob"]
            t_lbl2,t_col2,_ = tier(sel_prob)
            _cid2 = next((c for c in CLUSTER_CFG if CLUSTER_CFG[c]["name"] in sel.get("Archetype","")), 0)
            cfg2 = get_tier_cfg(_cid2, sel_prob)

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
# ══ USER PAGE: Audit Log ═════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif active_page == "📋 Audit Log":
    st.title("📋 Prediction Audit Log")
    st.caption(
        "Every prediction made in this application is automatically logged with a "
        "Singapore timestamp and model version. Use this log to review past assessments, "
        "monitor model usage, and support accountability."
    )

    if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
        log_df = pd.read_csv(LOG_PATH)

        # ── Summary KPIs ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total predictions", len(log_df))
        m2.metric("Single-case",       (log_df["source"] == "single").sum())
        m3.metric("Batch",             (log_df["source"] == "batch").sum())
        m4.metric("Model versions",    log_df["model_version"].nunique())

        # ── Filters ───────────────────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            src_filter  = st.multiselect("Source", ["single", "batch"],
                                          default=["single", "batch"],
                                          key="al_src")
        with col_f2:
            tier_filter = st.multiselect("Tier", log_df["tier"].unique().tolist(),
                                          default=log_df["tier"].unique().tolist(),
                                          key="al_tier")
        with col_f3:
            n_rows = st.slider("Rows to display", 10, min(500, len(log_df)), 50,
                               key="al_rows")

        # ── Filtered table (most recent first) ────────────────────────────
        filtered = log_df[
            log_df["source"].isin(src_filter) &
            log_df["tier"].isin(tier_filter)
        ].tail(n_rows).iloc[::-1].reset_index(drop=True)
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────
        st.download_button(
            "⬇ Download full audit log (CSV)",
            log_df.to_csv(index=False),
            file_name=f"msf_audit_log_{datetime.datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions logged yet. Run a case assessment to begin.")

elif active_page == "📊 Overview Dashboard":
    st.title("📊 Overview Dashboard")
    st.caption("Training population snapshot — 2,000 cases, 2021–2024 | MSF DV Trends Report 2025")

    c1,c2,c3 = st.columns(3)
    c1.metric("Total cases",       f"{len(base_df):,}")
    c2.metric("High Risk (HRF=1)", f"{base_df['High_Risk_Flag'].sum():,} ({base_df['High_Risk_Flag'].mean():.1%})")
    c3.metric("Best model",        arts['meta']['hrf_model_name'])

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
            f'<p style="font-size:13px;margin-top:8px">{cfg["profile"]}</p>'
            f'<p style="font-size:12px;color:#64748b"><b>Key signal:</b> {cfg["signal"]}</p>'
            f'</div>', unsafe_allow_html=True)

        stats = {
            "Cases in cluster"  : f"{len(sub):,} ({len(sub)/len(base_df):.0%} of all cases)",
            "High Risk rate"    : f"{sub['High_Risk_Flag'].mean():.1%}",
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

    st.markdown(f'<div class="sec-hdr">Intervention strategy by risk level</div>', unsafe_allow_html=True)
    st.caption(f"**Lead agency:** {cfg['who']}")
    for _tk, _tl in [("critical","🔴 Critical (≥70%)"),("elevated","🟡 Elevated (45–69%)"),("routine","🟢 Routine (<45%)")]:
        _td = cfg["tiers"][_tk]
        with st.expander(f"{_tl}  —  {_td['urgency']}", expanded=(_tk=="critical")):
            for i,(title,desc,badge_cls) in enumerate(_td["interventions"],1):
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

    st.markdown("### Two-layer predictive pipeline")
    st.markdown("""
| Layer | Method | Question answered |
|-------|--------|------------------|
| 1 — Archetypes | K-Means k=4 | *What type of family is this?* |
| 2 — High Risk | 5 classifiers compared | *Is this family Tier 2 high risk?* |

Layer 1 and Layer 2 are **independent** — the cluster label is not passed as a feature to the classifier. K-Means is fitted on the training split only to prevent centroid leakage into evaluation.

### The 4 risk archetypes

| Cluster | Archetype | Defining signal | Recommended intervention |
|---------|-----------|-----------------|--------------------------|
| C0 | Chronic Spousal Cycle | 100% prior contact, Spouse victims, chronic repeat pattern, high ERI (0.66), HRF=8%, Recidivism=0.65 | Long-term: DVERT, mandatory counselling, financial independence support |
| C1 | Silent Pressure Cooker | 0% prior contact, Spouse victims, lowest HRF (5%) and Recidivism (0.24) — latent risk not yet surfaced | Preventive: grassroots outreach, PPO awareness, proactive GP screening |
| C2 | Chronic Family Cycle | 100% prior contact, Child victims, highest HRF (66%) and Recidivism (0.73) — prior intervention has not broken the cycle | Urgent: CYPA escalation, intensive family therapy, out-of-home care assessment |
| C3 | Ecological Hotspot Child | ≈0% prior contact, Child victims, high HRF (64%) despite no history — first contact is the critical window | High alert: school monitoring, childcare alerts, estate-level campaigns — respond within 48 hours |

> **Layer 1 and Layer 2 are fully independent.** K-Means was fitted on training rows only to prevent centroid leakage.
""")

    st.markdown("### Feature engineering (all derived inside the notebook)")
    st.markdown("""
| Feature | Derived from | Formula / logic |
|---------|-------------|-----------------|
| `Financial_Stress_Index` (FSI) | CIR + Reporting_Source + HH | Continuous 0–1: 50% CIR + 30% reporting escalation + 20% household density |
| `Intergenerational_Risk_Flag` (IGF) | Victim_Category + PSC + HH | 1 if **Child** AND PSC=1 AND HH≥3 |
| `Gender_Power_Imbalance_Flag` (GPIF) | Victim_Category + Sex | 1 if Female AND Victim_Category=Spouse |
| `Recidivism_Risk_Score` (RRS) | PSC + FSI + IGF + GPIF | (PSC×2 + FSI + IGF + GPIF) ÷ 5 |
| `Ecological_Risk_Index` (ERI) | PSC + CIR + HH + Reporting_Source | Bronfenbrenner 5-layer composite: PSC 25% + CIR 25% + HH 20% + Mesosystem 20% + escalation 10% |
| `Stress_Isolation_Cross` | FSI × Mesosystem_Isolation | Interaction term — financial stress × social isolation (Exchange/Social Control Theory) |
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
msf_hrf_model_*.pkl                         ← Layer 2 classifier (High Risk Flag)
msf_cluster_model.pkl                       ← K-Means k=4 model (Layer 1)
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


