# FSC Triage Model — Domestic Violence Risk Predictor for Family Service Centres

A machine-learning triage tool that helps social workers at Singapore's Family Service Centres (FSCs) prioritise the families most likely to need urgent domestic violence intervention.

## Background

Family Service Centres are Singapore's front-line social service network. Caseworkers receive intake referrals faster than they can deeply assess each one. The cost of mis-prioritisation is concrete — a delayed intervention for a high-risk family can mean continued harm to children, partners, or vulnerable adults.

The opportunity: historical case data already exists, and the patterns within it can flag the highest-risk new intakes within seconds of referral. This project builds and deploys that triage layer as an end-to-end tool a non-technical caseworker can actually use.

The framing is deliberately operational, not academic. The model exists to change which case a social worker opens first on a Monday morning.

## Data and methodology

**Dataset.** 2,000 anonymised family-at-risk case records (`data/final_refined_family_at_risk_ml_dataset_v3.csv`). 19 features spanning demographics (victim category, age group, sex), context (reporting source, household size, prior social-service contact), and engineered risk indicators (Financial Stress Index, Intergenerational Risk Flag, Gender Power Imbalance Flag, Recidivism Risk Score, Ecological Risk Index). Target: high-risk classification with recommended immediate intervention.

**Privacy.** Raw case data is excluded from the public repository. Code paths reference a synthetic version of the dataset that preserves schema and distribution without real PII.

**Modelling.** Evaluated multiple classification algorithms (logistic regression, XGBoost, LightGBM, CatBoost — all packaged in `requirements.txt`). Selected **logistic regression** as the production model. The reasoning is documented in the notebook: it achieved zero train-test discrepancy, which matters more than marginal accuracy gains when the model decides which families a caseworker sees first. A model that overfits gives false confidence to a triage decision.

**Clustering layer.** A separate K-means clustering model (`models/msf_cluster_model.pkl`) groups cases into intelligence segments — caseworkers see not just a risk score but a behavioural cluster the case resembles. This adds interpretability over a single probability output.

**Production stack.** Trained model artifacts (`models/msf_hrf_model_logistic_regression.pkl`, scaler, label encoders, cluster model) are loaded by a Streamlit app (`app/app.py`) that accepts six core inputs from a social worker, derives engineered features automatically, and returns a risk tier plus cluster context.

## Key findings

- **ROC-AUC of 0.86 and Recall of 0.97** on the held-out test set for the high-risk minority class. The recall optimisation is deliberate — false negatives carry a higher human cost than false positives in triage.
- **Gender Power Imbalance Flag** emerged as one of the strongest single predictors of escalation, ranking above traditional socioeconomic indicators. That finding shifts where caseworkers should focus diagnostic questioning at intake.
- **Six inputs are enough.** A social worker only needs to enter Victim Category, Age Group, Sex, Reporting Source, Household Size, and Prior Social-Service Contact. The remaining engineered features are derived automatically — keeping the intake workflow under a minute.
- **Zero train-test discrepancy in the final model** — chosen over higher-accuracy alternatives because operational reliability beats benchmark performance for a tool that influences case prioritisation.

## How to run

**Live app:** [triage-predictor.streamlit.app](https://triage-predictor.streamlit.app/)

**Local reproduction:**

```bash
git clone https://github.com/manyuongsg/fsc-triage-model
cd fsc-triage-model
pip install -r requirements.txt
streamlit run app/app.py
```

**Re-train from scratch:**

```bash
jupyter notebook notebooks/02_classification_model.ipynb
```

A `.devcontainer/` configuration is included for reproducible environments via VS Code Remote Containers or GitHub Codespaces.

## About the author

I'm Ong Manyu, a junior data analyst based in Singapore. I came to data work via an unusual route — Marine Engineer Officer in the Republic of Singapore Navy, then operations and management at SIPMM, then General Assembly's AI Native programme in early 2026.

The combination is the offer: engineering discipline for debugging complex systems under pressure, operational experience for thinking about how a tool gets used by the people on the front line, and current data and ML tooling for building things that actually run. This project sits squarely in the social-impact direction I'm targeting next — public-sector and NGO data work where analytical rigour meets human consequence.

GitHub: [manyuongsg](https://github.com/manyuongsg)

---
*Drafted from real project files. Review before promoting to `projects/fsc-triage-model/README.md`.*
