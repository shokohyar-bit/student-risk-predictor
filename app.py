# app.py - Student At-Risk Predictor Pro (Unblockable + Cached Model)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
from pathlib import Path

# ====================== Page Config ======================
st.set_page_config(
    page_title="Student At-Risk Predictor Pro",
    page_icon="Student",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .header {font-size: 3rem; color: #d71921; text-align: center; font-weight: bold;}
    .subheader {color: #555; text-align: center;}
    .stButton>button {background-color: #d71921; color: white; border-radius: 8px; padding: 12px 28px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">Student At-Risk Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">AI-Powered Dropout Risk Prediction • Instant Power BI Dashboard</p>', unsafe_allow_html=True)

tab_dashboard, tab_download = st.tabs(["Dashboard", "Download Predictions"])

# ====================== Sidebar ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png", width=100)
    st.title("Upload Data")

    historical_file = st.file_uploader(
        "Historical Terms (any number of past terms)",
        type="csv",
        help="Must contain 'AtRisk Status' column"
    )
    next_term_file = st.file_uploader(
        "Next Term Students (for prediction)",
        type="csv",
        help="Students enrolled in upcoming term"
    )

    use_cached = st.checkbox(
        "Use previously trained model (3–5 sec)",
        value=True,
        help="Uncheck to retrain model from scratch"
    )

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not historical_file or not next_term_file:
            st.error("Please upload both files")
        else:
            st.session_state.run = True
            st.session_state.use_cached = use_cached

# ====================== Model Training with Disk Cache ======================
MODEL_PATH = "cached_model.pkl"

@st.cache_resource(show_spinner=False)
def train_and_save_model(df_hist):
    df = df_hist.copy()
    df["AtRisk"] = df["AtRisk Status"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["AtRisk"])

    def add_features(d):
        d = d.copy()
        for col_num, col_den in [
            ("Subjects Passed", "Subjects Enrolled"),
            ("Subjects Failed", "Subjects Enrolled"),
            ("Subjects Withdrawn", "Subjects Attempted")
        ]:
            if {col_num, col_den}.issubset(d.columns):
                d[f"{col_num.split()[1]}Rate"] = d[col_num] / d[col_den].replace(0, np.nan)
        if "Residential Country" in d.columns:
            d["IsInternational"] = (d["Residential Country"].astype(str).str.lower() != "australia").astype(int)
        return d

    df = add_features(df)
    features = ["Age", "Gender", "Orientation % Complete", "Behaviour Score",
                "PassRate", "FailRate", "WithdrawRate", "IsInternational",
                "FEE-HELP Approved", "FEE-HELP Eligibility"]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["AtRisk"]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), X.select_dtypes("number").columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes("object").columns)
    ])

    model = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("xgb", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),
            random_state=42, n_jobs=-1, verbosity=0
        ))
    ])
    model.fit(X, y)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    return model, features

# ====================== Main Execution ======================
if 'run' in st.session_state and st.session_state.run:
    progress = st.progress(0)
    status = st.empty()

    status.text("Loading historical data...")
    progress.progress(20)
    hist_df = pd.read_csv(historical_file)

    if st.session_state.use_cached and os.path.exists(MODEL_PATH):
        status.text("Loading cached model (super fast)...")
        progress.progress(70)
        cached = joblib.load(MODEL_PATH)
        model, features = cached["model"], cached["features"]
    else:
        status.text("Training new model (first time: ~40 sec)...")
        progress.progress(40)
        model, features = train_and_save_model(hist_df)

    status.text("Predicting next term...")
    progress.progress(90)

    next_df = pd.read_csv(next_term_file)
    next_df = next_df.copy()
    # Add same features
    for col_num, col_den in [
        ("Subjects Passed", "Subjects Enrolled"),
        ("Subjects Failed", "Subjects Enrolled"),
        ("Subjects Withdrawn", "Subjects Attempted")
    ]:
        if {col_num, col_den}.issubset(next_df.columns):
            next_df[f"{col_num.split()[1]}Rate"] = next_df[col_num] / next_df[col_den].replace(0, np.nan)
    if "Residential Country" in next_df.columns:
        next_df["IsInternational"] = (next_df["Residential Country"].astype(str).str.lower() != "australia").astype(int)

    X_next = next_df[features]
    prob = model.predict_proba(X_next)[:, 1]
    next_df["Risk_Probability"] = prob
    next_df["Risk_Status"] = np.where(prob >= 0.4, "At-Risk", "Not At-Risk")
    next_df["Risk_Level"] = pd.cut(prob, [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    next_df["Risk_Probability (%)"] = (prob * 100).round(1)

    st.session_state.results = next_df
    st.session_state.historical = hist_df

    progress.progress(100)
    status.text("Complete!")
    st.success(f"Prediction ready for {len(next_df):,} students")

# ====================== Dashboard & Download ======================
if 'results' in st.session_state:
    df = st.session_state.results
    hist = st.session_state.historical

    with tab_dashboard:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Students", f"{len(df):,}")
        with col2: st.metric("High Risk", len(df[df["Risk_Level"] == "High"]))
        with col3: st.metric("At-Risk Rate", f"{(len(df[df['Risk_Level']=='High'])/len(df)*100):.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            hist_rate = hist.groupby("Term").apply(lambda x: (x["AtRisk Status"].str.lower() == "yes").mean() * 100).round(1)
            pred_rate = (df["Risk_Level"] == "High").mean() * 100
            term_data = pd.DataFrame({
                "Term": list(hist_rate.index) + ["Predicted"],
                "Rate (%)": list(hist_rate.values) + [round(pred_rate, 1)]
            })
            fig1 = px.bar(term_data, x="Term", y="Rate (%)", text="Rate (%)",
                          color="Term", color_discrete_sequence=px.colors.sequential.Reds)
            fig1.update_traces(textposition="outside")
            fig1.update_layout(title="At-Risk Rate by Term")
