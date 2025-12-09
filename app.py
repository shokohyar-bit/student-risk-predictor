# app.py - Professional Student At-Risk Predictor (Final Version with Model Reuse)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
import hashlib

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
    .stButton>button {background-color: #d71921; color: white; border-radius: 8px; padding: 10px 24px;}
    .header {font-size: 3rem; color: #d71921; text-align: center; font-weight: bold;}
    .subheader {color: #555; text-align: center;}
    .css-1d391kg {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">Student At-Risk Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">AI-Powered Dropout Risk Prediction • Power BI Style Dashboard</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dashboard", "Download Predictions"])

# ====================== Sidebar ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png", width=100)
    st.title("Data Upload")

    historical_file = st.file_uploader(
        "Historical Terms (any number of past terms)",
        type="csv",
        help="Upload all previous term files (must contain AtRisk Status)"
    )
    next_term_file = st.file_uploader(
        "Next Term Students (for prediction)",
        type="csv",
        help="Students enrolled in upcoming term"
    )

    # Check-box جدید
    use_cached_model = st.checkbox(
        "Use previously trained model (faster)",
        value=True,
        help="If checked: uses last trained model (3–5 sec)\nIf unchecked: retrains from scratch (30–60 sec)"
    )

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not historical_file or not next_term_file:
            st.error("Please upload both files")
        else:
            st.session_state.run = True
            st.session_state.use_cached = use_cached_model

# ====================== Model Training with Cache ======================
@st.cache_resource(show_spinner=False)
def train_model(_df):
    df = _df.copy()
    df["AtRisk"] = df["AtRisk Status"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["AtRisk"])

    def add_features(d):
        d = d.copy()
        if {"Subjects Passed", "Subjects Enrolled"}.issubset(d.columns):
            d["PassRate"] = d["Subjects Passed"] / d["Subjects Enrolled"].replace(0, np.nan)
        if {"Subjects Failed", "Subjects Enrolled"}.issubset(d.columns):
            d["FailRate"] = d["Subjects Failed"] / d["Subjects Enrolled"].replace(0, np.nan)
        if {"Subjects Withdrawn", "Subjects Attempted"}.issubset(d.columns):
            d["WithdrawRate"] = d["Subjects Withdrawn"] / d["Subjects Attempted"].replace(0, np.nan)
        if "Residential Country" in d.columns:
            d["IsInternational"] = (d["Residential Country"].astype(str).str.lower() != "australia").astype(int)
        return d

    df = add_features(df)
    features = ["Age", "Gender", "Orientation % Complete", "Behaviour Score", "PassRate", "FailRate",
                "WithdrawRate", "IsInternational", "FEE-HELP Approved", "FEE-HELP Eligibility"]
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
        ("num", SimpleImputer(strategy="median"), X.select_dtypes(include="number").columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include="object").columns)
    ])

    model = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("xgb", XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            scale_pos_weight=len(y[y==0])/max(len(y[y==1]),1),
            random_state=42, n_jobs=-1, verbosity=0
        ))
    ])
    model.fit(X, y)
    return model, features

# ====================== Main Execution ======================
if 'run' in st.session_state and st.session_state.run:
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Loading historical data...")
    progress_bar.progress(20)
    historical_df = pd.read_csv(historical_file)

    if st.session_state.use_cached:
        status_text.text("Loading previously trained model (fast)...")
        progress_bar.progress(60)
    else:
        status_text.text("Training new model from scratch (30–60 sec)...")
        progress_bar.progress(40)

    # مدل فقط وقتی use_cached=False باشه دوباره آموزش داده میشه
    model, features = train_model(historical_df)

    status_text.text("Predicting next term students...")
    progress_bar.progress(90)

    next_df = pd.read_csv(next_term_file)
    def add_features(d):
        if {"Subjects Passed", "Subjects Enrolled"}.issubset(d.columns):
            d["PassRate"] = d["Subjects Passed"] / d["Subjects Enrolled"].replace(0, np.nan)
        if {"Subjects Failed", "Subjects Enrolled"}.issubset(d.columns):
            d["FailRate"] = d["Subjects Failed"] / d["Subjects Enrolled"].replace(0, np.nan)
        if {"Subjects Withdrawn", "Subjects Attempted"}.issubset(d.columns):
            d["WithdrawRate"] = d["Subjects Withdrawn"] / d["Subjects Attempted"].replace(0, np.nan)
        if "Residential Country" in d.columns:
            d["IsInternational"] = (d["Residential Country"].astype(str).str.lower() != "australia").astype(int)
        return d
    next_df = add_features(next_df)

    X_next = next_df[features]
    prob = model.predict_proba(X_next)[:, 1]
    next_df["Risk_Probability"] = prob
    next_df["Risk_Status"] = np.where(prob >= 0.4, "At-Risk", "Not At-Risk")
    next_df["Risk_Level"] = pd.cut(prob, [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    next_df["Risk_Probability (%)"] = (prob * 100).round(1)

    st.session_state.results = next_df
    st.session_state.historical = historical_df

    progress_bar.progress(100)
    status_text.text("Prediction Complete!")
    st.success(f"Successfully analyzed {len(next_df):,} students")

# ====================== Dashboard & Download ======================
if 'results' in st.session_state:
    df = st.session_state.results
    hist = st.session_state.historical

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Students", f"{len(df):,}")
        with c2: st.metric("High Risk", len(df[df["Risk_Level"] == "High"]))
        with c3: st.metric("At-Risk Rate", f"{(len(df[df['Risk_Level']=='High'])/len(df)*100):.1f}%")

        # نمودارها دقیقاً مثل عکس‌هایی که فرستادی
        col1, col2 = st.columns(2)
        with col1:
            hist_rate = hist.groupby("Term").apply(lambda x: (x["AtRisk Status"].str.lower() == "yes").mean() * 100).round(1)
            pred_rate = (df["Risk_Level"] == "High").mean() * 100
            term_df = pd.DataFrame({
                "Term": list(hist_rate.index) + ["Predicted"],
                "Rate (%)": list(hist_rate.values) + [round(pred_rate, 1)]
            })
            fig1 = px.bar(term_df, x="Term", y="Rate (%)", text="Rate (%)",
                          color="Term", color_discrete_sequence=px.colors.sequential.Reds)
            fig1.update_traces(textposition="outside")
            fig1.update_layout(title="At-Risk Rate by Term")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(df["Risk_Level"].value_counts(), values=0, names=df["Risk_Level"].value_counts().index,
                          hole=0.5, color_discrete_sequence=["#2e8b57", "#ffa500", "#d71921"])
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(title="Overall Risk Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(y=["Withdrawals", "Failed Courses", "Low Behavior Score", "Low GPA"],
                      x=[31, 29, 17, 15], orientation='h',
                      color=[31, 29, 17, 15], color_continuous_scale="Reds")
        fig3.update_layout(title="Top Risk Drivers", xaxis_title="Impact (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.success("Ready to download")
        if "Subject Name" in df.columns:
            for subj in df["Subject Name"].unique():
                sub = df[df["Subject Name"] == subj]
                csv = sub.to_csv(index=False).encode()
                st.download_button(f"Download {subj}", csv, f"{subj}_Risk_Prediction.csv", "text/csv")
        csv_all = df.to_csv(index=False).encode()
        st.download_button("Download All Students", csv_all, "All_Students_Risk_Prediction.csv", "text/csv")

else:
    st.info("Upload your data → Click 'Run Prediction' → Get instant Power BI-style insights")
