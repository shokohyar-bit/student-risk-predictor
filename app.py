# app.py - نسخه نهایی بدون خطا
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # این خط درست شد
import os
import joblib

# بقیه کد دقیقاً همون قبلیه — فقط این قسمت بالا رو جایگزین کن
st.set_page_config(page_title="Student At-Risk Predictor Pro", layout="wide", page_icon="Student")

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .header {font-size: 3rem; color: #d71921; text-align: center; font-weight: bold;}
    .stButton>button {background-color: #d71921; color: white; border-radius: 8px; padding: 12px 28px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header">Student At-Risk Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">AI-Powered Dropout Risk Prediction • Private & Secure</p>', unsafe_allow_html=True)

tab_dashboard, tab_download = st.tabs(["Dashboard", "Download Predictions"])

with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png", width=100)
    st.title("Upload Data")
    historical_file = st.file_uploader("Historical Terms (any number)", type="csv")
    next_term_file = st.file_uploader("Next Term Students", type="csv")
    use_cached = st.checkbox("Use cached model (faster)", value=True)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not historical_file or not next_term_file:
            st.error("Please upload both files")
        else:
            st.session_state.run = True
            st.session_state.use_cached = use_cached

# بقیه کد (همون قبلی) — فقط این دو تا فایل رو آپدیت کن
# اگر خواستی بقیه کد رو هم بدم، بگو

# (بقیه کد همون قبلیه — فقط این دو تا فایل رو آپدیت کن)
