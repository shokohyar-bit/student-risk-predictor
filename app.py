# app.py - فقط کپی کن و ذخیره کن
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import zipfile
from datetime import datetime

st.set_page_config(page_title="پیش‌بینی ریسک دانشجویان", layout="wide")
st.title("پیش‌بینی ریسک ترک تحصیل دانشجویان")
st.markdown("**فقط ۵ فایل CSV آپلود کنید → همه چیز خودکار انجام می‌شود**")

st.sidebar.header("راهنمای استفاده")
st.sidebar.success("۱. چهار فایل ترم قبلی (T1 تا T4)\n۲. یک فایل لیست دانشجویان ترم بعد")

t1 = st.file_uploader("ترم ۱ (T1)", type="csv")
t2 = st.file_uploader("ترم ۲ (T2)", type="csv")
t3 = st.file_uploader("ترم ۳ (T3)", type="csv")
t4 = st.file_uploader("ترم ۴ (T4)", type="csv")
t5 = st.file_uploader("لیست دانشجویان ترم بعد (مثل T5 بدون ستون AtRisk)", type="csv")

if st.button("شروع پیش‌بینی ریسک", type="primary"):
    if not all([t1,t2,t3,t4,t5]):
        st.error("لطفاً هر ۵ فایل را آپلود کنید!")
    else:
        with st.spinner("در حال آموزش مدل و پیش‌بینی... (۳۰–۶۰ ثانیه)"):
            # خواندن فایل‌ها
            df1 = pd.read_csv(t1); df1["Term"] = "T1"
            df2 = pd.read_csv(t2); df2["Term"] = "T2"
            df3 = pd.read_csv(t3); df3["Term"] = "T3"
            df4 = pd.read_csv(t4); df4["Term"] = "T4"
            df_next = pd.read_csv(t5)

            df_train = pd.concat([df1,df2,df3,df4], ignore_index=True)
            df_train["AtRisk"] = df_train["AtRisk Status"].astype(str).str.strip().str.lower().map({"yes":1, "no":0})
            df_train = df_train.dropna(subset=["AtRisk"])

            # مهندسی ویژگی
            def feat(df):
                df = df.copy()
                if "Subjects Passed" in df.columns and "Subjects Enrolled" in df.columns:
                    df["PassRate"] = df["Subjects Passed"] / df["Subjects Enrolled"].replace(0,np.nan)
                if "Subjects Failed" in df.columns and "Subjects Enrolled" in df.columns:
                    df["FailRate"] = df["Subjects Failed"] / df["Subjects Enrolled"].replace(0,np.nan)
                if "Subjects Withdrawn" in df.columns and "Subjects Attempted" in df.columns:
                    df["WithdrawRate"] = df["Subjects Withdrawn"] / df["Subjects Attempted"].replace(0,np.nan)
                if "Residential Country" in df.columns:
                    df["IsInternational"] = (df["Residential Country"].astype(str).str.lower() != "australia").astype(int)
                return df

            df_train = feat(df_train)
            df_next = feat(df_next)

            features = ["Age","Gender","Orientation % Complete","Behaviour Score","PassRate","FailRate",
                        "WithdrawRate","IsInternational","FEE-HELP Approved","FEE-HELP Eligibility"]
            features = [f for f in features if f in df_train.columns]

            X_train = df_train[features]
            y_train = df_train["AtRisk"]

            preprocessor = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), X_train.select_dtypes(include="number").columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), X_train.select_dtypes(include="object").columns)
            ])

            model = ImbPipeline([
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("xgb", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, 
                                    scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]),1),
                                    random_state=42, n_jobs=-1))
            ])
            model.fit(X_train, y_train)

            pred_prob = model.predict_proba(df_next[features])[:,1]
            df_next["احتمال_ریسک"] = pred_prob.round(4)
            df_next["وضعیت_پیش‌بینی"] = np.where(pred_prob >= 0.4, "در معرض ریسک", "در خطر نیست")
            df_next["سطح_ریسک"] = pd.cut(pred_prob, [0,0.3,0.6,1.0], labels=["پایین","متوسط","بالا"])

            st.success(f"تمام! پیش‌بینی برای {len(df_next)} دانشجو انجام شد")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("تعداد در ریسک بالا", len(df_next[df_next["سطح_ریسک"]=="بالا"]))
            with col2:
                st.metric("درصد ریسک بالا", f"{(len(df_next[df_next['سطح_ریسک']=='بالا'])/len(df_next)*100):.1f}%")

            st.bar_chart(df_next["سطح_ریسک"].value_counts())

            # دانلود فایل‌ها
            if "Subject Name" in df_next.columns:
                for subj in df_next["Subject Name"].unique():
                    sub_df = df_next[df_next["Subject Name"] == subj]
                    csv = sub_df.to_csv(index=False).encode()
                    st.download_button(f"دانلود {subj}", csv, f"{subj}_ریسک.csv", "text/csv")
            csv_all = df_next.to_csv(index=False).encode()
            st.download_button("دانلود همه دانشجویان", csv_all, "همه_دانشجویان_ریسک.csv", "text/csv")