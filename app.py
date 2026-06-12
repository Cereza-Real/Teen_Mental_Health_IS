import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE INITIALIZATION ---
st.set_page_config(
    page_title="Teen Mental Health Diagnostic Suite",
    page_icon="🧠",
    layout="wide"
)

# Define paths matching your exported model artifacts
MODEL_PATH = 'best_mental_health_model.pkl'
CM_PATH = 'confusion_matrix.png'
FI_PATH = 'feature_importance.png'

@st.cache_resource
def load_trained_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_trained_pipeline()

# --- APP LAYOUT ---
st.title("🧠 Teen Mental Health Diagnostic Triage Prototype")
st.markdown("""
**Group Members:** MUHAMMAD ILMAN ABQAARY, ADAM, MOHAMMAD HARITH, MUHAMMAD AFIQ, SHAZWAN HAQ, MUHAMMAD MUHAIMIN
""")
st.markdown("---")

# Navigation Sidebar
page = st.sidebar.radio("Navigate Prototype", ["Interactive Diagnostic Tool", "Model Evaluation & Metrics"])

# Error fallback if model isn't trained yet
if pipeline is None:
    st.error(f"❌ Core pipeline artifact '{MODEL_PATH}' was not found in this directory.")
    st.info("Please ensure your model pkl file is placed in this exact folder location.")
    st.stop()

# ==============================================================================
# VIEW 1: INTERACTIVE DIAGNOSTIC SCREEN
# ==============================================================================
if page == "Interactive Diagnostic Tool":
    st.subheader("📋 Patient Behavioral & Lifestyle Assessment Panel")
    st.write("Input a high-schooler's daily metrics below to evaluate potential risk indicators using your group's optimized machine learning pipeline.")

    with st.form("assessment_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 👤 Demographic & Social Context")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.slider("Age (Years)", min_value=10, max_value=21, value=15)
            platform_usage = st.selectbox("Primary Social Media Platform", ["TikTok", "Instagram", "YouTube", "Twitter", "Facebook", "Snapchat", "None"])
            social_interaction_level = st.selectbox("Real-life Social Interaction Level", ["Low", "Medium", "High"])

        with col2:
            st.markdown("##### ⏱️ Screen Time & Sleep Factors")
            daily_social_media_hours = st.number_input("Daily Social Media Activity (Hours)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            sleep_hours = st.number_input("Average Nightly Sleep Duration (Hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            screen_time_before_sleep = st.number_input("Screen Time Prior to Sleep (Minutes)", min_value=0, max_value=360, value=60, step=10)

        st.markdown("---")
        st.markdown("##### 📊 Self-Reported Psychological Severity Scale (1 - 10)")
        col3, col4, col5 = st.columns(3)
        with col3:
            stress_level = st.slider("Perceived Stress Level", 1, 10, 5)
            academic_performance = st.slider("Academic Performance Perception", 1, 10, 6)
        with col4:
            anxiety_level = st.slider("Anxiety Severity Level", 1, 10, 5)
            physical_activity = st.slider("Physical Activity Frequency", 1, 10, 4)
        with col5:
            addiction_level = st.slider("Social Media Addiction Tendency", 1, 10, 4)

        # Form Submit Button
        submit_btn = st.form_submit_button("Execute Automated Triage Scan", type="primary")

    if submit_btn:
        patient_record = {
            'gender': gender,
            'platform_usage': platform_usage,
            'social_interaction_level': social_interaction_level,
            'age': age,
            'daily_social_media_hours': daily_social_media_hours,
            'sleep_hours': sleep_hours,
            'screen_time_before_sleep': screen_time_before_sleep,
            'academic_performance': academic_performance,
            'physical_activity': physical_activity,
            'stress_level': stress_level,
            'anxiety_level': anxiety_level,
            'addiction_level': addiction_level
        }

        patient_df = pd.DataFrame([patient_record])

        try:
            prediction = pipeline.predict(patient_df)[0]
            st.markdown("### 📊 Automated Assessment Output")
            is_at_risk = str(prediction) == '1' or prediction == 1 or str(prediction).lower() == 'depressed'

            if is_at_risk:
                st.error("⚠️ **High Risk Profile Detected**")
                st.markdown("""
                **System Indication:** This student's behavioral vectors reflect a statistical risk matching profiles with active depressive trends. 
                *Preemptive counselor outreach is strongly recommended.*
                """)
            else:
                st.success("✅ **Low Risk / Stable Profile**")
                st.markdown("""
                **System Indication:** Lifestyle indicators maps inside healthy reference parameters. No immediate critical clinical flags are flagged.
                """)

            if hasattr(pipeline, "predict_proba"):
                risk_probabilities = pipeline.predict_proba(patient_df)[0]
                calculated_risk = risk_probabilities[1] if len(risk_probabilities) > 1 else risk_probabilities[0]
                st.write(f"**Algorithmic Risk Probability Score:** {calculated_risk * 100:.2f}%")
                st.progress(float(calculated_risk))

        except Exception as error:
            st.error(f"Error executing prediction sequence: {error}")

# ==============================================================================
# VIEW 2: MODEL EVALUATION PANEL
# ==============================================================================
elif page == "Model Evaluation & Metrics":
    st.subheader("📈 Scientific Proof & Backend Performance Framework")
    st.write("This tab serves as the empirical evaluation layer proving system validity against the dataset constraints.")
    st.info("🔬 **Data Balancing Engineering:** To address the critical 97.5% class imbalance threat (234 healthy cases vs. 6 depressed cases), this prototype embeds an integrated **SMOTE (Synthetic Minority Over-sampling Technique)** upsampling function dynamically inside a cross-validated processing pipeline.")

    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.markdown("#### 🔢 Confusion Matrix Heatmap")
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, use_column_width=True) # <-- Fixed argument name here
        else:
            st.warning("Heatmap chart graphic file missing ('confusion_matrix.png').")

    with vis_col2:
        st.markdown("#### 📊 Mathematical Feature Weights")
        if os.path.exists(FI_PATH):
            st.image(FI_PATH, use_column_width=True) # <-- Fixed argument name here
        else:
            st.warning("Feature bar chart graphic file missing ('feature_importance.png').")
