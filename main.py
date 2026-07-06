import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# =====================
# 0. Global Page Configuration
# =====================
st.set_page_config(
    page_title="GMO: A General Design Framework for Multi-Objective High-Performance Magnesium Alloys",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for spacing adjustments (optional)
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Global Main Title
st.title("🧪 GMO: A General Design Framework for Multi-Objective High-Performance Magnesium Alloys")
st.caption("Machine Learning Driven Alloy Design")

# =====================
# Left Sidebar (System Overview)
# =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103339.png", width=100)  # Placeholder Logo
    st.header("System Overview")
    st.info(
        "Integrates multiple machine learning algorithms such as XGBoost, Gradient Boosting, "
        "and Random Forest. Combined with physical/chemical descriptor extraction, "
        "it provides rapid guidance for the design and performance evaluation of new alloys."
    )
    st.markdown("---")
    st.markdown("### 📌 Supported Prediction Tasks:\n"
                "- Hardness (HV)\n"
                "- Corrosion Potential (ECORR)\n"
                "- Tensile Strength (UTS)\n"
                "- Elongation (EL)")
    st.markdown("---")
    st.caption("Version: v1.1 | 2026")

# =====================
# 1. Define Prediction Tasks & Required Features
# =====================
TARGET_FEATURES = {
    "Hardness (HV)": ['AF9-2', 'AF23-1'],
    "Corrosion Potential (ECORR)": ['AF13-2', 'AF26-2', 'AF7-2', 'AF57-2'],
    "Tensile Strength (TS)": ['AF58-2', 'AF20-1', 'AF25-2', 'AF22-1', 'AF64-1', 'AF28-1', 'AF51-2', 'AF26-1', 'AF1-2'],
    "Elongation (EL)": ['AF2-1', 'AF8-2', 'AF26-1', 'AF61-2', 'AF21-2']
}

elements = [
    'Mg', 'Al', 'Zn', 'Y', 'Zr',
    'Ca', 'Mn', 'Li', 'Sn', 'Nd',
    'Gd', 'La', 'Si', 'Sm', 'Ce',
    'Er', 'Cu', 'Sr', 'Bi', 'Sb',
    'Pb', 'Ni', 'Fe'
]

# Database file mapping
DATABASE_FILES = {
    "Hardness Database (HV)": {"file": "hvdata.xlsx"},
    "Corrosion Potential Database (ECORR)": {"file": "ecorrdata.xlsx"},
    "Tensile Strength Database (UTS)": {"file": "tsdata.xlsx"},
    "Elongation Database (EL)": {"file": "eldata.xlsx"}
}

# =====================
# 2. Cached Data Loading Functions
# =====================
@st.cache_data
def load_pro_data():
    pro = pd.read_excel("PRO.xlsx")
    pro = pro.iloc[:, 1:]
    return pro

@st.cache_data
def load_database(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            st.error(f"⚠️ Error reading {file_path}: {e}")
            return None
    else:
        return None

# Load base PRO data
try:
    pro = load_pro_data()
    column_names = pro.columns.tolist()
except Exception as e:
    st.error(f"Failed to load PRO.xlsx. Please ensure the file exists. Error: {e}")
    st.stop()

# =====================
# 3. Core Page Navigation (Tabs)
# =====================
tab_predict, tab_database = st.tabs(["🚀 Intelligent Prediction Panel", "📊 Experimental Database Center"])

# ==============================================================================
# TAB 1: Performance Prediction System
# ==============================================================================
with tab_predict:
    # Top Control Area
    col_sel, col_status = st.columns([2, 1])
    with col_sel:
        target = st.selectbox("🎯 Select the alloy property to predict:", list(TARGET_FEATURES.keys()))

        # Model file path configuration
        model_files = {
            "Hardness (HV)": ("standard_scalerHV.pkl", "xgboost_modelHV.pkl"),
            "Corrosion Potential (ECORR)": ("gbrecorr.joblib", "gbrecorr.joblib"), # Pipeline extraction
            "Tensile Strength (UTS)": ("standard_scalerUTS.pkl", "gbr_modelUTS.pkl"),
            "Elongation (EL)": ("standard_scalerEL.pkl", "rf_modelEL.pkl")
        }
        scaler_file, model_file = model_files[target]

        # Display model loading status
        with col_status:
            st.write(" ")  # Placeholder for alignment
            st.write(" ")
            try:
                if target == "Corrosion Potential (ECORR)":
                    # Unpack pipeline dictionary for ECORR
                    pipeline = joblib.load(model_file)
                    model = pipeline['model']  # Extract GBR model
                    scaler = pipeline['scaler']  # Extract standard scaler
                else:
                    # Traditional separate loading approach
                    scaler = joblib.load(scaler_file)
                    model = joblib.load(model_file)

                st.success("✅ Model successfully connected", icon="🟢")
            except FileNotFoundError:
                st.error("❌ Model file missing", icon="🔴")
                st.toast(f"Model file for {target} not found!", icon="⚠️")
                model, scaler = None, None

    st.divider()

    # Element Input Area (Card Design)
    with st.container(border=True):
        st.subheader("🧪 Alloy Element Composition Input (at.%)")
        st.caption("Enter the mass fraction (or atomic percentage, consistent with training data) of 23 elements:")
        values = []
        cols = st.columns(6)
        for i, ele in enumerate(elements):
            with cols[i % 6]:
                value = st.number_input(ele, value=0.0, step=0.1, format="%.4f", key=f"predict_{ele}")
                values.append(value)

    # Special Environmental Parameters for ECORR
    cl_ion_value = 0.0
    if target == "Corrosion Potential (ECORR)":
        with st.container(border=True):
            st.subheader("🌊 Environmental Parameters")
            cl_ion_value = st.number_input("Chloride Ion Concentration / Cl ion (mol·L⁻¹)", value=0.613, step=0.01, format="%.4f")

    # Prediction Button Area
    st.write("")  # Vertical spacing
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        start_predict = st.button("🚀 START INTELLIGENT PREDICTION", type="primary", use_container_width=True)

    if start_predict:
        if model is None or scaler is None:
            st.error("Unable to predict: Please ensure the corresponding model files exist in the server or local directory.")
        else:
            with st.spinner("🧠 Extracting physical/chemical descriptors and reasoning..."):
                time.sleep(0.5)  # Simulate calculation delay for better UX
                X = pd.DataFrame([values], columns=elements)

                # Type 1 Features (X1)
                X1 = np.dot(X, pro) / 100
                X1 = pd.DataFrame(X1, columns=column_names)
                X1.columns = [c + "-1" for c in column_names]

                # Type 2 Features (X2)
                f1 = np.square(pro - pd.DataFrame(np.tile(X1.iloc[0], (23, 1)), columns=column_names))
                X2 = np.dot(X.iloc[0], f1.values) / 100
                X2 = pd.DataFrame([X2], columns=column_names)
                X2.columns = [c + "-2" for c in column_names]

                # Merge feature pool and extract required descriptors
                feature_pool = pd.concat([X1, X2], axis=1)
                columns_to_keep = TARGET_FEATURES[target]
                feature = feature_pool[columns_to_keep].copy()

                # Dynamic Feature Alignment Mechanism
                expected_features = None
                if hasattr(scaler, "feature_names_in_"):
                    expected_features = scaler.feature_names_in_
                elif hasattr(model, "feature_names_in_"):
                    expected_features = model.feature_names_in_

                if expected_features is not None:
                    if target == "Corrosion Potential (ECORR)":
                        feature["cl_ion_temporary_column"] = cl_ion_value
                        for col in expected_features:
                            if col not in TARGET_FEATURES[target]:
                                feature = feature.rename(columns={"cl_ion_temporary_column": col})
                                break
                    feature = feature[expected_features]
                else:
                    if target == "Corrosion Potential (ECORR)":
                        feature["Cl ion (mol·L-1)"] = cl_ion_value

                st.divider()

                # Display Extracted Features
                with st.container(border=True):
                    st.subheader("💡 Extracted Descriptors")
                    num_features = feature.shape[1]
                    metric_cols = st.columns(min(num_features, 5))
                    for i, col_name in enumerate(feature.columns):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(label=col_name, value=f"{feature.iloc[0][col_name]:.6f}")

                # Model Prediction & Display
                try:
                    feature_scaled = scaler.transform(feature)
                    pred = model.predict(feature_scaled)[0]

                    # Highlight Result
                    st.success(f"### 🎯 Prediction Complete!\n#### Predicted value for 【 {target} 】: **`{pred:.4f}`**")

                    # Display classification probabilities if applicable
                    if hasattr(model, "predict_proba"):
                        with st.expander("📊 View Classification Probability Distribution"):
                            prob = model.predict_proba(feature_scaled)[0]
                            df_prob = pd.DataFrame({"Category": range(len(prob)), "Probability": prob})
                            st.bar_chart(df_prob.set_index("Category"), use_container_width=True)
                except Exception as e:
                    st.error(f"Error during prediction (possible feature dimension mismatch with the model): {e}")

# ==============================================================================
# TAB 2: Experimental Database Center
# ==============================================================================
with tab_database:
    st.subheader("Experimental Data Management & Visualization ")
    st.caption("Browse raw experimental data and analyze the trend impact of individual elements on final performance.")

    # Database Selector
    db_col1, db_col2 = st.columns([1, 2])
    with db_col1:
        selected_db_name = st.selectbox("📂 Switch Database View:", list(DATABASE_FILES.keys()))
    db_info = DATABASE_FILES[selected_db_name]

    # Load selected data
    df_db = load_database(db_info["file"])

    if df_db is not None:
        # 1. Data Overview
        with st.container(border=True):
            st.markdown(f"**Data Matrix Preview** (Total `{df_db.shape[0]}` samples, (wt.%) `{df_db.shape[1]}` features)")
            st.dataframe(df_db, use_container_width=True, hide_index=True, height=300)

        # 2. Summary Statistics (Placed in an expander for clean UI)
        with st.expander("📈 View Summary Statistics"):
            st.caption("Displays mean, standard deviation, min/max values, etc. for each component:")
            st.dataframe(df_db.describe(), use_container_width=True)

        # 3. Dynamic Data Visualization
        st.divider()
        st.subheader("🔍 Element Impact Trend Exploration")

        available_elements = [ele for ele in elements if ele in df_db.columns]
        target_col = df_db.columns[-1]

        if available_elements:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write("")  # Vertical alignment helper
                plot_element = st.selectbox(
                    "📌 Select element to analyze (X-axis):",
                    available_elements,
                    index=min(1, len(available_elements) - 1)
                )
                st.info(f"Plotting relationship: **{plot_element}** (wt.%) vs **{target_col}**.")

            with c2:
                chart_data = df_db[[plot_element, target_col]].dropna()
                try:
                    chart_data[target_col] = pd.to_numeric(chart_data[target_col])
                    st.scatter_chart(
                        data=chart_data,
                        x=plot_element,
                        y=target_col,
                        use_container_width=True
                    )
                except ValueError:
                    st.warning(f"⚠️ Unable to plot. Column '{target_col}' contains non-numeric data.")
        else:
            st.warning("No preset alloy element columns detected in the current database.")

    else:
        st.error(f"❌ Failed to load data. Please ensure `{db_info['file']}` exists in the current directory.")