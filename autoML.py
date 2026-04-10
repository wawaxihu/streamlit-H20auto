
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h2o

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Colorectal Polyp ESD Pathology Upgrade Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem; font-weight: 700;
        color: #1a4f8a; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.9rem; color: #888;
        text-align: center; margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem; border-radius: 10px;
        font-size: 1.05rem; font-weight: 600; text-align: center;
    }
    .high-risk { background:#fdecea; color:#c0392b; border:1.5px solid #e74c3c; }
    .low-risk  { background:#eafaf1; color:#1e8449; border:1.5px solid #27ae60; }
    .stButton>button {
        width:100%; background:#1a4f8a; color:white;
        font-size:1rem; font-weight:600;
        border-radius:8px; padding:0.55rem; border:none;
    }
    .stButton>button:hover { background:#1565c0; }
</style>
""", unsafe_allow_html=True)

# ── Load Model (cached) ──────────────────────────────────────
import os
@st.cache_resource
def load_model():
    h2o.init(max_mem_size="1g", nthreads=1, verbose=False)

    # 用绝对路径，兼容本地和 Streamlit Cloud
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "GLM_1_AutoML_2_20260408_210958")

    return h2o.load_model(model_path)

model = load_model()

# ── Page Header ──────────────────────────────────────────────
st.markdown(
    '<div class="main-title">'
    '🔬 Pathology Upgrade Risk Prediction after Colorectal Polyp ESD'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Powered by H2O AutoML · GLM &nbsp;|&nbsp; '
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar Inputs ───────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Feature Input")
    st.caption("Please fill in based on endoscopic and clinical findings.")

    st.subheader("📏 Clinical Metrics")
    size = st.number_input("Lesion Size (mm)", min_value=1,   max_value=200, value=15, step=1)
    bmi  = st.number_input("BMI (kg/m²)",      min_value=15.0, max_value=45.0,
                           value=23.0, step=0.1, format="%.1f")

    st.subheader("🔭 Endoscopic Features")
    number  = st.selectbox("Number of Lesions", ["Single", "Multiple"])
    villous = st.selectbox("Villous Structure",  ["Without", "With"])
    erosion = st.selectbox("Erosion",            ["No", "Yes"])

    st.subheader("🧪 Laboratory Findings")
    fob = st.selectbox("Fecal Occult Blood (FOB)", ["Negative", "Positive"])

    st.divider()
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

# ── Main Layout ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

input_data = {
    "Size": size, "BMI": bmi,
    "Number": number, "FOB": fob,
    "Villous": villous, "Erosion": erosion,
}
df_input = pd.DataFrame([input_data])

with col_left:
    st.subheader("🧾 Current Input Features")
    st.dataframe(df_input, use_container_width=True, hide_index=True)
    st.caption("Please confirm the information, then click [Run Prediction] on the left.")

with col_right:
    st.subheader("📊 Prediction Result")
    if not predict_btn:
        st.info("Fill in patient information on the left and click [Run Prediction] to see results.")
    else:
        with st.spinner("Running inference…"):
            h2o_df = h2o.H2OFrame(df_input)
            for col in ["Number", "FOB", "Villous", "Erosion"]:
                h2o_df[col] = h2o_df[col].asfactor()
            pred_df = model.predict(h2o_df).as_data_frame()

        # Parse probability
        prob_cols  = [c for c in pred_df.columns if c.startswith("p")]
        prob_pos   = float(pred_df[prob_cols[-1]].iloc[0])
        pred_label = pred_df["predict"].iloc[0]

        # Risk stratification
        if prob_pos >= 0.5:
            risk_cls, risk_text, risk_icon = "high-risk", "High Risk of Upgrade", "🔴"
        else:
            risk_cls, risk_text, risk_icon = "low-risk",  "Low Risk of Upgrade",  "🟢"

        st.markdown(
            f'<div class="result-box {risk_cls}">'
            f'{risk_icon} Prediction: <b>{pred_label}</b> &nbsp;|&nbsp;'
            f' Upgrade Probability: <b>{prob_pos:.1%}</b> &nbsp;|&nbsp;'
            f' Risk: <b>{risk_text}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(" ")
        st.caption("Predicted probability of pathology upgrade")
        st.progress(prob_pos)

        with st.expander("View full prediction probability table"):
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── SHAP Individual Explanation ──────────────────────────────
if predict_btn:
    st.divider()
    st.subheader("🔍 SHAP Feature Contribution")
    with st.spinner("Computing SHAP values…"):
        try:
            # Step 1：加载背景数据（提前保存的训练集小样本）
            # 首次部署时先在Jupyter里执行一次：
            #   train.as_data_frame().sample(100, random_state=42).to_csv("models/background.csv", index=False)
            bg_df   = pd.read_csv("models/background.csv")
            bg_h2o  = h2o.H2OFrame(bg_df)
            for col in ["Number", "FOB", "Villous", "Erosion"]:
                bg_h2o[col] = bg_h2o[col].asfactor()

            # Step 2：计算当前输入行的SHAP贡献值
            contrib = model.predict_contributions(
                h2o_df,
                background_frame=bg_h2o        # ← 用真实训练集背景
            ).as_data_frame()

            # Step 3：合并dummy列回原始变量名
            bias_col   = "BiasTerm"
            shap_vals  = contrib.drop(columns=[bias_col]).values[0]
            feat_names = [c for c in contrib.columns if c != bias_col]

            original_feats = list(input_data.keys())
            merged_vals, merged_names = [], []
            for orig in original_feats:
                cols = [c for c in feat_names
                        if c == orig or c.startswith(orig + ".")]
                if cols:
                    idx_list = [feat_names.index(c) for c in cols]
                    merged_vals.append(sum(shap_vals[i] for i in idx_list))
                    merged_names.append(orig)

            merged_arr = np.array(merged_vals)
            order      = np.argsort(np.abs(merged_arr))
            colors     = ["#e74c3c" if v > 0 else "#2ecc71" for v in merged_arr]

            # Step 4：绘图
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(
                [merged_names[i] for i in order],
                [merged_arr[i]   for i in order],
                color=[colors[i] for i in order],
                edgecolor="white", height=0.6,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(
                "SHAP Value  (positive → increases upgrade risk,"
                "  negative → decreases upgrade risk)",
                fontsize=8,
            )
            ax.set_title("Feature Contributions to This Prediction",
                         fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)      # ← 释放内存，避免Streamlit重复渲染旧图
            st.caption("🔴 Red: increases pathology upgrade risk　　"
                       "🟢 Green: decreases pathology upgrade risk")

        except Exception as e:
            st.warning(f"SHAP computation failed ({e}).")
# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is intended for clinical decision support only. "
    "Final diagnosis must be made by a qualified clinician based on the full clinical context. "
    "This does not constitute medical advice."
)