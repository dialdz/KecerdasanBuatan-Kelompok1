import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import MODEL_PATH, PREPROCESSOR_PATH, PredictiveMaintenanceMLP


DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
)

st.title("Tugas Besar Kecerdasan Buatan - Predictive Maintenance")
st.caption("Kelompok 1 | Minggu 12: Data, EDA, Baseline, dan kerangka dashboard")

if not DATA_PATH.exists():
    st.error(f"Dataset tidak ditemukan pada path: {DATA_PATH}")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    input_dim = len(preprocessor.get_feature_names_out())
    model = PredictiveMaintenanceMLP(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, preprocessor


df = load_data(DATA_PATH)

menu = st.sidebar.selectbox(
    "Menu",
    ["Overview", "Dataset Explorer", "EDA", "Baseline Model", "Prediction"],
)

if menu == "Overview":
    st.subheader("Overview Proyek")
    st.markdown(
        """
        - Topik: Predictive maintenance untuk mesin industri.
        - Dataset: AI4I 2020 Predictive Maintenance Dataset.
        - Target: `Machine failure`.
        - Fokus minggu ini: validasi target, EDA, split aman, baseline NN, dan kerangka dashboard.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    col2.metric("Jumlah Kolom", f"{df.shape[1]:,}")
    col3.metric("Target Failure", f"{df['Machine failure'].sum():,}")

    st.info("Catatan: Saat ini dashboard masih berupa skeleton awal yang siap dikembangkan menuju GA/PSO/Fuzzy dan deployment pada minggu berikutnya.")

elif menu == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    st.write("Statistik deskriptif")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    st.write("Missing values per kolom")
    st.dataframe(df.isnull().sum().to_frame(name="missing_values"), use_container_width=True)

elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")
    target_counts = df["Machine failure"].value_counts().sort_index()
    st.bar_chart(target_counts)

    st.write("Distribusi target")
    st.dataframe(target_counts.rename_axis("class").reset_index(name="count"), use_container_width=True)

    st.write("Type distribution")
    st.dataframe(df["Type"].value_counts().rename_axis("type").reset_index(name="count"), use_container_width=True)

elif menu == "Baseline Model":
    st.subheader("Baseline Model")
    st.markdown(
        """
        Model baseline berupa MLP (Multilayer Perceptron) dengan:
        - preprocessing numerik dan kategorikal
        - train/validation/test split aman
        - metrik klasifikasi: accuracy, precision, recall, F1-score, ROC-AUC
        """
    )
    st.code(
        """
        X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        y = df['Machine failure']
        """,
        language="python",
    )

elif menu == "Prediction":
    st.subheader("Prediksi Machine Failure")
    st.write("Masukkan kondisi mesin untuk mendapatkan prediksi dari baseline neural network.")
    with st.form("prediction_form"):
        air_temp = st.number_input("Air temperature [K]", value=298.0)
        process_temp = st.number_input("Process temperature [K]", value=308.0)
        rotational_speed = st.number_input("Rotational speed [rpm]", value=1551.0)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool wear [min]", value=200.0)
        machine_type = st.selectbox(
            "Type", sorted(df["Type"].dropna().unique().tolist())
        )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            st.error(
                "Model atau preprocessing artifact belum tersedia. "
                "Jalankan `python src/model.py` dari folder utama repo terlebih dahulu."
            )
        else:
            model, preprocessor = load_model_artifacts()
            input_data = pd.DataFrame(
                [
                    {
                        "Type": machine_type,
                        "Air temperature [K]": air_temp,
                        "Process temperature [K]": process_temp,
                        "Rotational speed [rpm]": rotational_speed,
                        "Torque [Nm]": torque,
                        "Tool wear [min]": tool_wear,
                    }
                ]
            )
            processed_input = preprocessor.transform(input_data)
            input_tensor = torch.tensor(processed_input, dtype=torch.float32)

            with torch.no_grad():
                failure_probability = model(input_tensor).item()

            predicted_failure = failure_probability >= 0.5
            if predicted_failure:
                st.error("Prediksi: berpotensi mengalami machine failure")
            else:
                st.success("Prediksi: tidak terindikasi machine failure")

            st.metric("Probabilitas machine failure", f"{failure_probability:.1%}")
            st.caption(
                "Probabilitas ini berasal dari baseline NN dengan ambang klasifikasi 0,5; "
                "belum merupakan keputusan Fuzzy Logic."
            )
