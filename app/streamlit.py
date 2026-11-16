import streamlit as st
st.title("Tugas Besar Predictive Maintenance System Kelompok A1")
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Dataset", "EDA", "Prediction", "Evaluation"]
)

if menu == "Home":
    st.write("Dashboard")

elif menu == "Dataset":
    st.write("Informasi Dataset")

elif menu == "EDA":
    st.write("Exploratory Data Analysis")

elif menu == "Prediction":
    st.write("Prediction")

elif menu == "Evaluation": 
    st.write("Evaluation")