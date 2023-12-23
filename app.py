import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.classification import setup, compare_models, pull, save_model, load_model


if os.path.exists("dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
    st.title("AutoML")
    choice = st.radio("Select the task", ["Upload", "Profiling", "Modeling", "Download"])
    st.info("This project application helps you build and explore your data")
    
    
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile = ydata_profiling.ProfileReport(df, explorative=True)
    st_profile_report(profile)

if choice == "Modeling":
    chosen_target = st.selectbox("Select the target column", df.columns)
    if st.button("Run Modelling"):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "trained_model")

if choice == "Download":
    with open('trained_model.pkl', 'rb') as f:
        st.download_button('Download trained model', f, file_name='trained_model.pkl')