import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Config
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
try:
    df = pd.read_csv("cleaned_titanic.csv")
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_titanic.csv' exists.")
    st.stop()

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender_options = ["All"] + df["Sex"].unique().tolist()
gender = st.sidebar.selectbox("Select Gender", options=gender_options)

pclass_options = ["All"] + sorted(df["Pclass"].unique().tolist())
pclass = st.sidebar.selectbox("Select Passenger Class", options=pclass_options)

# Apply Filters
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]
if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

# Preview Filtered Data
st.subheader("🔍 Filtered Data Preview")
st.write(filtered_df.head())

# Map Survival
filtered_df["Survival Status"] = filtered_df["Survived"].map({0: "Did No
